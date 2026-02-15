/// @file matmul_tiled64_kernel.metal
/// @brief High-performance 64x64 tiled matrix multiplication kernels for Apple Silicon.
///
/// Optimizations:
/// - 64x64 output tiles (vs 32x32) for 2x arithmetic intensity (AI=32 FLOPs/byte)
/// - Double-buffered shared memory (reduces barriers by 50%)
/// - 4x4 register tiling per SIMD group (16 accumulators of 8x8)
/// - Direct device memory writes (eliminates threadgroup float intermediate)
/// - 8 SIMD groups (256 threads) vs 4 SIMD groups (128 threads)
///
/// Performance target: 1.3x+ speedup over 32x32 kernel at M=N=K=4096 FP16

#include <metal_stdlib>
#include "common/metal_types.h"
using namespace metal;

// Tile configuration for 64x64 output
constant constexpr uint TILE_M = 64;  // rows per threadgroup
constant constexpr uint TILE_N = 64;  // cols per threadgroup
constant constexpr uint TILE_K = 32;  // K dimension tiling

// Threadgroup configuration
// 256 threads = 8 SIMD groups of 32 threads
// SIMD group layout: 4x2 grid
//   Each SIMD group computes 16x32 block as 4x4 grid of 8x8 accumulators
//   Total: 64x64 output tile

// Threadgroup memory budget (FP16):
//   shared_A[2]: 2 * 64 * 32 * 2 = 8,192 bytes (double-buffered)
//   shared_B[2]: 2 * 32 * 64 * 2 = 8,192 bytes (double-buffered)
//   Total: 16,384 bytes (50% of 32KB limit)

// ============================================================================
// Helper: Load tile into double-buffered shared memory
// ============================================================================

template<typename T>
inline void load_tile_A(
    device const T* A,
    threadgroup T* shared_A,
    uint row_base,
    uint k_tile,
    uint M, uint K, uint lda,
    bool transpose_a,
    uint tid  // global thread ID in threadgroup (0-255)
) {
    // 256 threads load 64*32 = 2048 elements
    // Each thread loads 2048/256 = 8 elements
    for (uint idx = tid; idx < TILE_M * TILE_K; idx += 256) {
        uint r = idx / TILE_K;
        uint c = idx % TILE_K;
        uint global_r = row_base + r;
        uint global_c = k_tile + c;
        if (global_r < M && global_c < K) {
            if (transpose_a) {
                shared_A[r * TILE_K + c] = A[global_c * lda + global_r];
            } else {
                shared_A[r * TILE_K + c] = A[global_r * lda + global_c];
            }
        } else {
            shared_A[r * TILE_K + c] = T(0.0f);
        }
    }
}

template<typename T>
inline void load_tile_B(
    device const T* B,
    threadgroup T* shared_B,
    uint col_base,
    uint k_tile,
    uint N, uint K, uint ldb,
    bool transpose_b,
    uint tid
) {
    // 256 threads load 32*64 = 2048 elements
    for (uint idx = tid; idx < TILE_K * TILE_N; idx += 256) {
        uint r = idx / TILE_N;
        uint c = idx % TILE_N;
        uint global_r = k_tile + r;
        uint global_c = col_base + c;
        if (global_r < K && global_c < N) {
            if (transpose_b) {
                shared_B[r * TILE_N + c] = B[global_c * ldb + global_r];
            } else {
                shared_B[r * TILE_N + c] = B[global_r * ldb + global_c];
            }
        } else {
            shared_B[r * TILE_N + c] = T(0.0f);
        }
    }
}

// ============================================================================
// FP16 kernel with FP32 accumulation and direct device write
// ============================================================================

kernel void matmul_tiled64_fp16(
    device const half* A      [[buffer(0)]],
    device const half* B      [[buffer(1)]],
    device half* C            [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant uint& lda        [[buffer(6)]],
    constant uint& ldb        [[buffer(7)]],
    constant uint& ldc        [[buffer(8)]],
    constant bool& transpose_a [[buffer(9)]],
    constant bool& transpose_b [[buffer(10)]],
    threadgroup half* shared_A_buf [[threadgroup(0)]],  // [2][TILE_M, TILE_K]
    threadgroup half* shared_B_buf [[threadgroup(1)]],  // [2][TILE_K, TILE_N]
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint row_base = gid.y * TILE_M;
    const uint col_base = gid.x * TILE_N;

    // 8 SIMD groups arranged in 4x2 grid
    // Each SIMD group computes 16x32 block (4 rows of 8x8 x 4 cols of 8x8)
    //   sg0: rows[0:16],   cols[0:32]   -> 4x4 of 8x8
    //   sg1: rows[0:16],   cols[32:64]  -> 4x4 of 8x8
    //   sg2: rows[16:32],  cols[0:32]   -> 4x4 of 8x8
    //   sg3: rows[16:32],  cols[32:64]  -> 4x4 of 8x8
    //   sg4: rows[32:48],  cols[0:32]   -> 4x4 of 8x8
    //   sg5: rows[32:48],  cols[32:64]  -> 4x4 of 8x8
    //   sg6: rows[48:64],  cols[0:32]   -> 4x4 of 8x8
    //   sg7: rows[48:64],  cols[32:64]  -> 4x4 of 8x8

    uint sg_row = (simd_group_id / 2) * 16;  // 0, 0, 16, 16, 32, 32, 48, 48
    uint sg_col = (simd_group_id % 2) * 32;  // 0, 32, 0, 32, 0, 32, 0, 32

    // 4x4 grid of 8x8 accumulators per SIMD group (16 total)
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }

    // Double-buffered shared memory pointers
    threadgroup half* shared_A[2] = {
        shared_A_buf,
        shared_A_buf + TILE_M * TILE_K
    };
    threadgroup half* shared_B[2] = {
        shared_B_buf,
        shared_B_buf + TILE_K * TILE_N
    };

    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Load first tile into buffer 0
    load_tile_A(A, shared_A[0], row_base, 0, M, K, lda, transpose_a, tid);
    load_tile_B(B, shared_B[0], col_base, 0, N, K, ldb, transpose_b, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main K-tile loop with double buffering
    for (uint k_tile_idx = 0; k_tile_idx < num_k_tiles; ++k_tile_idx) {
        uint buf_idx = k_tile_idx % 2;
        uint next_buf_idx = (k_tile_idx + 1) % 2;

        // Prefetch next tile while computing on current tile (overlap)
        if (k_tile_idx + 1 < num_k_tiles) {
            uint next_k_tile = (k_tile_idx + 1) * TILE_K;
            load_tile_A(A, shared_A[next_buf_idx], row_base, next_k_tile, M, K, lda, transpose_a, tid);
            load_tile_B(B, shared_B[next_buf_idx], col_base, next_k_tile, N, K, ldb, transpose_b, tid);
        }

        // Compute on current buffer
        // Each SIMD group processes its 16x32 block
        for (uint k = 0; k < TILE_K; k += 8) {
            // Load 4 A fragments (covering 4 rows of 8x8)
            simdgroup_half8x8 a_frag[4];
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(a_frag[i],
                    shared_A[buf_idx] + (sg_row + i * 8) * TILE_K + k,
                    TILE_K);
            }

            // Load 4 B fragments (covering 4 cols of 8x8)
            simdgroup_half8x8 b_frag[4];
            for (uint j = 0; j < 4; ++j) {
                simdgroup_load(b_frag[j],
                    shared_B[buf_idx] + k * TILE_N + (sg_col + j * 8),
                    TILE_N);
            }

            // Compute 4x4 outer product (16 multiply-accumulates)
            for (uint i = 0; i < 4; ++i) {
                for (uint j = 0; j < 4; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        // Barrier only if we need to load next tile
        if (k_tile_idx + 1 < num_k_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Batched FP32→FP16 conversion: all 16 accumulators at once
    // Each SIMD group owns a 16x32 region. We use shared_A_buf as scratch
    // (no longer needed after K-loop). Each SIMD group gets its own 16*32=512
    // float region for FP32 store, then converts in bulk with only 2 barriers.
    //
    // Layout in scratch: simd_group_id * (16 * 32) floats
    // Each accumulator acc[i][j] is 8x8, stored at offset (i*8)*32 + (j*8)

    // Reuse shared_A_buf as scratch (capacity: 2 * TILE_M * TILE_K = 2*64*32 = 4096 halves = 2048 floats)
    // We need 8 SIMD groups * 16*32 = 4096 floats → use both shared_A_buf and shared_B_buf as float scratch
    // shared_A_buf: SIMD groups 0-3 (4 * 512 = 2048 floats = 8192 bytes, fits in 2*64*32*2=8192 bytes)
    // shared_B_buf: SIMD groups 4-7 (4 * 512 = 2048 floats = 8192 bytes, fits in 2*32*64*2=8192 bytes)
    threadgroup float* sg_scratch;
    if (simd_group_id < 4) {
        sg_scratch = reinterpret_cast<threadgroup float*>(shared_A_buf) + simd_group_id * (16 * 32);
    } else {
        sg_scratch = reinterpret_cast<threadgroup float*>(shared_B_buf) + (simd_group_id - 4) * (16 * 32);
    }

    // Step 1: Store all 16 FP32 accumulators to threadgroup scratch (no barriers needed between stores)
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            simdgroup_store(acc[i][j], sg_scratch + (i * 8) * 32 + (j * 8), 32);
        }
    }

    // Step 2: Single barrier to ensure all stores complete
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Bulk FP32→FP16 convert and write directly to device memory
    // Each thread in the SIMD group converts elements from the 16x32 scratch block
    // 16*32 = 512 elements, 32 threads → 16 elements per thread
    for (uint t = simd_lane_id; t < 16 * 32; t += 32) {
        uint lr = t / 32;  // row within 16x32 block
        uint lc = t % 32;  // col within 16x32 block
        uint global_r = row_base + sg_row + lr;
        uint global_c = col_base + sg_col + lc;
        if (global_r < M && global_c < N) {
            C[global_r * ldc + global_c] = half(sg_scratch[lr * 32 + lc]);
        }
    }
}

// ============================================================================
// FP32 kernel (full precision throughout)
// ============================================================================

kernel void matmul_tiled64_fp32(
    device const float* A      [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& lda         [[buffer(6)]],
    constant uint& ldb         [[buffer(7)]],
    constant uint& ldc         [[buffer(8)]],
    constant bool& transpose_a [[buffer(9)]],
    constant bool& transpose_b [[buffer(10)]],
    threadgroup float* shared_A_buf [[threadgroup(0)]],
    threadgroup float* shared_B_buf [[threadgroup(1)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint row_base = gid.y * TILE_M;
    const uint col_base = gid.x * TILE_N;

    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 32;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }

    threadgroup float* shared_A[2] = {
        shared_A_buf,
        shared_A_buf + TILE_M * TILE_K
    };
    threadgroup float* shared_B[2] = {
        shared_B_buf,
        shared_B_buf + TILE_K * TILE_N
    };

    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    load_tile_A(A, shared_A[0], row_base, 0, M, K, lda, transpose_a, tid);
    load_tile_B(B, shared_B[0], col_base, 0, N, K, ldb, transpose_b, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k_tile_idx = 0; k_tile_idx < num_k_tiles; ++k_tile_idx) {
        uint buf_idx = k_tile_idx % 2;
        uint next_buf_idx = (k_tile_idx + 1) % 2;

        if (k_tile_idx + 1 < num_k_tiles) {
            uint next_k_tile = (k_tile_idx + 1) * TILE_K;
            load_tile_A(A, shared_A[next_buf_idx], row_base, next_k_tile, M, K, lda, transpose_a, tid);
            load_tile_B(B, shared_B[next_buf_idx], col_base, next_k_tile, N, K, ldb, transpose_b, tid);
        }

        for (uint k = 0; k < TILE_K; k += 8) {
            simdgroup_float8x8 a_frag[4];
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(a_frag[i],
                    shared_A[buf_idx] + (sg_row + i * 8) * TILE_K + k,
                    TILE_K);
            }

            simdgroup_float8x8 b_frag[4];
            for (uint j = 0; j < 4; ++j) {
                simdgroup_load(b_frag[j],
                    shared_B[buf_idx] + k * TILE_N + (sg_col + j * 8),
                    TILE_N);
            }

            for (uint i = 0; i < 4; ++i) {
                for (uint j = 0; j < 4; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        if (k_tile_idx + 1 < num_k_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // FP32 can store directly without conversion
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            uint gr = row_base + sg_row + i * 8;
            uint gc = col_base + sg_col + j * 8;
            if (gr + 7 < M && gc + 7 < N) {
                simdgroup_store(acc[i][j], C + gr * ldc + gc, ldc);
            } else {
                threadgroup float temp_out[8 * 8];
                simdgroup_store(acc[i][j], temp_out, 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint t = simd_lane_id; t < 64; t += 32) {
                    uint lr = t / 8;
                    uint lc = t % 8;
                    uint global_r = gr + lr;
                    uint global_c = gc + lc;
                    if (global_r < M && global_c < N) {
                        C[global_r * ldc + global_c] = temp_out[lr * 8 + lc];
                    }
                }
            }
        }
    }
}

// ============================================================================
// BF16 kernel with FP32 accumulation
// ============================================================================

kernel void matmul_tiled64_bf16(
    device const ushort* A     [[buffer(0)]],  // BF16 as ushort
    device const ushort* B     [[buffer(1)]],  // BF16 as ushort
    device ushort* C           [[buffer(2)]],  // BF16 output
    constant uint& M           [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& lda         [[buffer(6)]],
    constant uint& ldb         [[buffer(7)]],
    constant uint& ldc         [[buffer(8)]],
    constant bool& transpose_a [[buffer(9)]],
    constant bool& transpose_b [[buffer(10)]],
    threadgroup float* shared_A_buf [[threadgroup(0)]],
    threadgroup float* shared_B_buf [[threadgroup(1)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint row_base = gid.y * TILE_M;
    const uint col_base = gid.x * TILE_N;

    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 32;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }

    threadgroup float* shared_A[2] = {
        shared_A_buf,
        shared_A_buf + TILE_M * TILE_K
    };
    threadgroup float* shared_B[2] = {
        shared_B_buf,
        shared_B_buf + TILE_K * TILE_N
    };

    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Load first tile (BF16→FP32 conversion on load)
    for (uint idx = tid; idx < TILE_M * TILE_K; idx += 256) {
        uint r = idx / TILE_K;
        uint c = idx % TILE_K;
        uint global_r = row_base + r;
        uint global_c = c;
        if (global_r < M && global_c < K) {
            if (transpose_a) {
                shared_A[0][r * TILE_K + c] = bf16_to_float(A[global_c * lda + global_r]);
            } else {
                shared_A[0][r * TILE_K + c] = bf16_to_float(A[global_r * lda + global_c]);
            }
        } else {
            shared_A[0][r * TILE_K + c] = 0.0f;
        }
    }
    for (uint idx = tid; idx < TILE_K * TILE_N; idx += 256) {
        uint r = idx / TILE_N;
        uint c = idx % TILE_N;
        uint global_r = r;
        uint global_c = col_base + c;
        if (global_r < K && global_c < N) {
            if (transpose_b) {
                shared_B[0][r * TILE_N + c] = bf16_to_float(B[global_c * ldb + global_r]);
            } else {
                shared_B[0][r * TILE_N + c] = bf16_to_float(B[global_r * ldb + global_c]);
            }
        } else {
            shared_B[0][r * TILE_N + c] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k_tile_idx = 0; k_tile_idx < num_k_tiles; ++k_tile_idx) {
        uint buf_idx = k_tile_idx % 2;
        uint next_buf_idx = (k_tile_idx + 1) % 2;

        if (k_tile_idx + 1 < num_k_tiles) {
            uint next_k_tile = (k_tile_idx + 1) * TILE_K;
            // Load next tile with BF16→FP32 conversion
            for (uint idx = tid; idx < TILE_M * TILE_K; idx += 256) {
                uint r = idx / TILE_K;
                uint c = idx % TILE_K;
                uint global_r = row_base + r;
                uint global_c = next_k_tile + c;
                if (global_r < M && global_c < K) {
                    if (transpose_a) {
                        shared_A[next_buf_idx][r * TILE_K + c] = bf16_to_float(A[global_c * lda + global_r]);
                    } else {
                        shared_A[next_buf_idx][r * TILE_K + c] = bf16_to_float(A[global_r * lda + global_c]);
                    }
                } else {
                    shared_A[next_buf_idx][r * TILE_K + c] = 0.0f;
                }
            }
            for (uint idx = tid; idx < TILE_K * TILE_N; idx += 256) {
                uint r = idx / TILE_N;
                uint c = idx % TILE_N;
                uint global_r = next_k_tile + r;
                uint global_c = col_base + c;
                if (global_r < K && global_c < N) {
                    if (transpose_b) {
                        shared_B[next_buf_idx][r * TILE_N + c] = bf16_to_float(B[global_c * ldb + global_r]);
                    } else {
                        shared_B[next_buf_idx][r * TILE_N + c] = bf16_to_float(B[global_r * ldb + global_c]);
                    }
                } else {
                    shared_B[next_buf_idx][r * TILE_N + c] = 0.0f;
                }
            }
        }

        for (uint k = 0; k < TILE_K; k += 8) {
            simdgroup_float8x8 a_frag[4];
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(a_frag[i],
                    shared_A[buf_idx] + (sg_row + i * 8) * TILE_K + k,
                    TILE_K);
            }

            simdgroup_float8x8 b_frag[4];
            for (uint j = 0; j < 4; ++j) {
                simdgroup_load(b_frag[j],
                    shared_B[buf_idx] + k * TILE_N + (sg_col + j * 8),
                    TILE_N);
            }

            for (uint i = 0; i < 4; ++i) {
                for (uint j = 0; j < 4; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        if (k_tile_idx + 1 < num_k_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store results with FP32→BF16 conversion
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            uint gr = row_base + sg_row + i * 8;
            uint gc = col_base + sg_col + j * 8;
            if (gr + 7 < M && gc + 7 < N) {
                threadgroup float temp_out[8 * 8];
                simdgroup_store(acc[i][j], temp_out, 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint t = simd_lane_id; t < 64; t += 32) {
                    uint lr = t / 8;
                    uint lc = t % 8;
                    uint global_r = gr + lr;
                    uint global_c = gc + lc;
                    C[global_r * ldc + global_c] = float_to_bf16(temp_out[lr * 8 + lc]);
                }
            } else {
                threadgroup float temp_out[8 * 8];
                simdgroup_store(acc[i][j], temp_out, 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint t = simd_lane_id; t < 64; t += 32) {
                    uint lr = t / 8;
                    uint lc = t % 8;
                    uint global_r = gr + lr;
                    uint global_c = gc + lc;
                    if (global_r < M && global_c < N) {
                        C[global_r * ldc + global_c] = float_to_bf16(temp_out[lr * 8 + lc]);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Batched FP16 kernel
// ============================================================================

kernel void matmul_tiled64_batched_fp16(
    device const half* A      [[buffer(0)]],
    device const half* B      [[buffer(1)]],
    device half* C            [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant uint& lda        [[buffer(6)]],
    constant uint& ldb        [[buffer(7)]],
    constant uint& ldc        [[buffer(8)]],
    constant bool& transpose_a [[buffer(9)]],
    constant bool& transpose_b [[buffer(10)]],
    constant uint& batch_stride_a [[buffer(11)]],
    constant uint& batch_stride_b [[buffer(12)]],
    constant uint& batch_stride_c [[buffer(13)]],
    threadgroup half* shared_A_buf [[threadgroup(0)]],
    threadgroup half* shared_B_buf [[threadgroup(1)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_idx = gid.z;
    const uint row_base = gid.y * TILE_M;
    const uint col_base = gid.x * TILE_N;

    device const half* A_batch = A + batch_idx * batch_stride_a;
    device const half* B_batch = B + batch_idx * batch_stride_b;
    device half* C_batch = C + batch_idx * batch_stride_c;

    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 32;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }

    threadgroup half* shared_A[2] = {
        shared_A_buf,
        shared_A_buf + TILE_M * TILE_K
    };
    threadgroup half* shared_B[2] = {
        shared_B_buf,
        shared_B_buf + TILE_K * TILE_N
    };

    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    load_tile_A(A_batch, shared_A[0], row_base, 0, M, K, lda, transpose_a, tid);
    load_tile_B(B_batch, shared_B[0], col_base, 0, N, K, ldb, transpose_b, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k_tile_idx = 0; k_tile_idx < num_k_tiles; ++k_tile_idx) {
        uint buf_idx = k_tile_idx % 2;
        uint next_buf_idx = (k_tile_idx + 1) % 2;

        if (k_tile_idx + 1 < num_k_tiles) {
            uint next_k_tile = (k_tile_idx + 1) * TILE_K;
            load_tile_A(A_batch, shared_A[next_buf_idx], row_base, next_k_tile, M, K, lda, transpose_a, tid);
            load_tile_B(B_batch, shared_B[next_buf_idx], col_base, next_k_tile, N, K, ldb, transpose_b, tid);
        }

        for (uint k = 0; k < TILE_K; k += 8) {
            simdgroup_half8x8 a_frag[4];
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(a_frag[i],
                    shared_A[buf_idx] + (sg_row + i * 8) * TILE_K + k,
                    TILE_K);
            }

            simdgroup_half8x8 b_frag[4];
            for (uint j = 0; j < 4; ++j) {
                simdgroup_load(b_frag[j],
                    shared_B[buf_idx] + k * TILE_N + (sg_col + j * 8),
                    TILE_N);
            }

            for (uint i = 0; i < 4; ++i) {
                for (uint j = 0; j < 4; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        if (k_tile_idx + 1 < num_k_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Batched FP32→FP16 conversion: same optimization as non-batched kernel
    threadgroup float* sg_scratch;
    if (simd_group_id < 4) {
        sg_scratch = reinterpret_cast<threadgroup float*>(shared_A_buf) + simd_group_id * (16 * 32);
    } else {
        sg_scratch = reinterpret_cast<threadgroup float*>(shared_B_buf) + (simd_group_id - 4) * (16 * 32);
    }

    // Store all 16 FP32 accumulators to scratch
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            simdgroup_store(acc[i][j], sg_scratch + (i * 8) * 32 + (j * 8), 32);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bulk convert and write to device memory
    for (uint t = simd_lane_id; t < 16 * 32; t += 32) {
        uint lr = t / 32;
        uint lc = t % 32;
        uint global_r = row_base + sg_row + lr;
        uint global_c = col_base + sg_col + lc;
        if (global_r < M && global_c < N) {
            C_batch[global_r * ldc + global_c] = half(sg_scratch[lr * 32 + lc]);
        }
    }
}
