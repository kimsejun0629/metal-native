/// @file matmul_dequant_kernel.metal
/// @brief Fused INT4/INT8 dequantization + matrix multiplication kernels.
///
/// Eliminates intermediate FP16 buffer by dequantizing weights on-the-fly during matmul.
/// Memory flow: Read INT4/INT8 -> Dequantize in shared memory -> Compute -> Write FP16
/// vs. separate: Read INT4 -> Write FP16 -> Read FP16 -> Compute -> Write FP16

#include <metal_stdlib>
using namespace metal;

// Tile configuration optimized for quantized matmul
constant constexpr uint TILE_M = 64;  // rows per threadgroup
constant constexpr uint TILE_N = 64;  // cols per threadgroup
constant constexpr uint TILE_K = 32;  // K dimension tiling (matches typical group_size)

// Threadgroup configuration: 128 threads (4 SIMD groups of 32)
// Each SIMD group computes a 32x32 sub-block using 4x4 of 8x8 tiles

// ============================================================================
// Helper: INT4 unpacking and dequantization
// ============================================================================

/// Unpack and dequantize a single INT4 byte (2 weights) using group quantization.
/// @param packed Single byte containing 2 INT4 values (low nibble, high nibble)
/// @param scale Scale factor for this group
/// @param zero Zero point for this group (optional, can be 0 for symmetric)
/// @param out0 First dequantized FP16 value (low nibble)
/// @param out1 Second dequantized FP16 value (high nibble)
inline void unpack_int4_pair(
    uchar packed,
    half scale,
    half zero,
    thread half& out0,
    thread half& out1
) {
    // Extract nibbles (unsigned INT4: 0-15)
    int low = int(packed & 0x0F);
    int high = int((packed >> 4) & 0x0F);

    // Convert to signed: subtract 8 for zero-centered [-8, 7]
    // Dequantize: (int4_val - 8 - zero) * scale
    out0 = half(low - 8) * scale - zero;
    out1 = half(high - 8) * scale - zero;
}

/// Dequantize INT8 value
inline half dequant_int8(char val, half scale, half zero) {
    return half(val) * scale - zero;
}

// ============================================================================
// Main fused kernel: INT4 dequantize + matmul
// ============================================================================

kernel void dequant_matmul_int4_fp16(
    device const half* A              [[buffer(0)]],   // [M, K] FP16 activations
    device const uchar* B_packed      [[buffer(1)]],   // [N, K/2] INT4 packed weights
    device half* C                    [[buffer(2)]],   // [M, N] output
    device const half* scales         [[buffer(3)]],   // [N, K/group_size] scale factors
    device const half* zeros          [[buffer(4)]],   // [N, K/group_size] zero points
    constant uint& M                  [[buffer(5)]],
    constant uint& N                  [[buffer(6)]],
    constant uint& K                  [[buffer(7)]],
    constant uint& group_size         [[buffer(8)]],   // 32, 64, or 128
    constant uint& lda                [[buffer(9)]],   // leading dim of A (usually K)
    constant uint& ldc                [[buffer(10)]],  // leading dim of C (usually N)
    threadgroup half* shared_A        [[threadgroup(0)]],  // [TILE_M, TILE_K] = 4KB
    threadgroup half* shared_B        [[threadgroup(1)]],  // [TILE_K, TILE_N] = 4KB
    uint3 gid                         [[threadgroup_position_in_grid]],
    uint3 tid3                        [[thread_position_in_threadgroup]]
) {
    uint tid_in_tg = tid3.x;
    uint simd_lane_id = tid_in_tg % 32;
    uint simd_group_id = tid_in_tg / 32;

    // Each threadgroup computes a TILE_M x TILE_N output block
    const uint row_base = gid.y * TILE_M;
    const uint col_base = gid.x * TILE_N;

    // 4 SIMD groups arranged in 2x2 grid, each computing 32x32 (as 4x4 of 8x8)
    // SIMD group layout:
    //   0: rows 0-31,  cols 0-31
    //   1: rows 0-31,  cols 32-63
    //   2: rows 32-63, cols 0-31
    //   3: rows 32-63, cols 32-63

    // Each SIMD group maintains 4x4 = 16 accumulators of 8x8
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }

    uint sg_row = (simd_group_id / 2) * 32;  // 0 or 32
    uint sg_col = (simd_group_id % 2) * 32;  // 0 or 32

    uint num_groups_per_k = (K + group_size - 1) / group_size;

    // Iterate over K dimension in tiles
    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // ====================================================================
        // Load A tile [TILE_M, TILE_K] into shared memory (standard FP16 load)
        // ====================================================================
        // 128 threads, 2048 elements (64*32) -> 16 elements per thread
        for (uint idx = tid_in_tg; idx < TILE_M * TILE_K; idx += 128) {
            uint r = idx / TILE_K;
            uint c = idx % TILE_K;
            uint global_r = row_base + r;
            uint global_c = k_tile + c;
            if (global_r < M && global_c < K) {
                shared_A[r * TILE_K + c] = A[global_r * lda + global_c];
            } else {
                shared_A[r * TILE_K + c] = half(0.0f);
            }
        }

        // ====================================================================
        // Load and dequantize B tile [TILE_K, TILE_N] with INT4 unpacking
        // ====================================================================
        // B_packed: [N, K/2] layout - each row has K/2 bytes
        // Each packed byte contains 2 K-adjacent INT4 values for one N position

        // Total elements: TILE_K * TILE_N = 32 * 64 = 2048 FP16 values
        // Packed size: 2048 / 2 = 1024 bytes
        // Iterate over (local_n, local_k_pair) where local_k_pair indexes packed bytes along K

        uint total_packed = TILE_N * (TILE_K / 2);  // each N column has TILE_K/2 packed bytes
        for (uint p_idx = tid_in_tg; p_idx < total_packed; p_idx += 128) {
            uint local_n = p_idx / (TILE_K / 2);       // N position
            uint local_k_pair = p_idx % (TILE_K / 2);  // which packed byte pair along K
            uint local_k = local_k_pair * 2;            // K position of first value

            uint global_n = col_base + local_n;
            uint global_k = k_tile + local_k;

            if (global_n < N && global_k < K) {
                uint packed_k_idx = global_k / 2;
                uchar packed = B_packed[global_n * ((K + 1) / 2) + packed_k_idx];

                uint group_idx = global_k / group_size;
                half scale = scales[global_n * num_groups_per_k + group_idx];
                half zero = zeros[global_n * num_groups_per_k + group_idx];

                half val0, val1;
                unpack_int4_pair(packed, scale, zero, val0, val1);

                // Write to K-adjacent positions (same N column, consecutive K rows)
                shared_B[local_k * TILE_N + local_n] = val0;
                if (local_k + 1 < TILE_K && global_k + 1 < K) {
                    shared_B[(local_k + 1) * TILE_N + local_n] = val1;
                }
            } else {
                // Zero-fill out-of-bounds
                if (local_n < TILE_N && local_k < TILE_K) {
                    shared_B[local_k * TILE_N + local_n] = half(0.0h);
                }
                if (local_n < TILE_N && local_k + 1 < TILE_K) {
                    shared_B[(local_k + 1) * TILE_N + local_n] = half(0.0h);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ====================================================================
        // Compute: Each SIMD group computes its 32x32 sub-block
        // ====================================================================
        for (uint k = 0; k < TILE_K; k += 8) {
            // Load 4x4 grid of 8x8 tiles from shared memory
            simdgroup_half8x8 a_tiles[4];
            simdgroup_half8x8 b_tiles[4];

            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a_tiles[i], shared_A + (sg_row + i * 8) * TILE_K + k, TILE_K);
            }

            for (uint j = 0; j < 4; j++) {
                simdgroup_load(b_tiles[j], shared_B + k * TILE_N + (sg_col + j * 8), TILE_N);
            }

            // Outer product: accumulate all 16 combinations
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a_tiles[i], b_tiles[j], acc[i][j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ====================================================================
    // Write results: Serialize SIMD groups through shared workspace (4KB)
    // ====================================================================
    threadgroup float shared_out[32 * 32];

    for (uint sg = 0; sg < 4; sg++) {
        if (simd_group_id == sg) {
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_store(acc[i][j],
                                  shared_out + (i * 8) * 32 + (j * 8),
                                  32);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == sg) {
            for (uint idx = simd_lane_id; idx < 32 * 32; idx += 32) {
                uint lr = idx / 32;
                uint lc = idx % 32;
                uint gr = row_base + sg_row + lr;
                uint gc = col_base + sg_col + lc;
                if (gr < M && gc < N) {
                    C[gr * ldc + gc] = half(shared_out[lr * 32 + lc]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ============================================================================
// Optimized vec-mat variant for decode phase (M=1)
// ============================================================================

kernel void dequant_matmul_int4_vecmat_fp16(
    device const half* A              [[buffer(0)]],   // [1, K] FP16 activation vector
    device const uchar* B_packed      [[buffer(1)]],   // [N, K/2] INT4 packed weights
    device half* C                    [[buffer(2)]],   // [1, N] output vector
    device const half* scales         [[buffer(3)]],   // [N, K/group_size] scale factors
    device const half* zeros          [[buffer(4)]],   // [N, K/group_size] zero points
    constant uint& N                  [[buffer(5)]],
    constant uint& K                  [[buffer(6)]],
    constant uint& group_size         [[buffer(7)]],
    threadgroup half* shared_A        [[threadgroup(0)]],  // [TILE_K] shared activation
    threadgroup half* shared_B        [[threadgroup(1)]],  // [TILE_N, TILE_K] weight tile
    uint3 gid                         [[threadgroup_position_in_grid]],
    uint3 tid3                        [[thread_position_in_threadgroup]]
) {
    uint tid_in_tg = tid3.x;
    uint simd_lane_id = tid_in_tg % 32;
    uint simd_group_id = tid_in_tg / 32;

    // Each threadgroup computes TILE_N output elements
    const uint col_base = gid.x * TILE_N;

    // 4 SIMD groups, each computes 16 output elements
    uint sg_col = (simd_group_id % 4) * 16;  // 0, 16, 32, 48

    // Accumulators: 2 tiles of 8 elements each
    simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc1 = simdgroup_float8x8(0.0f);

    uint num_groups_per_k = (K + group_size - 1) / group_size;

    // Iterate over K dimension
    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load A vector tile into shared memory (broadcast to all threads)
        for (uint k = tid_in_tg; k < TILE_K; k += 128) {
            uint global_k = k_tile + k;
            if (global_k < K) {
                shared_A[k] = A[global_k];
            } else {
                shared_A[k] = half(0.0f);
            }
        }

        // Load and dequantize B tile [TILE_N, TILE_K] (transposed for vecmat)
        // Each packed byte contains 2 K-adjacent INT4 values for one N position
        uint total_packed = TILE_N * (TILE_K / 2);
        for (uint p_idx = tid_in_tg; p_idx < total_packed; p_idx += 128) {
            uint local_n = p_idx / (TILE_K / 2);       // N position
            uint local_k_pair = p_idx % (TILE_K / 2);  // which packed byte pair along K
            uint local_k = local_k_pair * 2;            // K position of first value

            uint global_n = col_base + local_n;
            uint global_k = k_tile + local_k;

            if (global_n < N && global_k < K) {
                uint packed_k_idx = global_k / 2;
                uchar packed = B_packed[global_n * ((K + 1) / 2) + packed_k_idx];

                uint group_idx = global_k / group_size;
                half scale = scales[global_n * num_groups_per_k + group_idx];
                half zero = zeros[global_n * num_groups_per_k + group_idx];

                half val0, val1;
                unpack_int4_pair(packed, scale, zero, val0, val1);

                // K-major layout [TILE_K, TILE_N]: shared_B[k * TILE_N + n]
                shared_B[local_k * TILE_N + local_n] = val0;
                if (local_k + 1 < TILE_K && global_k + 1 < K) {
                    shared_B[(local_k + 1) * TILE_N + local_n] = val1;
                }
            } else {
                // Zero-fill out-of-bounds (K-major layout)
                if (local_n < TILE_N && local_k < TILE_K) {
                    shared_B[local_k * TILE_N + local_n] = half(0.0h);
                }
                if (local_n < TILE_N && local_k + 1 < TILE_K) {
                    shared_B[(local_k + 1) * TILE_N + local_n] = half(0.0h);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute: dot product of A vector with B columns (K-major layout)
        for (uint k = 0; k < TILE_K; k += 8) {
            simdgroup_half8x8 a_tile;
            simdgroup_half8x8 b_tile0, b_tile1;

            // Load A tile (broadcast as row vector)
            simdgroup_load(a_tile, shared_A + k, TILE_K);

            // Load B tiles: K in rows, N in cols (K-major, stride=TILE_N)
            simdgroup_load(b_tile0, shared_B + k * TILE_N + sg_col, TILE_N);
            simdgroup_load(b_tile1, shared_B + k * TILE_N + sg_col + 8, TILE_N);

            simdgroup_multiply_accumulate(acc0, a_tile, b_tile0, acc0);
            simdgroup_multiply_accumulate(acc1, a_tile, b_tile1, acc1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results: per-SIMD-group workspace (128 floats for 8x8 store with stride 16)
    threadgroup float shared_out[4 * 128];
    threadgroup float* my_out = shared_out + simd_group_id * 128;

    simdgroup_store(acc0, my_out + 0, 16);
    simdgroup_store(acc1, my_out + 8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Read only row 0 of the 8x8 result (the vecmat output)
    for (uint i = simd_lane_id; i < 16; i += 32) {
        uint gc = col_base + sg_col + i;
        if (gc < N) {
            C[gc] = half(my_out[i]);
        }
    }
}

// ============================================================================
// INT8 fused kernel (simpler: no nibble unpacking)
// ============================================================================

kernel void dequant_matmul_int8_fp16(
    device const half* A              [[buffer(0)]],   // [M, K] FP16 activations
    device const char* B_packed       [[buffer(1)]],   // [N, K] INT8 weights
    device half* C                    [[buffer(2)]],   // [M, N] output
    device const half* scales         [[buffer(3)]],   // [N, K/group_size] scale factors
    device const half* zeros          [[buffer(4)]],   // [N, K/group_size] zero points
    constant uint& M                  [[buffer(5)]],
    constant uint& N                  [[buffer(6)]],
    constant uint& K                  [[buffer(7)]],
    constant uint& group_size         [[buffer(8)]],
    constant uint& lda                [[buffer(9)]],
    constant uint& ldc                [[buffer(10)]],
    threadgroup half* shared_A        [[threadgroup(0)]],
    threadgroup half* shared_B        [[threadgroup(1)]],
    uint3 gid                         [[threadgroup_position_in_grid]],
    uint3 tid3                        [[thread_position_in_threadgroup]]
) {
    uint tid_in_tg = tid3.x;
    uint simd_lane_id = tid_in_tg % 32;
    uint simd_group_id = tid_in_tg / 32;

    const uint row_base = gid.y * TILE_M;
    const uint col_base = gid.x * TILE_N;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }

    uint sg_row = (simd_group_id / 2) * 32;
    uint sg_col = (simd_group_id % 2) * 32;

    uint num_groups_per_k = (K + group_size - 1) / group_size;

    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load A tile (standard FP16)
        for (uint idx = tid_in_tg; idx < TILE_M * TILE_K; idx += 128) {
            uint r = idx / TILE_K;
            uint c = idx % TILE_K;
            uint global_r = row_base + r;
            uint global_c = k_tile + c;
            if (global_r < M && global_c < K) {
                shared_A[r * TILE_K + c] = A[global_r * lda + global_c];
            } else {
                shared_A[r * TILE_K + c] = half(0.0f);
            }
        }

        // Load and dequantize B tile (INT8 -> FP16)
        // INT8 is simpler: 1 value per byte, no unpacking needed
        for (uint idx = tid_in_tg; idx < TILE_K * TILE_N; idx += 128) {
            uint local_k = idx / TILE_N;
            uint local_n = idx % TILE_N;

            uint global_k = k_tile + local_k;
            uint global_n = col_base + local_n;

            if (global_k < K && global_n < N) {
                // Read INT8 value
                char int8_val = B_packed[global_n * K + global_k];

                // Get scale and zero for this group
                uint group_idx = global_k / group_size;
                half scale = scales[global_n * num_groups_per_k + group_idx];
                half zero = zeros[global_n * num_groups_per_k + group_idx];

                // Dequantize
                shared_B[local_k * TILE_N + local_n] = dequant_int8(int8_val, scale, zero);
            } else {
                shared_B[local_k * TILE_N + local_n] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute (same as INT4 version)
        for (uint k = 0; k < TILE_K; k += 8) {
            simdgroup_half8x8 a_tiles[4];
            simdgroup_half8x8 b_tiles[4];

            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a_tiles[i], shared_A + (sg_row + i * 8) * TILE_K + k, TILE_K);
            }

            for (uint j = 0; j < 4; j++) {
                simdgroup_load(b_tiles[j], shared_B + k * TILE_N + (sg_col + j * 8), TILE_N);
            }

            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a_tiles[i], b_tiles[j], acc[i][j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results: Serialize SIMD groups through shared workspace (4KB)
    threadgroup float shared_out[32 * 32];

    for (uint sg = 0; sg < 4; sg++) {
        if (simd_group_id == sg) {
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_store(acc[i][j],
                                  shared_out + (i * 8) * 32 + (j * 8),
                                  32);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == sg) {
            for (uint idx = simd_lane_id; idx < 32 * 32; idx += 32) {
                uint lr = idx / 32;
                uint lc = idx % 32;
                uint gr = row_base + sg_row + lr;
                uint gc = col_base + sg_col + lc;
                if (gr < M && gc < N) {
                    C[gr * ldc + gc] = half(shared_out[lr * 32 + lc]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ============================================================================
// INT8 vec-mat variant
// ============================================================================

kernel void dequant_matmul_int8_vecmat_fp16(
    device const half* A              [[buffer(0)]],   // [1, K] FP16 activation vector
    device const char* B_packed       [[buffer(1)]],   // [N, K] INT8 weights
    device half* C                    [[buffer(2)]],   // [1, N] output vector
    device const half* scales         [[buffer(3)]],   // [N, K/group_size] scale factors
    device const half* zeros          [[buffer(4)]],   // [N, K/group_size] zero points
    constant uint& N                  [[buffer(5)]],
    constant uint& K                  [[buffer(6)]],
    constant uint& group_size         [[buffer(7)]],
    threadgroup half* shared_A        [[threadgroup(0)]],
    threadgroup half* shared_B        [[threadgroup(1)]],
    uint3 gid                         [[threadgroup_position_in_grid]],
    uint3 tid3                        [[thread_position_in_threadgroup]]
) {
    uint tid_in_tg = tid3.x;
    uint simd_lane_id = tid_in_tg % 32;
    uint simd_group_id = tid_in_tg / 32;

    const uint col_base = gid.x * TILE_N;
    uint sg_col = (simd_group_id % 4) * 16;

    simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc1 = simdgroup_float8x8(0.0f);

    uint num_groups_per_k = (K + group_size - 1) / group_size;

    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load A vector
        for (uint k = tid_in_tg; k < TILE_K; k += 128) {
            uint global_k = k_tile + k;
            if (global_k < K) {
                shared_A[k] = A[global_k];
            } else {
                shared_A[k] = half(0.0f);
            }
        }

        // Load and dequantize B tile
        for (uint idx = tid_in_tg; idx < TILE_K * TILE_N; idx += 128) {
            uint local_k = idx / TILE_N;
            uint local_n = idx % TILE_N;

            uint global_k = k_tile + local_k;
            uint global_n = col_base + local_n;

            if (global_k < K && global_n < N) {
                char int8_val = B_packed[global_n * K + global_k];

                uint group_idx = global_k / group_size;
                half scale = scales[global_n * num_groups_per_k + group_idx];
                half zero = zeros[global_n * num_groups_per_k + group_idx];

                // K-major layout for vec-mat: shared_B[k * TILE_N + n]
                shared_B[local_k * TILE_N + local_n] = dequant_int8(int8_val, scale, zero);
            } else {
                shared_B[local_k * TILE_N + local_n] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute: K-major layout, stride=TILE_N
        for (uint k = 0; k < TILE_K; k += 8) {
            simdgroup_half8x8 a_tile;
            simdgroup_half8x8 b_tile0, b_tile1;

            simdgroup_load(a_tile, shared_A + k, TILE_K);
            simdgroup_load(b_tile0, shared_B + k * TILE_N + sg_col, TILE_N);
            simdgroup_load(b_tile1, shared_B + k * TILE_N + sg_col + 8, TILE_N);

            simdgroup_multiply_accumulate(acc0, a_tile, b_tile0, acc0);
            simdgroup_multiply_accumulate(acc1, a_tile, b_tile1, acc1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results: per-SIMD-group workspace
    threadgroup float shared_out[4 * 128];
    threadgroup float* my_out = shared_out + simd_group_id * 128;

    simdgroup_store(acc0, my_out + 0, 16);
    simdgroup_store(acc1, my_out + 8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = simd_lane_id; i < 16; i += 32) {
        uint gc = col_base + sg_col + i;
        if (gc < N) {
            C[gc] = half(my_out[i]);
        }
    }
}
