/// @file matmul_simd_kernel.metal
/// @brief Custom SIMD matrix multiplication kernels using AMX acceleration.
///
/// These kernels avoid MPSGraph dispatch overhead for small/medium matrices
/// (M, N, K ≤ 2048) by using simdgroup_matrix_multiply_accumulate directly.

#include <metal_stdlib>
#include "common/metal_types.h"
using namespace metal;

// Tile configuration for output computation
constant constexpr uint TILE_M = 32;  // rows per threadgroup
constant constexpr uint TILE_N = 32;  // cols per threadgroup
constant constexpr uint TILE_K = 32;  // K dimension tiling

// Each threadgroup has 128 threads (4 SIMD groups of 32 threads)
// Output: 32x32 tile = 4 blocks of 16x16 = 16 sub-blocks of 8x8
// 4 SIMD groups, each computes a 16x16 block (as 2x2 of 8x8)

// ============================================================================
// FP16 kernel with FP32 accumulation (mixed precision)
// ============================================================================

kernel void matmul_simd_fp16(
    device const half* A      [[buffer(0)]],
    device const half* B      [[buffer(1)]],
    device half* C            [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant uint& lda        [[buffer(6)]],  // leading dim of A
    constant uint& ldb        [[buffer(7)]],  // leading dim of B
    constant uint& ldc        [[buffer(8)]],  // leading dim of C
    constant bool& transpose_a [[buffer(9)]],
    constant bool& transpose_b [[buffer(10)]],
    threadgroup half* shared_A [[threadgroup(0)]],  // [TILE_M, TILE_K]
    threadgroup half* shared_B [[threadgroup(1)]],  // [TILE_K, TILE_N]
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup computes a TILE_M x TILE_N output block
    const uint row_base = gid.y * TILE_M;
    const uint col_base = gid.x * TILE_N;

    // 4 SIMD groups arranged in 2x2 grid, each computing 16x16 (as 2x2 of 8x8)
    // SIMD group layout:
    //   0: rows 0-15,  cols 0-15
    //   1: rows 0-15,  cols 16-31
    //   2: rows 16-31, cols 0-15
    //   3: rows 16-31, cols 16-31

    // Each SIMD group maintains 2x2 = 4 accumulators of 8x8
    simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc11 = simdgroup_float8x8(0.0f);

    uint sg_row = (simd_group_id / 2) * 16;  // 0 or 16
    uint sg_col = (simd_group_id % 2) * 16;  // 0 or 16

    // Iterate over K dimension in tiles
    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperatively load A tile [TILE_M, TILE_K] into shared memory
        // 128 threads total, 1024 elements to load (32*32)
        for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_M * TILE_K; idx += 128) {
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
                shared_A[r * TILE_K + c] = half(0.0f);
            }
        }

        // Cooperatively load B tile [TILE_K, TILE_N]
        for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_K * TILE_N; idx += 128) {
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
                shared_B[r * TILE_N + c] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each SIMD group computes its 16x16 sub-block using 2x2 of 8x8 SIMD ops
        for (uint k = 0; k < TILE_K; k += 8) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, shared_A + (sg_row + 0) * TILE_K + k, TILE_K);
            simdgroup_load(a1, shared_A + (sg_row + 8) * TILE_K + k, TILE_K);
            simdgroup_load(b0, shared_B + k * TILE_N + (sg_col + 0), TILE_N);
            simdgroup_load(b1, shared_B + k * TILE_N + (sg_col + 8), TILE_N);

            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results - convert FP32 accumulators back to FP16
    // Use threadgroup memory for conversion workspace
    threadgroup float shared_float[16 * 16];  // Per SIMD group workspace

    simdgroup_store(acc00, shared_float + 0 * 16 + 0, 16);
    simdgroup_store(acc01, shared_float + 0 * 16 + 8, 16);
    simdgroup_store(acc10, shared_float + 8 * 16 + 0, 16);
    simdgroup_store(acc11, shared_float + 8 * 16 + 8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Convert and write to global output
    for (uint idx = simd_lane_id; idx < 16 * 16; idx += 32) {
        uint lr = idx / 16;
        uint lc = idx % 16;
        uint gr = row_base + sg_row + lr;
        uint gc = col_base + sg_col + lc;
        if (gr < M && gc < N) {
            C[gr * ldc + gc] = half(shared_float[lr * 16 + lc]);
        }
    }
}

// ============================================================================
// FP32 kernel (full precision throughout)
// ============================================================================

kernel void matmul_simd_fp32(
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
    threadgroup float* shared_A [[threadgroup(0)]],  // [TILE_M, TILE_K]
    threadgroup float* shared_B [[threadgroup(1)]],  // [TILE_K, TILE_N]
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint row_base = gid.y * TILE_M;
    const uint col_base = gid.x * TILE_N;

    simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc11 = simdgroup_float8x8(0.0f);

    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 16;

    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load A tile
        for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_M * TILE_K; idx += 128) {
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
                shared_A[r * TILE_K + c] = 0.0f;
            }
        }

        // Load B tile
        for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_K * TILE_N; idx += 128) {
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
                shared_B[r * TILE_N + c] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute 16x16 block
        for (uint k = 0; k < TILE_K; k += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, shared_A + (sg_row + 0) * TILE_K + k, TILE_K);
            simdgroup_load(a1, shared_A + (sg_row + 8) * TILE_K + k, TILE_K);
            simdgroup_load(b0, shared_B + k * TILE_N + (sg_col + 0), TILE_N);
            simdgroup_load(b1, shared_B + k * TILE_N + (sg_col + 8), TILE_N);

            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results directly (no conversion needed for FP32)
    threadgroup float shared_out[16 * 16];

    simdgroup_store(acc00, shared_out + 0 * 16 + 0, 16);
    simdgroup_store(acc01, shared_out + 0 * 16 + 8, 16);
    simdgroup_store(acc10, shared_out + 8 * 16 + 0, 16);
    simdgroup_store(acc11, shared_out + 8 * 16 + 8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = simd_lane_id; idx < 16 * 16; idx += 32) {
        uint lr = idx / 16;
        uint lc = idx % 16;
        uint gr = row_base + sg_row + lr;
        uint gc = col_base + sg_col + lc;
        if (gr < M && gc < N) {
            C[gr * ldc + gc] = shared_out[lr * 16 + lc];
        }
    }
}

// ============================================================================
// Batched FP16 kernel
// ============================================================================

kernel void matmul_simd_batched_fp16(
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
    threadgroup half* shared_A [[threadgroup(0)]],
    threadgroup half* shared_B [[threadgroup(1)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_idx = gid.z;
    const uint row_base = gid.y * TILE_M;
    const uint col_base = gid.x * TILE_N;

    // Offset pointers for current batch
    device const half* A_batch = A + batch_idx * batch_stride_a;
    device const half* B_batch = B + batch_idx * batch_stride_b;
    device half* C_batch = C + batch_idx * batch_stride_c;

    simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc11 = simdgroup_float8x8(0.0f);

    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 16;

    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_M * TILE_K; idx += 128) {
            uint r = idx / TILE_K;
            uint c = idx % TILE_K;
            uint global_r = row_base + r;
            uint global_c = k_tile + c;
            if (global_r < M && global_c < K) {
                if (transpose_a) {
                    shared_A[r * TILE_K + c] = A_batch[global_c * lda + global_r];
                } else {
                    shared_A[r * TILE_K + c] = A_batch[global_r * lda + global_c];
                }
            } else {
                shared_A[r * TILE_K + c] = half(0.0f);
            }
        }

        for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_K * TILE_N; idx += 128) {
            uint r = idx / TILE_N;
            uint c = idx % TILE_N;
            uint global_r = k_tile + r;
            uint global_c = col_base + c;
            if (global_r < K && global_c < N) {
                if (transpose_b) {
                    shared_B[r * TILE_N + c] = B_batch[global_c * ldb + global_r];
                } else {
                    shared_B[r * TILE_N + c] = B_batch[global_r * ldb + global_c];
                }
            } else {
                shared_B[r * TILE_N + c] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_K; k += 8) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, shared_A + (sg_row + 0) * TILE_K + k, TILE_K);
            simdgroup_load(a1, shared_A + (sg_row + 8) * TILE_K + k, TILE_K);
            simdgroup_load(b0, shared_B + k * TILE_N + (sg_col + 0), TILE_N);
            simdgroup_load(b1, shared_B + k * TILE_N + (sg_col + 8), TILE_N);

            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float shared_float[16 * 16];

    simdgroup_store(acc00, shared_float + 0 * 16 + 0, 16);
    simdgroup_store(acc01, shared_float + 0 * 16 + 8, 16);
    simdgroup_store(acc10, shared_float + 8 * 16 + 0, 16);
    simdgroup_store(acc11, shared_float + 8 * 16 + 8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = simd_lane_id; idx < 16 * 16; idx += 32) {
        uint lr = idx / 16;
        uint lc = idx % 16;
        uint gr = row_base + sg_row + lr;
        uint gc = col_base + sg_col + lc;
        if (gr < M && gc < N) {
            C_batch[gr * ldc + gc] = half(shared_float[lr * 16 + lc]);
        }
    }
}

// ============================================================================
// Batched FP32 kernel
// ============================================================================

kernel void matmul_simd_batched_fp32(
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
    constant uint& batch_stride_a [[buffer(11)]],
    constant uint& batch_stride_b [[buffer(12)]],
    constant uint& batch_stride_c [[buffer(13)]],
    threadgroup float* shared_A [[threadgroup(0)]],
    threadgroup float* shared_B [[threadgroup(1)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_idx = gid.z;
    const uint row_base = gid.y * TILE_M;
    const uint col_base = gid.x * TILE_N;

    device const float* A_batch = A + batch_idx * batch_stride_a;
    device const float* B_batch = B + batch_idx * batch_stride_b;
    device float* C_batch = C + batch_idx * batch_stride_c;

    simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc11 = simdgroup_float8x8(0.0f);

    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 16;

    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_M * TILE_K; idx += 128) {
            uint r = idx / TILE_K;
            uint c = idx % TILE_K;
            uint global_r = row_base + r;
            uint global_c = k_tile + c;
            if (global_r < M && global_c < K) {
                if (transpose_a) {
                    shared_A[r * TILE_K + c] = A_batch[global_c * lda + global_r];
                } else {
                    shared_A[r * TILE_K + c] = A_batch[global_r * lda + global_c];
                }
            } else {
                shared_A[r * TILE_K + c] = 0.0f;
            }
        }

        for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_K * TILE_N; idx += 128) {
            uint r = idx / TILE_N;
            uint c = idx % TILE_N;
            uint global_r = k_tile + r;
            uint global_c = col_base + c;
            if (global_r < K && global_c < N) {
                if (transpose_b) {
                    shared_B[r * TILE_N + c] = B_batch[global_c * ldb + global_r];
                } else {
                    shared_B[r * TILE_N + c] = B_batch[global_r * ldb + global_c];
                }
            } else {
                shared_B[r * TILE_N + c] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_K; k += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, shared_A + (sg_row + 0) * TILE_K + k, TILE_K);
            simdgroup_load(a1, shared_A + (sg_row + 8) * TILE_K + k, TILE_K);
            simdgroup_load(b0, shared_B + k * TILE_N + (sg_col + 0), TILE_N);
            simdgroup_load(b1, shared_B + k * TILE_N + (sg_col + 8), TILE_N);

            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float shared_out[16 * 16];

    simdgroup_store(acc00, shared_out + 0 * 16 + 0, 16);
    simdgroup_store(acc01, shared_out + 0 * 16 + 8, 16);
    simdgroup_store(acc10, shared_out + 8 * 16 + 0, 16);
    simdgroup_store(acc11, shared_out + 8 * 16 + 8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = simd_lane_id; idx < 16 * 16; idx += 32) {
        uint lr = idx / 16;
        uint lc = idx % 16;
        uint gr = row_base + sg_row + lr;
        uint gc = col_base + sg_col + lc;
        if (gr < M && gc < N) {
            C_batch[gr * ldc + gc] = shared_out[lr * 16 + lc];
        }
    }
}

// ============================================================================
// BF16 kernel with FP32 accumulation (BFloat16 input/output, FP32 compute)
// ============================================================================

kernel void matmul_simd_bf16(
    device const ushort* A    [[buffer(0)]],  // BF16 as ushort
    device const ushort* B    [[buffer(1)]],  // BF16 as ushort
    device ushort* C          [[buffer(2)]],  // BF16 output
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant uint& lda        [[buffer(6)]],
    constant uint& ldb        [[buffer(7)]],
    constant uint& ldc        [[buffer(8)]],
    constant bool& transpose_a [[buffer(9)]],
    constant bool& transpose_b [[buffer(10)]],
    threadgroup float* shared_A [[threadgroup(0)]],  // Convert to FP32 on load
    threadgroup float* shared_B [[threadgroup(1)]],  // Convert to FP32 on load
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint row_base = gid.y * TILE_M;
    const uint col_base = gid.x * TILE_N;

    simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc11 = simdgroup_float8x8(0.0f);

    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 16;

    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load A tile and convert BF16 -> FP32
        for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_M * TILE_K; idx += 128) {
            uint r = idx / TILE_K;
            uint c = idx % TILE_K;
            uint global_r = row_base + r;
            uint global_c = k_tile + c;
            if (global_r < M && global_c < K) {
                if (transpose_a) {
                    shared_A[r * TILE_K + c] = bf16_to_float(A[global_c * lda + global_r]);
                } else {
                    shared_A[r * TILE_K + c] = bf16_to_float(A[global_r * lda + global_c]);
                }
            } else {
                shared_A[r * TILE_K + c] = 0.0f;
            }
        }

        // Load B tile and convert BF16 -> FP32
        for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_K * TILE_N; idx += 128) {
            uint r = idx / TILE_N;
            uint c = idx % TILE_N;
            uint global_r = k_tile + r;
            uint global_c = col_base + c;
            if (global_r < K && global_c < N) {
                if (transpose_b) {
                    shared_B[r * TILE_N + c] = bf16_to_float(B[global_c * ldb + global_r]);
                } else {
                    shared_B[r * TILE_N + c] = bf16_to_float(B[global_r * ldb + global_c]);
                }
            } else {
                shared_B[r * TILE_N + c] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute using FP32 (same as FP32 kernel)
        for (uint k = 0; k < TILE_K; k += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, shared_A + (sg_row + 0) * TILE_K + k, TILE_K);
            simdgroup_load(a1, shared_A + (sg_row + 8) * TILE_K + k, TILE_K);
            simdgroup_load(b0, shared_B + k * TILE_N + (sg_col + 0), TILE_N);
            simdgroup_load(b1, shared_B + k * TILE_N + (sg_col + 8), TILE_N);

            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results and convert FP32 -> BF16
    threadgroup float shared_out[16 * 16];

    simdgroup_store(acc00, shared_out + 0 * 16 + 0, 16);
    simdgroup_store(acc01, shared_out + 0 * 16 + 8, 16);
    simdgroup_store(acc10, shared_out + 8 * 16 + 0, 16);
    simdgroup_store(acc11, shared_out + 8 * 16 + 8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = simd_lane_id; idx < 16 * 16; idx += 32) {
        uint lr = idx / 16;
        uint lc = idx % 16;
        uint gr = row_base + sg_row + lr;
        uint gc = col_base + sg_col + lc;
        if (gr < M && gc < N) {
            C[gr * ldc + gc] = float_to_bf16(shared_out[lr * 16 + lc]);
        }
    }
}

// ============================================================================
// Batched BF16 kernel
// ============================================================================

kernel void matmul_simd_batched_bf16(
    device const ushort* A    [[buffer(0)]],
    device const ushort* B    [[buffer(1)]],
    device ushort* C          [[buffer(2)]],
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
    threadgroup float* shared_A [[threadgroup(0)]],
    threadgroup float* shared_B [[threadgroup(1)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_idx = gid.z;
    const uint row_base = gid.y * TILE_M;
    const uint col_base = gid.x * TILE_N;

    device const ushort* A_batch = A + batch_idx * batch_stride_a;
    device const ushort* B_batch = B + batch_idx * batch_stride_b;
    device ushort* C_batch = C + batch_idx * batch_stride_c;

    simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc11 = simdgroup_float8x8(0.0f);

    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 16;

    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_M * TILE_K; idx += 128) {
            uint r = idx / TILE_K;
            uint c = idx % TILE_K;
            uint global_r = row_base + r;
            uint global_c = k_tile + c;
            if (global_r < M && global_c < K) {
                if (transpose_a) {
                    shared_A[r * TILE_K + c] = bf16_to_float(A_batch[global_c * lda + global_r]);
                } else {
                    shared_A[r * TILE_K + c] = bf16_to_float(A_batch[global_r * lda + global_c]);
                }
            } else {
                shared_A[r * TILE_K + c] = 0.0f;
            }
        }

        for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_K * TILE_N; idx += 128) {
            uint r = idx / TILE_N;
            uint c = idx % TILE_N;
            uint global_r = k_tile + r;
            uint global_c = col_base + c;
            if (global_r < K && global_c < N) {
                if (transpose_b) {
                    shared_B[r * TILE_N + c] = bf16_to_float(B_batch[global_c * ldb + global_r]);
                } else {
                    shared_B[r * TILE_N + c] = bf16_to_float(B_batch[global_r * ldb + global_c]);
                }
            } else {
                shared_B[r * TILE_N + c] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_K; k += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, shared_A + (sg_row + 0) * TILE_K + k, TILE_K);
            simdgroup_load(a1, shared_A + (sg_row + 8) * TILE_K + k, TILE_K);
            simdgroup_load(b0, shared_B + k * TILE_N + (sg_col + 0), TILE_N);
            simdgroup_load(b1, shared_B + k * TILE_N + (sg_col + 8), TILE_N);

            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float shared_out[16 * 16];

    simdgroup_store(acc00, shared_out + 0 * 16 + 0, 16);
    simdgroup_store(acc01, shared_out + 0 * 16 + 8, 16);
    simdgroup_store(acc10, shared_out + 8 * 16 + 0, 16);
    simdgroup_store(acc11, shared_out + 8 * 16 + 8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = simd_lane_id; idx < 16 * 16; idx += 32) {
        uint lr = idx / 16;
        uint lc = idx % 16;
        uint gr = row_base + sg_row + lr;
        uint gc = col_base + sg_col + lc;
        if (gr < M && gc < N) {
            C_batch[gr * ldc + gc] = float_to_bf16(shared_out[lr * 16 + lc]);
        }
    }
}
