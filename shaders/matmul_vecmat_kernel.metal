/// @file matmul_vecmat_kernel.metal
/// @brief Optimized vector-matrix multiplication for LLM decode phase (M=1 or small batch).
///
/// This kernel is specialized for the decode phase where M=1 (single token generation).
/// During decode, all projections become vector-matrix multiplies, which are completely
/// memory-bandwidth-bound. This kernel avoids the overhead of 32x32 tiling and uses
/// direct vectorized loads with SIMD reduction.
///
/// Performance targets:
/// - M=1, N=4096, K=4096 FP16: 1.3x+ over MPSGraph
/// - M=1, N=11008, K=4096 FP16 (Llama FFN): 1.2x+
///
/// Key optimizations:
/// - No threadgroup memory needed (A is just one row)
/// - Vectorized loads: half4 (8 bytes, 4 FP16 elements)
/// - FP32 accumulation for numerical stability
/// - SIMD-wide reduction using simd_sum()
/// - Each SIMD group (32 threads) computes one output element

#include <metal_stdlib>
using namespace metal;

// Configuration for M=1 decode
constant constexpr uint THREADS_PER_TG = 256;  // 8 SIMD groups of 32 threads
constant constexpr uint SIMD_SIZE = 32;
constant constexpr uint VEC_SIZE = 4;  // half4 / float4 vectorization

// ============================================================================
// FP16 Vector-Matrix (M=1, single token decode)
// ============================================================================

kernel void matmul_vecmat_fp16(
    device const half* A      [[buffer(0)]],   // Input vector [1, K]
    device const half* B      [[buffer(1)]],   // Weight matrix [N, K]
    device half* C            [[buffer(2)]],   // Output vector [1, N]
    constant uint& M          [[buffer(3)]],   // Should be 1
    constant uint& N          [[buffer(4)]],   // Output dimension
    constant uint& K          [[buffer(5)]],   // Input dimension
    constant uint& lda        [[buffer(6)]],   // Leading dim of A (=K)
    constant uint& ldb        [[buffer(7)]],   // Leading dim of B (=K for row-major)
    constant uint& ldc        [[buffer(8)]],   // Leading dim of C (=N)
    constant bool& transpose_a [[buffer(9)]],  // Should be false for [1,K]
    constant bool& transpose_b [[buffer(10)]], // True if B is [K,N] instead of [N,K]
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup computes 8 output elements (one per SIMD group)
    const uint outputs_per_tg = THREADS_PER_TG / SIMD_SIZE;  // 8
    const uint output_base = gid.x * outputs_per_tg;
    const uint output_idx = output_base + simd_group_id;

    // Bounds check
    if (output_idx >= N) return;

    // Each thread in the SIMD group computes a partial sum
    float partial_sum = 0.0f;

    if (!transpose_b) {
        // B is [N, K] row-major: each row is one output neuron
        device const half* weight_row = B + output_idx * ldb;

        // Vectorized accumulation: each thread processes VEC_SIZE elements at a time
        // Total threads: 32, each processes 4 elements per iteration
        // Stride: 32 * 4 = 128 elements between iterations
        uint base_k = simd_lane_id * VEC_SIZE;
        const uint stride = SIMD_SIZE * VEC_SIZE;
        const uint stride2 = stride * 2;

        // 2x unrolled main loop
        uint k = base_k;
        for (; k + VEC_SIZE + stride <= K; k += stride2) {
            half4 a_vec0 = *((device const half4*)(A + k));
            half4 w_vec0 = *((device const half4*)(weight_row + k));
            half4 a_vec1 = *((device const half4*)(A + k + stride));
            half4 w_vec1 = *((device const half4*)(weight_row + k + stride));

            float4 a_f32_0 = float4(a_vec0);
            float4 w_f32_0 = float4(w_vec0);
            float4 a_f32_1 = float4(a_vec1);
            float4 w_f32_1 = float4(w_vec1);

            partial_sum += dot(a_f32_0, w_f32_0);
            partial_sum += dot(a_f32_1, w_f32_1);
        }

        // Handle remainder with single unroll
        for (; k + VEC_SIZE <= K; k += stride) {
            half4 a_vec = *((device const half4*)(A + k));
            half4 w_vec = *((device const half4*)(weight_row + k));

            float4 a_f32 = float4(a_vec);
            float4 w_f32 = float4(w_vec);
            partial_sum += dot(a_f32, w_f32);
        }

        // Handle remaining elements (K % 4 != 0)
        if (k < K) {
            for (uint i = k; i < K && i < k + VEC_SIZE; ++i) {
                partial_sum += float(A[i]) * float(weight_row[i]);
            }
        }
    } else {
        // B is [K, N] column-major (transposed): strided access for each output
        device const half* weight_col = B + output_idx;

        uint k = simd_lane_id * VEC_SIZE;
        const uint stride = SIMD_SIZE * VEC_SIZE;

        for (; k + VEC_SIZE <= K; k += stride) {
            // Manual gather for vectorized strided access
            half4 a_vec = *((device const half4*)(A + k));
            half4 w_vec;
            w_vec[0] = weight_col[(k + 0) * ldb];
            w_vec[1] = weight_col[(k + 1) * ldb];
            w_vec[2] = weight_col[(k + 2) * ldb];
            w_vec[3] = weight_col[(k + 3) * ldb];

            float4 a_f32 = float4(a_vec);
            float4 w_f32 = float4(w_vec);
            partial_sum += dot(a_f32, w_f32);
        }

        // Handle remainder
        if (k < K) {
            for (uint i = k; i < K && i < k + VEC_SIZE; ++i) {
                partial_sum += float(A[i]) * float(weight_col[i * ldb]);
            }
        }
    }

    // SIMD-wide reduction: sum across all 32 threads in the SIMD group
    float result = simd_sum(partial_sum);

    // Thread 0 of each SIMD group writes the result
    if (simd_lane_id == 0) {
        C[output_idx] = half(result);
    }
}

// ============================================================================
// FP16 Vector-Matrix Multi-Output (M=1, each SIMD group computes 2 outputs)
// ============================================================================

kernel void matmul_vecmat_fp16_multi2(
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
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup: 256 threads = 8 SIMD groups
    // Each SIMD group computes 2 output elements
    // Total: 16 outputs per threadgroup (vs 8 in original)
    const uint outputs_per_simd = 2;
    const uint outputs_per_tg = (THREADS_PER_TG / SIMD_SIZE) * outputs_per_simd;  // 16
    const uint output_base = gid.x * outputs_per_tg;
    const uint out_base = output_base + simd_group_id * outputs_per_simd;

    uint out_idx0 = out_base;
    uint out_idx1 = out_base + 1;

    // Early exit if first output is out of bounds
    if (out_idx0 >= N) return;

    float partial_sum0 = 0.0f;
    float partial_sum1 = 0.0f;

    if (!transpose_b) {
        // B is [N, K] row-major
        device const half* weight_row0 = B + out_idx0 * ldb;
        device const half* weight_row1 = (out_idx1 < N) ? (B + out_idx1 * ldb) : nullptr;

        uint base_k = simd_lane_id * VEC_SIZE;
        const uint stride = SIMD_SIZE * VEC_SIZE;
        const uint stride2 = stride * 2;

        // 2x unrolled main loop
        uint k = base_k;
        for (; k + VEC_SIZE + stride <= K; k += stride2) {
            half4 a_vec0 = *((device const half4*)(A + k));
            half4 a_vec1 = *((device const half4*)(A + k + stride));

            half4 w0_vec0 = *((device const half4*)(weight_row0 + k));
            half4 w0_vec1 = *((device const half4*)(weight_row0 + k + stride));

            partial_sum0 += dot(float4(a_vec0), float4(w0_vec0));
            partial_sum0 += dot(float4(a_vec1), float4(w0_vec1));

            if (weight_row1) {
                half4 w1_vec0 = *((device const half4*)(weight_row1 + k));
                half4 w1_vec1 = *((device const half4*)(weight_row1 + k + stride));
                partial_sum1 += dot(float4(a_vec0), float4(w1_vec0));
                partial_sum1 += dot(float4(a_vec1), float4(w1_vec1));
            }
        }

        // Remainder loop (single unroll)
        for (; k + VEC_SIZE <= K; k += stride) {
            half4 a_vec = *((device const half4*)(A + k));
            half4 w0_vec = *((device const half4*)(weight_row0 + k));
            partial_sum0 += dot(float4(a_vec), float4(w0_vec));

            if (weight_row1) {
                half4 w1_vec = *((device const half4*)(weight_row1 + k));
                partial_sum1 += dot(float4(a_vec), float4(w1_vec));
            }
        }

        // Handle remaining elements (K % 4 != 0)
        if (k < K) {
            for (uint i = k; i < K && i < k + VEC_SIZE; ++i) {
                partial_sum0 += float(A[i]) * float(weight_row0[i]);
                if (weight_row1) {
                    partial_sum1 += float(A[i]) * float(weight_row1[i]);
                }
            }
        }
    } else {
        // B is [K, N] column-major (transposed): strided access
        device const half* weight_col0 = B + out_idx0;
        device const half* weight_col1 = (out_idx1 < N) ? (B + out_idx1) : nullptr;

        uint base_k = simd_lane_id * VEC_SIZE;
        const uint stride = SIMD_SIZE * VEC_SIZE;

        for (uint k = base_k; k + VEC_SIZE <= K; k += stride) {
            half4 a_vec = *((device const half4*)(A + k));

            // Manual gather for output 0
            half4 w0_vec;
            w0_vec[0] = weight_col0[(k + 0) * ldb];
            w0_vec[1] = weight_col0[(k + 1) * ldb];
            w0_vec[2] = weight_col0[(k + 2) * ldb];
            w0_vec[3] = weight_col0[(k + 3) * ldb];
            partial_sum0 += dot(float4(a_vec), float4(w0_vec));

            if (weight_col1) {
                half4 w1_vec;
                w1_vec[0] = weight_col1[(k + 0) * ldb];
                w1_vec[1] = weight_col1[(k + 1) * ldb];
                w1_vec[2] = weight_col1[(k + 2) * ldb];
                w1_vec[3] = weight_col1[(k + 3) * ldb];
                partial_sum1 += dot(float4(a_vec), float4(w1_vec));
            }
        }

        // Handle remainder
        uint k_rem = base_k + ((K - base_k) / stride) * stride;
        if (k_rem < K) {
            for (uint i = k_rem; i < K && i < k_rem + VEC_SIZE; ++i) {
                partial_sum0 += float(A[i]) * float(weight_col0[i * ldb]);
                if (weight_col1) {
                    partial_sum1 += float(A[i]) * float(weight_col1[i * ldb]);
                }
            }
        }
    }

    // SIMD-wide reduction
    float result0 = simd_sum(partial_sum0);
    float result1 = simd_sum(partial_sum1);

    // Thread 0 of each SIMD group writes results
    if (simd_lane_id == 0) {
        C[out_idx0] = half(result0);
        if (out_idx1 < N) {
            C[out_idx1] = half(result1);
        }
    }
}

// ============================================================================
// FP32 Vector-Matrix (M=1, single token decode)
// ============================================================================

kernel void matmul_vecmat_fp32(
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
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint outputs_per_tg = THREADS_PER_TG / SIMD_SIZE;
    const uint output_base = gid.x * outputs_per_tg;
    const uint output_idx = output_base + simd_group_id;

    if (output_idx >= N) return;

    float partial_sum = 0.0f;

    if (!transpose_b) {
        device const float* weight_row = B + output_idx * ldb;

        uint base_k = simd_lane_id * VEC_SIZE;
        const uint stride = SIMD_SIZE * VEC_SIZE;
        const uint stride2 = stride * 2;

        // 2x unrolled main loop
        uint k = base_k;
        for (; k + VEC_SIZE + stride <= K; k += stride2) {
            float4 a_vec0 = *((device const float4*)(A + k));
            float4 w_vec0 = *((device const float4*)(weight_row + k));
            float4 a_vec1 = *((device const float4*)(A + k + stride));
            float4 w_vec1 = *((device const float4*)(weight_row + k + stride));

            partial_sum += dot(a_vec0, w_vec0);
            partial_sum += dot(a_vec1, w_vec1);
        }

        // Remainder loop (single unroll)
        for (; k + VEC_SIZE <= K; k += stride) {
            float4 a_vec = *((device const float4*)(A + k));
            float4 w_vec = *((device const float4*)(weight_row + k));
            partial_sum += dot(a_vec, w_vec);
        }

        if (k < K) {
            for (uint i = k; i < K && i < k + VEC_SIZE; ++i) {
                partial_sum += A[i] * weight_row[i];
            }
        }
    } else {
        device const float* weight_col = B + output_idx;

        uint k = simd_lane_id * VEC_SIZE;
        const uint stride = SIMD_SIZE * VEC_SIZE;

        for (; k + VEC_SIZE <= K; k += stride) {
            float4 a_vec = *((device const float4*)(A + k));
            float4 w_vec;
            w_vec[0] = weight_col[(k + 0) * ldb];
            w_vec[1] = weight_col[(k + 1) * ldb];
            w_vec[2] = weight_col[(k + 2) * ldb];
            w_vec[3] = weight_col[(k + 3) * ldb];
            partial_sum += dot(a_vec, w_vec);
        }

        if (k < K) {
            for (uint i = k; i < K && i < k + VEC_SIZE; ++i) {
                partial_sum += A[i] * weight_col[i * ldb];
            }
        }
    }

    float result = simd_sum(partial_sum);

    if (simd_lane_id == 0) {
        C[output_idx] = result;
    }
}

// ============================================================================
// Small Batch Vector-Matrix (M=1..8, multi-token decode or small batch)
// ============================================================================

kernel void matmul_vecmat_batch_fp16(
    device const half* A      [[buffer(0)]],   // Input [M, K] where M is small (1-8)
    device const half* B      [[buffer(1)]],   // Weight matrix [N, K]
    device half* C            [[buffer(2)]],   // Output [M, N]
    constant uint& M          [[buffer(3)]],   // Batch size (1-8)
    constant uint& N          [[buffer(4)]],   // Output dimension
    constant uint& K          [[buffer(5)]],   // Input dimension
    constant uint& lda        [[buffer(6)]],
    constant uint& ldb        [[buffer(7)]],
    constant uint& ldc        [[buffer(8)]],
    constant bool& transpose_a [[buffer(9)]],
    constant bool& transpose_b [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Grid: (ceil(N/8), M, 1)
    // Each threadgroup handles one (batch_row, output_range) pair
    const uint batch_idx = gid.y;  // Which row of A (0..M-1)
    const uint outputs_per_tg = THREADS_PER_TG / SIMD_SIZE;
    const uint output_base = gid.x * outputs_per_tg;
    const uint output_idx = output_base + simd_group_id;

    if (batch_idx >= M || output_idx >= N) return;

    // Get input row for this batch element
    device const half* input_row = A + batch_idx * lda;

    float partial_sum = 0.0f;

    if (!transpose_b) {
        device const half* weight_row = B + output_idx * ldb;

        uint k = simd_lane_id * VEC_SIZE;
        const uint stride = SIMD_SIZE * VEC_SIZE;

        for (; k + VEC_SIZE <= K; k += stride) {
            half4 a_vec = *((device const half4*)(input_row + k));
            half4 w_vec = *((device const half4*)(weight_row + k));

            float4 a_f32 = float4(a_vec);
            float4 w_f32 = float4(w_vec);
            partial_sum += dot(a_f32, w_f32);
        }

        if (k < K) {
            for (uint i = k; i < K && i < k + VEC_SIZE; ++i) {
                partial_sum += float(input_row[i]) * float(weight_row[i]);
            }
        }
    } else {
        device const half* weight_col = B + output_idx;

        uint k = simd_lane_id * VEC_SIZE;
        const uint stride = SIMD_SIZE * VEC_SIZE;

        for (; k + VEC_SIZE <= K; k += stride) {
            half4 a_vec = *((device const half4*)(input_row + k));
            half4 w_vec;
            w_vec[0] = weight_col[(k + 0) * ldb];
            w_vec[1] = weight_col[(k + 1) * ldb];
            w_vec[2] = weight_col[(k + 2) * ldb];
            w_vec[3] = weight_col[(k + 3) * ldb];

            float4 a_f32 = float4(a_vec);
            float4 w_f32 = float4(w_vec);
            partial_sum += dot(a_f32, w_f32);
        }

        if (k < K) {
            for (uint i = k; i < K && i < k + VEC_SIZE; ++i) {
                partial_sum += float(input_row[i]) * float(weight_col[i * ldb]);
            }
        }
    }

    float result = simd_sum(partial_sum);

    if (simd_lane_id == 0) {
        C[batch_idx * ldc + output_idx] = half(result);
    }
}

// ============================================================================
// Small Batch Vector-Matrix FP32 (M=1..8)
// ============================================================================

kernel void matmul_vecmat_batch_fp32(
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
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_idx = gid.y;
    const uint outputs_per_tg = THREADS_PER_TG / SIMD_SIZE;
    const uint output_base = gid.x * outputs_per_tg;
    const uint output_idx = output_base + simd_group_id;

    if (batch_idx >= M || output_idx >= N) return;

    device const float* input_row = A + batch_idx * lda;

    float partial_sum = 0.0f;

    if (!transpose_b) {
        device const float* weight_row = B + output_idx * ldb;

        uint k = simd_lane_id * VEC_SIZE;
        const uint stride = SIMD_SIZE * VEC_SIZE;

        for (; k + VEC_SIZE <= K; k += stride) {
            float4 a_vec = *((device const float4*)(input_row + k));
            float4 w_vec = *((device const float4*)(weight_row + k));
            partial_sum += dot(a_vec, w_vec);
        }

        if (k < K) {
            for (uint i = k; i < K && i < k + VEC_SIZE; ++i) {
                partial_sum += input_row[i] * weight_row[i];
            }
        }
    } else {
        device const float* weight_col = B + output_idx;

        uint k = simd_lane_id * VEC_SIZE;
        const uint stride = SIMD_SIZE * VEC_SIZE;

        for (; k + VEC_SIZE <= K; k += stride) {
            float4 a_vec = *((device const float4*)(input_row + k));
            float4 w_vec;
            w_vec[0] = weight_col[(k + 0) * ldb];
            w_vec[1] = weight_col[(k + 1) * ldb];
            w_vec[2] = weight_col[(k + 2) * ldb];
            w_vec[3] = weight_col[(k + 3) * ldb];
            partial_sum += dot(a_vec, w_vec);
        }

        if (k < K) {
            for (uint i = k; i < K && i < k + VEC_SIZE; ++i) {
                partial_sum += input_row[i] * weight_col[i * ldb];
            }
        }
    }

    float result = simd_sum(partial_sum);

    if (simd_lane_id == 0) {
        C[batch_idx * ldc + output_idx] = result;
    }
}
