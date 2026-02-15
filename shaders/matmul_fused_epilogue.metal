/// @file matmul_fused_epilogue.metal
/// @brief 64x64 tiled matmul with fused epilogue operations.
///
/// Epilogue fusion eliminates memory round-trips by applying bias, activation,
/// and residual operations during the matmul output write step.
///
/// Supported epilogues:
/// - EPILOGUE_NONE (0): C = A @ B
/// - EPILOGUE_BIAS (1): C = A @ B + bias
/// - EPILOGUE_BIAS_RELU (2): C = ReLU(A @ B + bias)
/// - EPILOGUE_BIAS_SILU (3): C = SiLU(A @ B + bias)
/// - EPILOGUE_RESIDUAL (4): C = A @ B + residual
///
/// Performance target: 1.1-1.2x speedup over separate ops (eliminates memory round-trip)

#include <metal_stdlib>
using namespace metal;

// Tile configuration (same as matmul_tiled64_kernel.metal)
constant constexpr uint TILE_M = 64;
constant constexpr uint TILE_N = 64;
constant constexpr uint TILE_K = 32;

// Epilogue type constants
constant constexpr uint EPILOGUE_NONE = 0;
constant constexpr uint EPILOGUE_BIAS = 1;
constant constexpr uint EPILOGUE_BIAS_RELU = 2;
constant constexpr uint EPILOGUE_BIAS_SILU = 3;
constant constexpr uint EPILOGUE_RESIDUAL = 4;

// Function constants for compile-time specialization
constant uint EPILOGUE_TYPE [[function_constant(2)]];
constant bool HAS_EPILOGUE = (EPILOGUE_TYPE != EPILOGUE_NONE);

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
    uint tid
) {
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
// Epilogue helper: Apply epilogue operation to accumulated value
// ============================================================================

template<typename T>
inline float apply_epilogue(
    float accumulated_value,
    uint global_row,
    uint global_col,
    device const T* bias,
    device const T* residual,
    uint ldc
) {
    float val = accumulated_value;

    if (EPILOGUE_TYPE == EPILOGUE_BIAS) {
        // Bias is [N] shaped, broadcast across M dimension
        val += float(bias[global_col]);
    } else if (EPILOGUE_TYPE == EPILOGUE_BIAS_RELU) {
        val += float(bias[global_col]);
        val = max(0.0f, val);
    } else if (EPILOGUE_TYPE == EPILOGUE_BIAS_SILU) {
        val += float(bias[global_col]);
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        val = val / (1.0f + exp(-val));
    } else if (EPILOGUE_TYPE == EPILOGUE_RESIDUAL) {
        // Residual is [M, N] shaped, element-wise add
        val += float(residual[global_row * ldc + global_col]);
    }

    return val;
}

// ============================================================================
// FP16 kernel with FP32 accumulation and fused epilogue
// ============================================================================

kernel void matmul_fused_epilogue_fp16(
    device const half* A          [[buffer(0)]],
    device const half* B          [[buffer(1)]],
    device half* C                [[buffer(2)]],
    constant uint& M              [[buffer(3)]],
    constant uint& N              [[buffer(4)]],
    constant uint& K              [[buffer(5)]],
    constant uint& lda            [[buffer(6)]],
    constant uint& ldb            [[buffer(7)]],
    constant uint& ldc            [[buffer(8)]],
    constant bool& transpose_a    [[buffer(9)]],
    constant bool& transpose_b    [[buffer(10)]],
    // Epilogue buffers (conditionally bound via function constants)
    device const half* bias       [[buffer(14), function_constant(HAS_EPILOGUE)]],
    device const half* residual   [[buffer(15), function_constant(HAS_EPILOGUE)]],
    threadgroup half* shared_A_buf [[threadgroup(0)]],
    threadgroup half* shared_B_buf [[threadgroup(1)]],
    uint3 gid                     [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint simd_lane_id             [[thread_index_in_simdgroup]],
    uint simd_group_id            [[simdgroup_index_in_threadgroup]]
) {
    const uint row_base = gid.y * TILE_M;
    const uint col_base = gid.x * TILE_N;

    // 8 SIMD groups in 4x2 grid, each computes 16x32 block
    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 32;

    // 4x4 grid of 8x8 accumulators per SIMD group
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }

    // Double-buffered shared memory
    threadgroup half* shared_A[2] = {
        shared_A_buf,
        shared_A_buf + TILE_M * TILE_K
    };
    threadgroup half* shared_B[2] = {
        shared_B_buf,
        shared_B_buf + TILE_K * TILE_N
    };

    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Load first tile
    load_tile_A(A, shared_A[0], row_base, 0, M, K, lda, transpose_a, tid);
    load_tile_B(B, shared_B[0], col_base, 0, N, K, ldb, transpose_b, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main K-tile loop with double buffering
    for (uint k_tile_idx = 0; k_tile_idx < num_k_tiles; ++k_tile_idx) {
        uint buf_idx = k_tile_idx % 2;
        uint next_buf_idx = (k_tile_idx + 1) % 2;

        // Prefetch next tile
        if (k_tile_idx + 1 < num_k_tiles) {
            uint next_k_tile = (k_tile_idx + 1) * TILE_K;
            load_tile_A(A, shared_A[next_buf_idx], row_base, next_k_tile, M, K, lda, transpose_a, tid);
            load_tile_B(B, shared_B[next_buf_idx], col_base, next_k_tile, N, K, ldb, transpose_b, tid);
        }

        // Compute on current buffer
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

    // ========================================================================
    // Output write with FUSED EPILOGUE
    // ========================================================================

    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            uint gr = row_base + sg_row + i * 8;
            uint gc = col_base + sg_col + j * 8;

            // Store accumulator to temp buffer for conversion + epilogue
            threadgroup float temp_f32[8 * 8];
            threadgroup half temp_f16[8 * 8];

            simdgroup_store(acc[i][j], temp_f32, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Apply epilogue and convert FP32 -> FP16
            for (uint t = simd_lane_id; t < 64; t += 32) {
                uint lr = t / 8;
                uint lc = t % 8;
                uint elem_gr = gr + lr;
                uint elem_gc = gc + lc;

                float val = temp_f32[t];

                // Apply epilogue if within bounds
                if (elem_gr < M && elem_gc < N && HAS_EPILOGUE) {
                    val = apply_epilogue(val, elem_gr, elem_gc, bias, residual, ldc);
                }

                temp_f16[t] = half(val);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Store result
            if (gr + 7 < M && gc + 7 < N) {
                simdgroup_half8x8 result;
                simdgroup_load(result, temp_f16, 8);
                simdgroup_store(result, C + gr * ldc + gc, ldc);
            } else {
                // Handle boundary with scalar writes
                for (uint t = simd_lane_id; t < 64; t += 32) {
                    uint lr = t / 8;
                    uint lc = t % 8;
                    uint global_r = gr + lr;
                    uint global_c = gc + lc;
                    if (global_r < M && global_c < N) {
                        C[global_r * ldc + global_c] = temp_f16[lr * 8 + lc];
                    }
                }
            }
        }
    }
}

// ============================================================================
// FP32 kernel with fused epilogue
// ============================================================================

kernel void matmul_fused_epilogue_fp32(
    device const float* A         [[buffer(0)]],
    device const float* B         [[buffer(1)]],
    device float* C               [[buffer(2)]],
    constant uint& M              [[buffer(3)]],
    constant uint& N              [[buffer(4)]],
    constant uint& K              [[buffer(5)]],
    constant uint& lda            [[buffer(6)]],
    constant uint& ldb            [[buffer(7)]],
    constant uint& ldc            [[buffer(8)]],
    constant bool& transpose_a    [[buffer(9)]],
    constant bool& transpose_b    [[buffer(10)]],
    device const float* bias      [[buffer(14), function_constant(HAS_EPILOGUE)]],
    device const float* residual  [[buffer(15), function_constant(HAS_EPILOGUE)]],
    threadgroup float* shared_A_buf [[threadgroup(0)]],
    threadgroup float* shared_B_buf [[threadgroup(1)]],
    uint3 gid                     [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint simd_lane_id             [[thread_index_in_simdgroup]],
    uint simd_group_id            [[simdgroup_index_in_threadgroup]]
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

    // Output write with fused epilogue
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            uint gr = row_base + sg_row + i * 8;
            uint gc = col_base + sg_col + j * 8;

            threadgroup float temp_out[8 * 8];
            simdgroup_store(acc[i][j], temp_out, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Apply epilogue
            for (uint t = simd_lane_id; t < 64; t += 32) {
                uint lr = t / 8;
                uint lc = t % 8;
                uint elem_gr = gr + lr;
                uint elem_gc = gc + lc;

                if (elem_gr < M && elem_gc < N && HAS_EPILOGUE) {
                    temp_out[t] = apply_epilogue(temp_out[t], elem_gr, elem_gc, bias, residual, ldc);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (gr + 7 < M && gc + 7 < N) {
                simdgroup_float8x8 result;
                simdgroup_load(result, temp_out, 8);
                simdgroup_store(result, C + gr * ldc + gc, ldc);
            } else {
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
// Vec-mat fused epilogue (M=1 decode phase)
// ============================================================================

constant constexpr uint VECMAT_THREADS_PER_TG = 256;
constant constexpr uint VECMAT_SIMD_SIZE = 32;
constant constexpr uint VECMAT_VEC_SIZE = 4;

kernel void matmul_vecmat_fused_epilogue_fp16(
    device const half* A          [[buffer(0)]],
    device const half* B          [[buffer(1)]],
    device half* C                [[buffer(2)]],
    constant uint& M              [[buffer(3)]],
    constant uint& N              [[buffer(4)]],
    constant uint& K              [[buffer(5)]],
    constant uint& lda            [[buffer(6)]],
    constant uint& ldb            [[buffer(7)]],
    constant uint& ldc            [[buffer(8)]],
    constant bool& transpose_a    [[buffer(9)]],
    constant bool& transpose_b    [[buffer(10)]],
    device const half* bias       [[buffer(14), function_constant(HAS_EPILOGUE)]],
    device const half* residual   [[buffer(15), function_constant(HAS_EPILOGUE)]],
    uint3 gid                     [[threadgroup_position_in_grid]],
    uint simd_lane_id             [[thread_index_in_simdgroup]],
    uint simd_group_id            [[simdgroup_index_in_threadgroup]]
) {
    const uint outputs_per_tg = VECMAT_THREADS_PER_TG / VECMAT_SIMD_SIZE;
    const uint output_base = gid.x * outputs_per_tg;
    const uint output_idx = output_base + simd_group_id;

    if (output_idx >= N) return;

    float partial_sum = 0.0f;

    if (!transpose_b) {
        device const half* weight_row = B + output_idx * ldb;

        uint k = simd_lane_id * VECMAT_VEC_SIZE;
        const uint stride = VECMAT_SIMD_SIZE * VECMAT_VEC_SIZE;

        for (; k + VECMAT_VEC_SIZE <= K; k += stride) {
            half4 a_vec = *((device const half4*)(A + k));
            half4 w_vec = *((device const half4*)(weight_row + k));

            float4 a_f32 = float4(a_vec);
            float4 w_f32 = float4(w_vec);
            partial_sum += dot(a_f32, w_f32);
        }

        if (k < K) {
            for (uint i = k; i < K && i < k + VECMAT_VEC_SIZE; ++i) {
                partial_sum += float(A[i]) * float(weight_row[i]);
            }
        }
    } else {
        device const half* weight_col = B + output_idx;

        uint k = simd_lane_id * VECMAT_VEC_SIZE;
        const uint stride = VECMAT_SIMD_SIZE * VECMAT_VEC_SIZE;

        for (; k + VECMAT_VEC_SIZE <= K; k += stride) {
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

        if (k < K) {
            for (uint i = k; i < K && i < k + VECMAT_VEC_SIZE; ++i) {
                partial_sum += float(A[i]) * float(weight_col[i * ldb]);
            }
        }
    }

    float result = simd_sum(partial_sum);

    // Apply epilogue (vec-mat is always M=1, so row=0)
    if (simd_lane_id == 0) {
        if (HAS_EPILOGUE) {
            result = apply_epilogue(result, 0, output_idx, bias, residual, ldc);
        }
        C[output_idx] = half(result);
    }
}
