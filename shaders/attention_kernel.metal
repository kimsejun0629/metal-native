/// @file attention_kernel.metal
/// @brief Metal kernel implementations for FlashAttention with SIMD matrix operations.

#include <metal_stdlib>
using namespace metal;

// Tile size constant
constant constexpr uint TILE_SIZE = 16;

/// Online softmax state for numerically stable softmax computation
struct OnlineSoftmaxState {
    float max_val;
    float sum_exp;
};

/// Update online softmax state with a new score
inline OnlineSoftmaxState update_online_softmax(OnlineSoftmaxState state, float score) {
    float new_max = max(state.max_val, score);
    float exp_diff = exp(state.max_val - new_max);
    float new_sum = state.sum_exp * exp_diff + exp(score - new_max);
    return {new_max, new_sum};
}

/// FlashAttention kernel using SIMD matrix operations for FP32
/// This version uses simdgroup_float8x8 with AMX acceleration
///
/// Memory layout:
/// - shared_Q: [TILE_SIZE, head_dim] in threadgroup memory
/// - shared_KV: [TILE_SIZE, head_dim] in threadgroup memory (K then V, aliased)
/// - shared_scores: [TILE_SIZE, TILE_SIZE] in threadgroup memory
/// - shared_output: [TILE_SIZE, head_dim] in threadgroup memory (FP32 accumulator)
///
/// Threadgroup: 32 threads (one SIMD group)
/// Grid: [num_q_tiles, num_heads, batch]
kernel void flash_attention_simd_kernel(
    device const float* Q               [[buffer(0)]],
    device const float* K               [[buffer(1)]],
    device const float* V               [[buffer(2)]],
    device const float* mask            [[buffer(3)]],
    device float* output                [[buffer(4)]],
    constant uint& batch                [[buffer(5)]],
    constant uint& num_heads            [[buffer(6)]],
    constant uint& seq_len_q            [[buffer(7)]],
    constant uint& seq_len_k            [[buffer(8)]],
    constant uint& head_dim             [[buffer(9)]],
    constant float& scale               [[buffer(10)]],
    constant bool& has_mask             [[buffer(11)]],
    threadgroup float* shared_Q         [[threadgroup(0)]],
    threadgroup float* shared_KV        [[threadgroup(1)]],  // K then V (aliased)
    threadgroup float* shared_scores    [[threadgroup(2)]],
    threadgroup float* shared_output    [[threadgroup(3)]],
    uint3 gid                           [[threadgroup_position_in_grid]],
    uint  simd_lane_id                  [[thread_index_in_simdgroup]],
    uint  simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    const uint b = gid.z;
    const uint h = gid.y;
    const uint q_tile_idx = gid.x;

    if (b >= batch || h >= num_heads) return;

    const uint q_start = q_tile_idx * TILE_SIZE;
    const uint qkv_offset = (b * num_heads + h) * seq_len_q * head_dim;
    const uint k_offset = (b * num_heads + h) * seq_len_k * head_dim;

    // Load Q tile into shared memory
    for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32) {
        uint row = idx / head_dim;
        uint col = idx % head_dim;
        uint q_idx = q_start + row;
        if (q_idx < seq_len_q) {
            shared_Q[row * head_dim + col] = Q[qkv_offset + q_idx * head_dim + col];
        } else {
            shared_Q[row * head_dim + col] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Initialize output accumulator
    for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32) {
        shared_output[idx] = 0.0f;
    }

    // Initialize online softmax state for each query
    threadgroup OnlineSoftmaxState softmax_states[TILE_SIZE];
    if (simd_lane_id < TILE_SIZE) {
        softmax_states[simd_lane_id].max_val = -INFINITY;
        softmax_states[simd_lane_id].sum_exp = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_k_tiles = (seq_len_k + TILE_SIZE - 1) / TILE_SIZE;

    // Process K/V tiles
    for (uint k_tile_idx = 0; k_tile_idx < num_k_tiles; k_tile_idx++) {
        const uint k_start = k_tile_idx * TILE_SIZE;
        const uint k_end = min(k_start + TILE_SIZE, seq_len_k);
        const uint k_tile_size = k_end - k_start;

        // Load K tile
        for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32) {
            uint row = idx / head_dim;
            uint col = idx % head_dim;
            uint k_idx = k_start + row;
            if (k_idx < seq_len_k && row < k_tile_size) {
                shared_KV[row * head_dim + col] = K[k_offset + k_idx * head_dim + col];
            } else {
                shared_KV[row * head_dim + col] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute Q @ K^T using SIMD matrix operations
        // Decompose 16x16 tile into 2x2 blocks of 8x8
        simdgroup_float8x8 scores_00, scores_01, scores_10, scores_11;
        scores_00 = simdgroup_float8x8(0.0f);
        scores_01 = simdgroup_float8x8(0.0f);
        scores_10 = simdgroup_float8x8(0.0f);
        scores_11 = simdgroup_float8x8(0.0f);

        for (uint d = 0; d < head_dim; d += 8) {
            simdgroup_float8x8 q0, q1, k0, k1;
            simdgroup_load(q0, shared_Q + 0 * head_dim + d, head_dim);
            simdgroup_load(q1, shared_Q + 8 * head_dim + d, head_dim);
            simdgroup_load(k0, shared_KV + 0 * head_dim + d, head_dim);
            simdgroup_load(k1, shared_KV + 8 * head_dim + d, head_dim);

            simdgroup_multiply_accumulate(scores_00, q0, k0, scores_00);
            simdgroup_multiply_accumulate(scores_01, q0, k1, scores_01);
            simdgroup_multiply_accumulate(scores_10, q1, k0, scores_10);
            simdgroup_multiply_accumulate(scores_11, q1, k1, scores_11);
        }

        // Store scores and apply scaling/masking
        simdgroup_store(scores_00, shared_scores + 0 * TILE_SIZE + 0, TILE_SIZE);
        simdgroup_store(scores_01, shared_scores + 0 * TILE_SIZE + 8, TILE_SIZE);
        simdgroup_store(scores_10, shared_scores + 8 * TILE_SIZE + 0, TILE_SIZE);
        simdgroup_store(scores_11, shared_scores + 8 * TILE_SIZE + 8, TILE_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply scaling and mask
        for (uint idx = simd_lane_id; idx < TILE_SIZE * TILE_SIZE; idx += 32) {
            uint row = idx / TILE_SIZE;
            uint col = idx % TILE_SIZE;
            uint q_idx = q_start + row;
            uint k_idx = k_start + col;

            if (q_idx < seq_len_q && col < k_tile_size) {
                float score = shared_scores[row * TILE_SIZE + col] * scale;
                if (has_mask) {
                    score += mask[b * seq_len_q * seq_len_k + q_idx * seq_len_k + k_idx];
                }
                shared_scores[row * TILE_SIZE + col] = score;
            } else {
                shared_scores[row * TILE_SIZE + col] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update online softmax
        if (simd_lane_id < TILE_SIZE) {
            uint q_idx = q_start + simd_lane_id;
            if (q_idx < seq_len_q) {
                float old_max = softmax_states[simd_lane_id].max_val;

                // Update state with all scores in this row
                for (uint k_local = 0; k_local < k_tile_size; k_local++) {
                    float score = shared_scores[simd_lane_id * TILE_SIZE + k_local];
                    softmax_states[simd_lane_id] = update_online_softmax(softmax_states[simd_lane_id], score);
                }

                // Rescale previous accumulator if max changed
                if (softmax_states[simd_lane_id].max_val > old_max) {
                    float rescale = exp(old_max - softmax_states[simd_lane_id].max_val);
                    for (uint d = 0; d < head_dim; d++) {
                        shared_output[simd_lane_id * head_dim + d] *= rescale;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute softmax weights
        for (uint idx = simd_lane_id; idx < TILE_SIZE * TILE_SIZE; idx += 32) {
            uint row = idx / TILE_SIZE;
            uint col = idx % TILE_SIZE;
            uint q_idx = q_start + row;

            if (q_idx < seq_len_q && col < k_tile_size) {
                float score = shared_scores[row * TILE_SIZE + col];
                shared_scores[row * TILE_SIZE + col] = exp(score - softmax_states[row].max_val);
            } else {
                shared_scores[row * TILE_SIZE + col] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load V tile
        for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32) {
            uint row = idx / head_dim;
            uint col = idx % head_dim;
            uint v_idx = k_start + row;
            if (v_idx < seq_len_k && row < k_tile_size) {
                shared_KV[row * head_dim + col] = V[k_offset + v_idx * head_dim + col];
            } else {
                shared_KV[row * head_dim + col] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate weighted V using SIMD operations
        // Process head_dim in blocks of 16 (2x 8x8 blocks)
        for (uint d_block = 0; d_block < head_dim; d_block += 16) {
            simdgroup_float8x8 out_00, out_01, out_10, out_11;
            simdgroup_float8x8 w00, w01, w10, w11;
            simdgroup_float8x8 v0_0, v0_1, v1_0, v1_1;

            // Load output accumulators
            simdgroup_load(out_00, shared_output + 0 * head_dim + d_block + 0, head_dim);
            simdgroup_load(out_01, shared_output + 0 * head_dim + d_block + 8, head_dim);
            simdgroup_load(out_10, shared_output + 8 * head_dim + d_block + 0, head_dim);
            simdgroup_load(out_11, shared_output + 8 * head_dim + d_block + 8, head_dim);

            // Load weights (softmax scores)
            simdgroup_load(w00, shared_scores + 0 * TILE_SIZE + 0, TILE_SIZE);
            simdgroup_load(w01, shared_scores + 0 * TILE_SIZE + 8, TILE_SIZE);
            simdgroup_load(w10, shared_scores + 8 * TILE_SIZE + 0, TILE_SIZE);
            simdgroup_load(w11, shared_scores + 8 * TILE_SIZE + 8, TILE_SIZE);

            // Load V values
            simdgroup_load(v0_0, shared_KV + 0 * head_dim + d_block + 0, head_dim);
            simdgroup_load(v0_1, shared_KV + 0 * head_dim + d_block + 8, head_dim);
            simdgroup_load(v1_0, shared_KV + 8 * head_dim + d_block + 0, head_dim);
            simdgroup_load(v1_1, shared_KV + 8 * head_dim + d_block + 8, head_dim);

            // Accumulate: out += weights @ V
            simdgroup_multiply_accumulate(out_00, w00, v0_0, out_00);
            simdgroup_multiply_accumulate(out_00, w01, v1_0, out_00);
            simdgroup_multiply_accumulate(out_01, w00, v0_1, out_01);
            simdgroup_multiply_accumulate(out_01, w01, v1_1, out_01);
            simdgroup_multiply_accumulate(out_10, w10, v0_0, out_10);
            simdgroup_multiply_accumulate(out_10, w11, v1_0, out_10);
            simdgroup_multiply_accumulate(out_11, w10, v0_1, out_11);
            simdgroup_multiply_accumulate(out_11, w11, v1_1, out_11);

            // Store back
            simdgroup_store(out_00, shared_output + 0 * head_dim + d_block + 0, head_dim);
            simdgroup_store(out_01, shared_output + 0 * head_dim + d_block + 8, head_dim);
            simdgroup_store(out_10, shared_output + 8 * head_dim + d_block + 0, head_dim);
            simdgroup_store(out_11, shared_output + 8 * head_dim + d_block + 8, head_dim);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Final normalization and write output
    for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32) {
        uint row = idx / head_dim;
        uint col = idx % head_dim;
        uint q_idx = q_start + row;

        if (q_idx < seq_len_q) {
            float normalized = shared_output[row * head_dim + col] / softmax_states[row].sum_exp;
            output[qkv_offset + q_idx * head_dim + col] = normalized;
        }
    }
}

/// FlashAttention kernel using SIMD matrix operations for FP16
/// This version uses simdgroup_half8x8 with AMX acceleration but FP32 accumulation
///
/// Memory layout:
/// - shared_Q: [TILE_SIZE, head_dim] FP16
/// - shared_KV: [TILE_SIZE, head_dim] FP16 (K then V, aliased)
/// - shared_scores: [TILE_SIZE, TILE_SIZE] FP32 for numerical stability
/// - shared_output: [TILE_SIZE, head_dim] FP32 accumulation
///
/// Threadgroup: 32 threads (one SIMD group)
/// Grid: [num_q_tiles, num_heads, batch]
kernel void flash_attention_simd_kernel_fp16(
    device const half* Q                [[buffer(0)]],
    device const half* K                [[buffer(1)]],
    device const half* V                [[buffer(2)]],
    device const half* mask             [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& batch                [[buffer(5)]],
    constant uint& num_heads            [[buffer(6)]],
    constant uint& seq_len_q            [[buffer(7)]],
    constant uint& seq_len_k            [[buffer(8)]],
    constant uint& head_dim             [[buffer(9)]],
    constant float& scale               [[buffer(10)]],
    constant bool& has_mask             [[buffer(11)]],
    threadgroup half* shared_Q          [[threadgroup(0)]],
    threadgroup half* shared_KV         [[threadgroup(1)]],  // K then V (aliased)
    threadgroup float* shared_scores    [[threadgroup(2)]],  // FP32 for numerical stability
    threadgroup float* shared_output    [[threadgroup(3)]],  // FP32 accumulation
    uint3 gid                           [[threadgroup_position_in_grid]],
    uint  simd_lane_id                  [[thread_index_in_simdgroup]],
    uint  simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    const uint b = gid.z;
    const uint h = gid.y;
    const uint q_tile_idx = gid.x;

    if (b >= batch || h >= num_heads) return;

    const uint q_start = q_tile_idx * TILE_SIZE;
    const uint qkv_offset = (b * num_heads + h) * seq_len_q * head_dim;
    const uint k_offset = (b * num_heads + h) * seq_len_k * head_dim;

    // Load Q tile
    for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32) {
        uint row = idx / head_dim;
        uint col = idx % head_dim;
        uint q_idx = q_start + row;
        if (q_idx < seq_len_q) {
            shared_Q[row * head_dim + col] = Q[qkv_offset + q_idx * head_dim + col];
        } else {
            shared_Q[row * head_dim + col] = half(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Initialize FP32 output accumulator
    for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32) {
        shared_output[idx] = 0.0f;
    }

    // Online softmax state
    threadgroup OnlineSoftmaxState softmax_states[TILE_SIZE];
    if (simd_lane_id < TILE_SIZE) {
        softmax_states[simd_lane_id].max_val = -INFINITY;
        softmax_states[simd_lane_id].sum_exp = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_k_tiles = (seq_len_k + TILE_SIZE - 1) / TILE_SIZE;

    for (uint k_tile_idx = 0; k_tile_idx < num_k_tiles; k_tile_idx++) {
        const uint k_start = k_tile_idx * TILE_SIZE;
        const uint k_end = min(k_start + TILE_SIZE, seq_len_k);
        const uint k_tile_size = k_end - k_start;

        // Load K tile
        for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32) {
            uint row = idx / head_dim;
            uint col = idx % head_dim;
            uint k_idx = k_start + row;
            if (k_idx < seq_len_k && row < k_tile_size) {
                shared_KV[row * head_dim + col] = K[k_offset + k_idx * head_dim + col];
            } else {
                shared_KV[row * head_dim + col] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute Q @ K^T with FP16 SIMD matrices (uses AMX with FP16 inputs)
        simdgroup_half8x8 scores_00, scores_01, scores_10, scores_11;
        scores_00 = simdgroup_half8x8(half(0.0f));
        scores_01 = simdgroup_half8x8(half(0.0f));
        scores_10 = simdgroup_half8x8(half(0.0f));
        scores_11 = simdgroup_half8x8(half(0.0f));

        for (uint d = 0; d < head_dim; d += 8) {
            simdgroup_half8x8 q0, q1, k0, k1;
            simdgroup_load(q0, shared_Q + 0 * head_dim + d, head_dim);
            simdgroup_load(q1, shared_Q + 8 * head_dim + d, head_dim);
            simdgroup_load(k0, shared_KV + 0 * head_dim + d, head_dim);
            simdgroup_load(k1, shared_KV + 8 * head_dim + d, head_dim);

            simdgroup_multiply_accumulate(scores_00, q0, k0, scores_00);
            simdgroup_multiply_accumulate(scores_01, q0, k1, scores_01);
            simdgroup_multiply_accumulate(scores_10, q1, k0, scores_10);
            simdgroup_multiply_accumulate(scores_11, q1, k1, scores_11);
        }

        // Convert scores to FP32 and store
        threadgroup half temp_scores_half[TILE_SIZE * TILE_SIZE];
        simdgroup_store(scores_00, temp_scores_half + 0 * TILE_SIZE + 0, TILE_SIZE);
        simdgroup_store(scores_01, temp_scores_half + 0 * TILE_SIZE + 8, TILE_SIZE);
        simdgroup_store(scores_10, temp_scores_half + 8 * TILE_SIZE + 0, TILE_SIZE);
        simdgroup_store(scores_11, temp_scores_half + 8 * TILE_SIZE + 8, TILE_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Convert to FP32 and apply scaling/masking
        for (uint idx = simd_lane_id; idx < TILE_SIZE * TILE_SIZE; idx += 32) {
            uint row = idx / TILE_SIZE;
            uint col = idx % TILE_SIZE;
            uint q_idx = q_start + row;
            uint k_idx = k_start + col;

            if (q_idx < seq_len_q && col < k_tile_size) {
                float score = float(temp_scores_half[row * TILE_SIZE + col]) * scale;
                if (has_mask) {
                    score += float(mask[b * seq_len_q * seq_len_k + q_idx * seq_len_k + k_idx]);
                }
                shared_scores[row * TILE_SIZE + col] = score;
            } else {
                shared_scores[row * TILE_SIZE + col] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update online softmax
        if (simd_lane_id < TILE_SIZE) {
            uint q_idx = q_start + simd_lane_id;
            if (q_idx < seq_len_q) {
                float old_max = softmax_states[simd_lane_id].max_val;

                for (uint k_local = 0; k_local < k_tile_size; k_local++) {
                    float score = shared_scores[simd_lane_id * TILE_SIZE + k_local];
                    softmax_states[simd_lane_id] = update_online_softmax(softmax_states[simd_lane_id], score);
                }

                if (softmax_states[simd_lane_id].max_val > old_max) {
                    float rescale = exp(old_max - softmax_states[simd_lane_id].max_val);
                    for (uint d = 0; d < head_dim; d++) {
                        shared_output[simd_lane_id * head_dim + d] *= rescale;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute softmax weights
        for (uint idx = simd_lane_id; idx < TILE_SIZE * TILE_SIZE; idx += 32) {
            uint row = idx / TILE_SIZE;
            uint col = idx % TILE_SIZE;
            uint q_idx = q_start + row;

            if (q_idx < seq_len_q && col < k_tile_size) {
                float score = shared_scores[row * TILE_SIZE + col];
                shared_scores[row * TILE_SIZE + col] = exp(score - softmax_states[row].max_val);
            } else {
                shared_scores[row * TILE_SIZE + col] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load V tile
        for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32) {
            uint row = idx / head_dim;
            uint col = idx % head_dim;
            uint v_idx = k_start + row;
            if (v_idx < seq_len_k && row < k_tile_size) {
                shared_KV[row * head_dim + col] = V[k_offset + v_idx * head_dim + col];
            } else {
                shared_KV[row * head_dim + col] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate weighted V (FP32 accumulation for stability)
        // Convert weights and V to FP16 SIMD matrices, accumulate in FP32
        for (uint d_block = 0; d_block < head_dim; d_block += 16) {
            simdgroup_float8x8 out_00, out_01, out_10, out_11;

            simdgroup_load(out_00, shared_output + 0 * head_dim + d_block + 0, head_dim);
            simdgroup_load(out_01, shared_output + 0 * head_dim + d_block + 8, head_dim);
            simdgroup_load(out_10, shared_output + 8 * head_dim + d_block + 0, head_dim);
            simdgroup_load(out_11, shared_output + 8 * head_dim + d_block + 8, head_dim);

            // Load weights as FP32 then convert for multiply
            threadgroup half temp_weights_half[TILE_SIZE * TILE_SIZE];
            for (uint i = simd_lane_id; i < TILE_SIZE * TILE_SIZE; i += 32) {
                temp_weights_half[i] = half(shared_scores[i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_half8x8 w00, w01, w10, w11;
            simdgroup_load(w00, temp_weights_half + 0 * TILE_SIZE + 0, TILE_SIZE);
            simdgroup_load(w01, temp_weights_half + 0 * TILE_SIZE + 8, TILE_SIZE);
            simdgroup_load(w10, temp_weights_half + 8 * TILE_SIZE + 0, TILE_SIZE);
            simdgroup_load(w11, temp_weights_half + 8 * TILE_SIZE + 8, TILE_SIZE);

            simdgroup_half8x8 v0_0, v0_1, v1_0, v1_1;
            simdgroup_load(v0_0, shared_KV + 0 * head_dim + d_block + 0, head_dim);
            simdgroup_load(v0_1, shared_KV + 0 * head_dim + d_block + 8, head_dim);
            simdgroup_load(v1_0, shared_KV + 8 * head_dim + d_block + 0, head_dim);
            simdgroup_load(v1_1, shared_KV + 8 * head_dim + d_block + 8, head_dim);

            // Mixed precision: FP16 multiply with FP32 accumulate
            simdgroup_multiply_accumulate(out_00, w00, v0_0, out_00);
            simdgroup_multiply_accumulate(out_00, w01, v1_0, out_00);
            simdgroup_multiply_accumulate(out_01, w00, v0_1, out_01);
            simdgroup_multiply_accumulate(out_01, w01, v1_1, out_01);
            simdgroup_multiply_accumulate(out_10, w10, v0_0, out_10);
            simdgroup_multiply_accumulate(out_10, w11, v1_0, out_10);
            simdgroup_multiply_accumulate(out_11, w10, v0_1, out_11);
            simdgroup_multiply_accumulate(out_11, w11, v1_1, out_11);

            simdgroup_store(out_00, shared_output + 0 * head_dim + d_block + 0, head_dim);
            simdgroup_store(out_01, shared_output + 0 * head_dim + d_block + 8, head_dim);
            simdgroup_store(out_10, shared_output + 8 * head_dim + d_block + 0, head_dim);
            simdgroup_store(out_11, shared_output + 8 * head_dim + d_block + 8, head_dim);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Final normalization and write output as FP16
    for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32) {
        uint row = idx / head_dim;
        uint col = idx % head_dim;
        uint q_idx = q_start + row;

        if (q_idx < seq_len_q) {
            float normalized = shared_output[row * head_dim + col] / softmax_states[row].sum_exp;
            output[qkv_offset + q_idx * head_dim + col] = half(normalized);
        }
    }
}

/// FlashAttention kernel using SIMD matrix operations for FP16 with TILE_SIZE=24
/// This version uses 3x3 decomposition of simdgroup_half8x8 matrices
///
/// Memory layout:
/// - shared_Q: [24, head_dim] FP16
/// - shared_KV: [24, head_dim] FP16 (K then V, aliased)
/// - shared_scores: [24, 24] FP32 for numerical stability
/// - shared_output: [24, head_dim] FP32 accumulation
///
/// Threadgroup memory at head_dim=128:
/// - shared_Q: 24 * 128 * 2 = 6,144 bytes
/// - shared_KV: 24 * 128 * 2 = 6,144 bytes
/// - shared_scores: 24 * 24 * 4 = 2,304 bytes
/// - shared_output: 24 * 128 * 4 = 12,288 bytes
/// - Total: 26,880 bytes (fits in 32KB limit)
///
/// Threadgroup: 32 threads (one SIMD group)
/// Grid: [num_q_tiles, num_heads, batch]
kernel void flash_attention_simd_kernel_fp16_tile24(
    device const half* Q                [[buffer(0)]],
    device const half* K                [[buffer(1)]],
    device const half* V                [[buffer(2)]],
    device const half* mask             [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& batch                [[buffer(5)]],
    constant uint& num_heads            [[buffer(6)]],
    constant uint& seq_len_q            [[buffer(7)]],
    constant uint& seq_len_k            [[buffer(8)]],
    constant uint& head_dim             [[buffer(9)]],
    constant float& scale               [[buffer(10)]],
    constant bool& has_mask             [[buffer(11)]],
    threadgroup half* shared_Q          [[threadgroup(0)]],
    threadgroup half* shared_KV         [[threadgroup(1)]],  // K then V (aliased)
    threadgroup float* shared_scores    [[threadgroup(2)]],  // FP32 for numerical stability
    threadgroup float* shared_output    [[threadgroup(3)]],  // FP32 accumulation
    uint3 gid                           [[threadgroup_position_in_grid]],
    uint  simd_lane_id                  [[thread_index_in_simdgroup]],
    uint  simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TILE_SIZE_24 = 24;

    const uint b = gid.z;
    const uint h = gid.y;
    const uint q_tile_idx = gid.x;

    if (b >= batch || h >= num_heads) return;

    const uint q_start = q_tile_idx * TILE_SIZE_24;
    const uint qkv_offset = (b * num_heads + h) * seq_len_q * head_dim;
    const uint k_offset = (b * num_heads + h) * seq_len_k * head_dim;

    // Load Q tile
    for (uint idx = simd_lane_id; idx < TILE_SIZE_24 * head_dim; idx += 32) {
        uint row = idx / head_dim;
        uint col = idx % head_dim;
        uint q_idx = q_start + row;
        if (q_idx < seq_len_q) {
            shared_Q[row * head_dim + col] = Q[qkv_offset + q_idx * head_dim + col];
        } else {
            shared_Q[row * head_dim + col] = half(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Initialize FP32 output accumulator
    for (uint idx = simd_lane_id; idx < TILE_SIZE_24 * head_dim; idx += 32) {
        shared_output[idx] = 0.0f;
    }

    // Online softmax state
    threadgroup OnlineSoftmaxState softmax_states[TILE_SIZE_24];
    if (simd_lane_id < TILE_SIZE_24) {
        softmax_states[simd_lane_id].max_val = -INFINITY;
        softmax_states[simd_lane_id].sum_exp = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_k_tiles = (seq_len_k + TILE_SIZE_24 - 1) / TILE_SIZE_24;

    for (uint k_tile_idx = 0; k_tile_idx < num_k_tiles; k_tile_idx++) {
        const uint k_start = k_tile_idx * TILE_SIZE_24;
        const uint k_end = min(k_start + TILE_SIZE_24, seq_len_k);
        const uint k_tile_size = k_end - k_start;

        // Load K tile
        for (uint idx = simd_lane_id; idx < TILE_SIZE_24 * head_dim; idx += 32) {
            uint row = idx / head_dim;
            uint col = idx % head_dim;
            uint k_idx = k_start + row;
            if (k_idx < seq_len_k && row < k_tile_size) {
                shared_KV[row * head_dim + col] = K[k_offset + k_idx * head_dim + col];
            } else {
                shared_KV[row * head_dim + col] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute Q @ K^T with 3x3 decomposition of 8x8 blocks
        simdgroup_half8x8 s_00, s_01, s_02;
        simdgroup_half8x8 s_10, s_11, s_12;
        simdgroup_half8x8 s_20, s_21, s_22;

        s_00 = simdgroup_half8x8(half(0.0f));
        s_01 = simdgroup_half8x8(half(0.0f));
        s_02 = simdgroup_half8x8(half(0.0f));
        s_10 = simdgroup_half8x8(half(0.0f));
        s_11 = simdgroup_half8x8(half(0.0f));
        s_12 = simdgroup_half8x8(half(0.0f));
        s_20 = simdgroup_half8x8(half(0.0f));
        s_21 = simdgroup_half8x8(half(0.0f));
        s_22 = simdgroup_half8x8(half(0.0f));

        for (uint d = 0; d < head_dim; d += 8) {
            simdgroup_half8x8 q0, q1, q2, k0, k1, k2;
            simdgroup_load(q0, shared_Q + 0 * head_dim + d, head_dim);
            simdgroup_load(q1, shared_Q + 8 * head_dim + d, head_dim);
            simdgroup_load(q2, shared_Q + 16 * head_dim + d, head_dim);
            simdgroup_load(k0, shared_KV + 0 * head_dim + d, head_dim);
            simdgroup_load(k1, shared_KV + 8 * head_dim + d, head_dim);
            simdgroup_load(k2, shared_KV + 16 * head_dim + d, head_dim);

            simdgroup_multiply_accumulate(s_00, q0, k0, s_00);
            simdgroup_multiply_accumulate(s_01, q0, k1, s_01);
            simdgroup_multiply_accumulate(s_02, q0, k2, s_02);
            simdgroup_multiply_accumulate(s_10, q1, k0, s_10);
            simdgroup_multiply_accumulate(s_11, q1, k1, s_11);
            simdgroup_multiply_accumulate(s_12, q1, k2, s_12);
            simdgroup_multiply_accumulate(s_20, q2, k0, s_20);
            simdgroup_multiply_accumulate(s_21, q2, k1, s_21);
            simdgroup_multiply_accumulate(s_22, q2, k2, s_22);
        }

        // Convert scores to FP32 and store
        threadgroup half temp_scores_half[TILE_SIZE_24 * TILE_SIZE_24];
        simdgroup_store(s_00, temp_scores_half + 0 * TILE_SIZE_24 + 0, TILE_SIZE_24);
        simdgroup_store(s_01, temp_scores_half + 0 * TILE_SIZE_24 + 8, TILE_SIZE_24);
        simdgroup_store(s_02, temp_scores_half + 0 * TILE_SIZE_24 + 16, TILE_SIZE_24);
        simdgroup_store(s_10, temp_scores_half + 8 * TILE_SIZE_24 + 0, TILE_SIZE_24);
        simdgroup_store(s_11, temp_scores_half + 8 * TILE_SIZE_24 + 8, TILE_SIZE_24);
        simdgroup_store(s_12, temp_scores_half + 8 * TILE_SIZE_24 + 16, TILE_SIZE_24);
        simdgroup_store(s_20, temp_scores_half + 16 * TILE_SIZE_24 + 0, TILE_SIZE_24);
        simdgroup_store(s_21, temp_scores_half + 16 * TILE_SIZE_24 + 8, TILE_SIZE_24);
        simdgroup_store(s_22, temp_scores_half + 16 * TILE_SIZE_24 + 16, TILE_SIZE_24);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Convert to FP32 and apply scaling/masking
        for (uint idx = simd_lane_id; idx < TILE_SIZE_24 * TILE_SIZE_24; idx += 32) {
            uint row = idx / TILE_SIZE_24;
            uint col = idx % TILE_SIZE_24;
            uint q_idx = q_start + row;
            uint k_idx = k_start + col;

            if (q_idx < seq_len_q && col < k_tile_size) {
                float score = float(temp_scores_half[row * TILE_SIZE_24 + col]) * scale;
                if (has_mask) {
                    score += float(mask[b * seq_len_q * seq_len_k + q_idx * seq_len_k + k_idx]);
                }
                shared_scores[row * TILE_SIZE_24 + col] = score;
            } else {
                shared_scores[row * TILE_SIZE_24 + col] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update online softmax
        if (simd_lane_id < TILE_SIZE_24) {
            uint q_idx = q_start + simd_lane_id;
            if (q_idx < seq_len_q) {
                float old_max = softmax_states[simd_lane_id].max_val;

                for (uint k_local = 0; k_local < k_tile_size; k_local++) {
                    float score = shared_scores[simd_lane_id * TILE_SIZE_24 + k_local];
                    softmax_states[simd_lane_id] = update_online_softmax(softmax_states[simd_lane_id], score);
                }

                if (softmax_states[simd_lane_id].max_val > old_max) {
                    float rescale = exp(old_max - softmax_states[simd_lane_id].max_val);
                    for (uint d = 0; d < head_dim; d++) {
                        shared_output[simd_lane_id * head_dim + d] *= rescale;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute softmax weights
        for (uint idx = simd_lane_id; idx < TILE_SIZE_24 * TILE_SIZE_24; idx += 32) {
            uint row = idx / TILE_SIZE_24;
            uint col = idx % TILE_SIZE_24;
            uint q_idx = q_start + row;

            if (q_idx < seq_len_q && col < k_tile_size) {
                float score = shared_scores[row * TILE_SIZE_24 + col];
                shared_scores[row * TILE_SIZE_24 + col] = exp(score - softmax_states[row].max_val);
            } else {
                shared_scores[row * TILE_SIZE_24 + col] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load V tile
        for (uint idx = simd_lane_id; idx < TILE_SIZE_24 * head_dim; idx += 32) {
            uint row = idx / head_dim;
            uint col = idx % head_dim;
            uint v_idx = k_start + row;
            if (v_idx < seq_len_k && row < k_tile_size) {
                shared_KV[row * head_dim + col] = V[k_offset + v_idx * head_dim + col];
            } else {
                shared_KV[row * head_dim + col] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate weighted V with 3x3 weight blocks and 3x2 V blocks per d_block
        for (uint d_block = 0; d_block < head_dim; d_block += 16) {
            simdgroup_float8x8 out_00, out_01, out_10, out_11, out_20, out_21;

            simdgroup_load(out_00, shared_output + 0 * head_dim + d_block + 0, head_dim);
            simdgroup_load(out_01, shared_output + 0 * head_dim + d_block + 8, head_dim);
            simdgroup_load(out_10, shared_output + 8 * head_dim + d_block + 0, head_dim);
            simdgroup_load(out_11, shared_output + 8 * head_dim + d_block + 8, head_dim);
            simdgroup_load(out_20, shared_output + 16 * head_dim + d_block + 0, head_dim);
            simdgroup_load(out_21, shared_output + 16 * head_dim + d_block + 8, head_dim);

            // Load weights as FP32 then convert for multiply
            threadgroup half temp_weights_half[TILE_SIZE_24 * TILE_SIZE_24];
            for (uint i = simd_lane_id; i < TILE_SIZE_24 * TILE_SIZE_24; i += 32) {
                temp_weights_half[i] = half(shared_scores[i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_half8x8 w00, w01, w02, w10, w11, w12, w20, w21, w22;
            simdgroup_load(w00, temp_weights_half + 0 * TILE_SIZE_24 + 0, TILE_SIZE_24);
            simdgroup_load(w01, temp_weights_half + 0 * TILE_SIZE_24 + 8, TILE_SIZE_24);
            simdgroup_load(w02, temp_weights_half + 0 * TILE_SIZE_24 + 16, TILE_SIZE_24);
            simdgroup_load(w10, temp_weights_half + 8 * TILE_SIZE_24 + 0, TILE_SIZE_24);
            simdgroup_load(w11, temp_weights_half + 8 * TILE_SIZE_24 + 8, TILE_SIZE_24);
            simdgroup_load(w12, temp_weights_half + 8 * TILE_SIZE_24 + 16, TILE_SIZE_24);
            simdgroup_load(w20, temp_weights_half + 16 * TILE_SIZE_24 + 0, TILE_SIZE_24);
            simdgroup_load(w21, temp_weights_half + 16 * TILE_SIZE_24 + 8, TILE_SIZE_24);
            simdgroup_load(w22, temp_weights_half + 16 * TILE_SIZE_24 + 16, TILE_SIZE_24);

            simdgroup_half8x8 v0_0, v0_1, v1_0, v1_1, v2_0, v2_1;
            simdgroup_load(v0_0, shared_KV + 0 * head_dim + d_block + 0, head_dim);
            simdgroup_load(v0_1, shared_KV + 0 * head_dim + d_block + 8, head_dim);
            simdgroup_load(v1_0, shared_KV + 8 * head_dim + d_block + 0, head_dim);
            simdgroup_load(v1_1, shared_KV + 8 * head_dim + d_block + 8, head_dim);
            simdgroup_load(v2_0, shared_KV + 16 * head_dim + d_block + 0, head_dim);
            simdgroup_load(v2_1, shared_KV + 16 * head_dim + d_block + 8, head_dim);

            // Mixed precision: FP16 multiply with FP32 accumulate
            simdgroup_multiply_accumulate(out_00, w00, v0_0, out_00);
            simdgroup_multiply_accumulate(out_00, w01, v1_0, out_00);
            simdgroup_multiply_accumulate(out_00, w02, v2_0, out_00);
            simdgroup_multiply_accumulate(out_01, w00, v0_1, out_01);
            simdgroup_multiply_accumulate(out_01, w01, v1_1, out_01);
            simdgroup_multiply_accumulate(out_01, w02, v2_1, out_01);

            simdgroup_multiply_accumulate(out_10, w10, v0_0, out_10);
            simdgroup_multiply_accumulate(out_10, w11, v1_0, out_10);
            simdgroup_multiply_accumulate(out_10, w12, v2_0, out_10);
            simdgroup_multiply_accumulate(out_11, w10, v0_1, out_11);
            simdgroup_multiply_accumulate(out_11, w11, v1_1, out_11);
            simdgroup_multiply_accumulate(out_11, w12, v2_1, out_11);

            simdgroup_multiply_accumulate(out_20, w20, v0_0, out_20);
            simdgroup_multiply_accumulate(out_20, w21, v1_0, out_20);
            simdgroup_multiply_accumulate(out_20, w22, v2_0, out_20);
            simdgroup_multiply_accumulate(out_21, w20, v0_1, out_21);
            simdgroup_multiply_accumulate(out_21, w21, v1_1, out_21);
            simdgroup_multiply_accumulate(out_21, w22, v2_1, out_21);

            simdgroup_store(out_00, shared_output + 0 * head_dim + d_block + 0, head_dim);
            simdgroup_store(out_01, shared_output + 0 * head_dim + d_block + 8, head_dim);
            simdgroup_store(out_10, shared_output + 8 * head_dim + d_block + 0, head_dim);
            simdgroup_store(out_11, shared_output + 8 * head_dim + d_block + 8, head_dim);
            simdgroup_store(out_20, shared_output + 16 * head_dim + d_block + 0, head_dim);
            simdgroup_store(out_21, shared_output + 16 * head_dim + d_block + 8, head_dim);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Final normalization and write output as FP16
    for (uint idx = simd_lane_id; idx < TILE_SIZE_24 * head_dim; idx += 32) {
        uint row = idx / head_dim;
        uint col = idx % head_dim;
        uint q_idx = q_start + row;

        if (q_idx < seq_len_q) {
            float normalized = shared_output[row * head_dim + col] / softmax_states[row].sum_exp;
            output[qkv_offset + q_idx * head_dim + col] = half(normalized);
        }
    }
}
