/// @file attention_kernel.metal
/// @brief Metal kernel implementations for FlashAttention with SIMD matrix operations.

#include <metal_stdlib>
using namespace metal;

// Tile size constant
constant constexpr uint TILE_SIZE = 16;

// Threadgroup memory budget: 32KB = 32768 bytes
// Apple Silicon has a 32KB limit per threadgroup for threadgroup memory.
// Memory calculations by tile size and precision:
//
// tile16 + head_dim=128 + FP32:
//   shared_Q:      16 * 128 * 4 =  8,192 bytes
//   shared_KV:     16 * 128 * 4 =  8,192 bytes (aliased for K then V)
//   shared_scores: 16 * 16  * 4 =  1,024 bytes
//   shared_output: 16 * 128 * 4 =  8,192 bytes
//   TOTAL:                        25,600 bytes (OK, fits in 32KB)
//
// tile16 + head_dim=128 + FP16:
//   shared_Q:      16 * 128 * 2 =  4,096 bytes
//   shared_KV:     16 * 128 * 2 =  4,096 bytes
//   shared_scores: 16 * 16  * 4 =  1,024 bytes (FP32 for numerical stability)
//   shared_output: 16 * 128 * 4 =  8,192 bytes (FP32 accumulation)
//   TOTAL:                        17,408 bytes (OK, fits in 32KB)
//
// tile24 + head_dim=128 + FP16:
//   shared_Q:      24 * 128 * 2 =  6,144 bytes
//   shared_KV:     24 * 128 * 2 =  6,144 bytes
//   shared_scores: 24 * 24  * 4 =  2,304 bytes
//   shared_output: 24 * 128 * 4 = 12,288 bytes
//   temp buffers:  24 * 24  * 2 =  1,152 bytes (temp_scores_half)
//                  24 * 24  * 2 =  1,152 bytes (temp_weights_half)
//   TOTAL:                        29,184 bytes (OK, tight but fits)
//
// tile32 + head_dim=128 + FP16: EXCEEDS 32KB - DO NOT USE
//   shared_Q:      32 * 128 * 2 =  8,192 bytes
//   shared_KV:     32 * 128 * 2 =  8,192 bytes
//   shared_scores: 32 * 32  * 4 =  4,096 bytes
//   shared_output: 32 * 128 * 4 = 16,384 bytes
//   temp buffers:  32 * 32  * 2 =  2,048 bytes (temp_scores_half)
//                  32 * 32  * 2 =  2,048 bytes (temp_weights_half)
//   TOTAL:                        40,960 bytes (EXCEEDS 32KB limit!)
//
// tile32 + head_dim=64 + FP16:
//   shared_Q:      32 * 64  * 2 =  4,096 bytes
//   shared_KV:     32 * 64  * 2 =  4,096 bytes
//   shared_scores: 32 * 32  * 4 =  4,096 bytes
//   shared_output: 32 * 64  * 4 =  8,192 bytes
//   temp buffers:  32 * 32  * 2 =  2,048 bytes (temp_scores_half)
//                  32 * 32  * 2 =  2,048 bytes (temp_weights_half)
//   TOTAL:                        24,576 bytes (OK, fits in 32KB)
//
// CONSTRAINT: tile32 variant (if added) must only be used when head_dim <= 64

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
        threadgroup float rescale_factors[TILE_SIZE];
        threadgroup bool needs_rescale[TILE_SIZE];

        if (simd_lane_id < TILE_SIZE) {
            uint q_idx = q_start + simd_lane_id;
            if (q_idx < seq_len_q) {
                float old_max = softmax_states[simd_lane_id].max_val;

                // Update state with all scores in this row
                for (uint k_local = 0; k_local < k_tile_size; k_local++) {
                    float score = shared_scores[simd_lane_id * TILE_SIZE + k_local];
                    softmax_states[simd_lane_id] = update_online_softmax(softmax_states[simd_lane_id], score);
                }

                // Compute rescale factor if max changed
                needs_rescale[simd_lane_id] = (softmax_states[simd_lane_id].max_val > old_max);
                if (needs_rescale[simd_lane_id]) {
                    rescale_factors[simd_lane_id] = exp(old_max - softmax_states[simd_lane_id].max_val);
                }
            } else {
                needs_rescale[simd_lane_id] = false;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale using ALL 32 threads
        for (uint row = 0; row < TILE_SIZE; row++) {
            if (needs_rescale[row]) {
                for (uint d = simd_lane_id; d < head_dim; d += 32) {
                    shared_output[row * head_dim + d] *= rescale_factors[row];
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
        // Hoist weight loads outside loop - weights don't change per d_block
        {
            simdgroup_float8x8 w00, w01, w10, w11;
            simdgroup_load(w00, shared_scores + 0 * TILE_SIZE + 0, TILE_SIZE);
            simdgroup_load(w01, shared_scores + 0 * TILE_SIZE + 8, TILE_SIZE);
            simdgroup_load(w10, shared_scores + 8 * TILE_SIZE + 0, TILE_SIZE);
            simdgroup_load(w11, shared_scores + 8 * TILE_SIZE + 8, TILE_SIZE);

            for (uint d_block = 0; d_block < head_dim; d_block += 16) {
                simdgroup_float8x8 out_00, out_01, out_10, out_11;
                simdgroup_float8x8 v0_0, v0_1, v1_0, v1_1;

                // Load output accumulators
                simdgroup_load(out_00, shared_output + 0 * head_dim + d_block + 0, head_dim);
                simdgroup_load(out_01, shared_output + 0 * head_dim + d_block + 8, head_dim);
                simdgroup_load(out_10, shared_output + 8 * head_dim + d_block + 0, head_dim);
                simdgroup_load(out_11, shared_output + 8 * head_dim + d_block + 8, head_dim);

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
            }
            // No barrier needed - single SIMD group, d_blocks write to non-overlapping columns
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
        threadgroup float rescale_factors[TILE_SIZE];
        threadgroup bool needs_rescale[TILE_SIZE];

        if (simd_lane_id < TILE_SIZE) {
            uint q_idx = q_start + simd_lane_id;
            if (q_idx < seq_len_q) {
                float old_max = softmax_states[simd_lane_id].max_val;

                for (uint k_local = 0; k_local < k_tile_size; k_local++) {
                    float score = shared_scores[simd_lane_id * TILE_SIZE + k_local];
                    softmax_states[simd_lane_id] = update_online_softmax(softmax_states[simd_lane_id], score);
                }

                needs_rescale[simd_lane_id] = (softmax_states[simd_lane_id].max_val > old_max);
                if (needs_rescale[simd_lane_id]) {
                    rescale_factors[simd_lane_id] = exp(old_max - softmax_states[simd_lane_id].max_val);
                }
            } else {
                needs_rescale[simd_lane_id] = false;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale using ALL 32 threads
        for (uint row = 0; row < TILE_SIZE; row++) {
            if (needs_rescale[row]) {
                for (uint d = simd_lane_id; d < head_dim; d += 32) {
                    shared_output[row * head_dim + d] *= rescale_factors[row];
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
        // Convert weights to FP16 once, then iterate d_blocks
        {
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

            for (uint d_block = 0; d_block < head_dim; d_block += 16) {
                simdgroup_float8x8 out_00, out_01, out_10, out_11;

                simdgroup_load(out_00, shared_output + 0 * head_dim + d_block + 0, head_dim);
                simdgroup_load(out_01, shared_output + 0 * head_dim + d_block + 8, head_dim);
                simdgroup_load(out_10, shared_output + 8 * head_dim + d_block + 0, head_dim);
                simdgroup_load(out_11, shared_output + 8 * head_dim + d_block + 8, head_dim);

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
            }
            // No barrier needed - single SIMD group, non-overlapping d_block columns
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
        threadgroup float rescale_factors[TILE_SIZE_24];
        threadgroup bool needs_rescale[TILE_SIZE_24];

        if (simd_lane_id < TILE_SIZE_24) {
            uint q_idx = q_start + simd_lane_id;
            if (q_idx < seq_len_q) {
                float old_max = softmax_states[simd_lane_id].max_val;

                for (uint k_local = 0; k_local < k_tile_size; k_local++) {
                    float score = shared_scores[simd_lane_id * TILE_SIZE_24 + k_local];
                    softmax_states[simd_lane_id] = update_online_softmax(softmax_states[simd_lane_id], score);
                }

                needs_rescale[simd_lane_id] = (softmax_states[simd_lane_id].max_val > old_max);
                if (needs_rescale[simd_lane_id]) {
                    rescale_factors[simd_lane_id] = exp(old_max - softmax_states[simd_lane_id].max_val);
                }
            } else {
                needs_rescale[simd_lane_id] = false;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale using ALL 32 threads
        for (uint row = 0; row < TILE_SIZE_24; row++) {
            if (needs_rescale[row]) {
                for (uint d = simd_lane_id; d < head_dim; d += 32) {
                    shared_output[row * head_dim + d] *= rescale_factors[row];
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
        // Hoist weight conversion outside d_block loop -- weights are the same for all d_blocks
        {
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

        for (uint d_block = 0; d_block < head_dim; d_block += 16) {
            simdgroup_float8x8 out_00, out_01, out_10, out_11, out_20, out_21;

            simdgroup_load(out_00, shared_output + 0 * head_dim + d_block + 0, head_dim);
            simdgroup_load(out_01, shared_output + 0 * head_dim + d_block + 8, head_dim);
            simdgroup_load(out_10, shared_output + 8 * head_dim + d_block + 0, head_dim);
            simdgroup_load(out_11, shared_output + 8 * head_dim + d_block + 8, head_dim);
            simdgroup_load(out_20, shared_output + 16 * head_dim + d_block + 0, head_dim);
            simdgroup_load(out_21, shared_output + 16 * head_dim + d_block + 8, head_dim);

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
        } // end hoisted weight block
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

/// FlashAttention kernel with Grouped Query Attention (GQA) support
/// Supports MHA (group_ratio=1), GQA (group_ratio=4,8), and MQA (group_ratio=num_heads)
///
/// GQA: Multiple Q heads share the same KV heads
/// Example: Llama 3 has num_heads=32, num_kv_heads=8, group_ratio=4
/// Four Q heads share one KV head, reducing KV cache size by 4x
///
/// K/V Bandwidth Optimization:
/// The loop order is [K/V tiles (outer)] x [Q heads (inner)].
/// K and V tiles are loaded ONCE per tile and reused across all Q heads
/// in the group, reducing K/V global memory bandwidth by group_ratio.
/// Per-head output accumulators live in thread-private memory to avoid
/// global memory read-modify-write overhead.
///
/// Memory layout (threadgroup):
/// - shared_Q: [TILE_SIZE, head_dim] FP16 (loaded per Q head)       = 4,096 bytes
/// - shared_K: [TILE_SIZE, head_dim] FP16 (loaded once per K/V tile) = 4,096 bytes
/// - shared_scores: [TILE_SIZE, TILE_SIZE] FP32                      = 1,024 bytes
/// - shared_output: [TILE_SIZE, head_dim] FP32 (SIMD matmul scratch) = 8,192 bytes
/// - shared_V: [TILE_SIZE, head_dim] FP16 (loaded once per K/V tile) = 4,096 bytes
/// Total: 21,504 bytes (fits in 32KB limit)
///
/// Per-thread private memory (for head_dim=128, group_ratio=4):
/// - output_acc: group_ratio * 64 floats = 1,024 bytes
/// - softmax state: group_ratio * 2 floats = 32 bytes (per-row, only lane < 16)
/// Total per thread: ~1,056 bytes (fits in register file)
///
/// Threadgroup: 32 threads (one SIMD group)
/// Grid: [num_q_tiles, num_kv_heads, batch]

// Maximum supported group ratio for private array sizing
constant constexpr uint MAX_GROUP_RATIO = 8;
// Max elements per thread: TILE_SIZE * max_head_dim / 32 (SIMD width)
// For head_dim=128: 16 * 128 / 32 = 64
constant constexpr uint MAX_ELEMS_PER_THREAD = (TILE_SIZE * 128) / 32;

kernel void flash_attention_gqa_fp16(
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
    constant uint& num_kv_heads         [[buffer(12)]],
    constant uint& group_ratio          [[buffer(13)]],
    threadgroup half* shared_Q          [[threadgroup(0)]],
    threadgroup half* shared_K          [[threadgroup(1)]],  // K tile (persistent across Q heads)
    threadgroup float* shared_scores    [[threadgroup(2)]],  // FP32 for numerical stability
    threadgroup float* shared_output    [[threadgroup(3)]],  // FP32 scratch for SIMD matmul
    threadgroup half* shared_V          [[threadgroup(4)]],  // V tile (separate from K)
    uint3 gid                           [[threadgroup_position_in_grid]],
    uint  simd_lane_id                  [[thread_index_in_simdgroup]],
    uint  simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    const uint b = gid.z;
    const uint kv_head = gid.y;
    const uint q_tile_idx = gid.x;

    if (b >= batch || kv_head >= num_kv_heads) return;

    const uint q_start = q_tile_idx * TILE_SIZE;
    const uint kv_offset = (b * num_kv_heads + kv_head) * seq_len_k * head_dim;
    const uint num_k_tiles = (seq_len_k + TILE_SIZE - 1) / TILE_SIZE;
    const uint elems_per_thread = (TILE_SIZE * head_dim) / 32;

    // Per-head output accumulators in thread-private memory.
    // Thread t owns elements at flat indices t, t+32, t+64, ... in the
    // [TILE_SIZE, head_dim] output tile. elem e maps to flat index e*32+t.
    float output_acc[MAX_GROUP_RATIO][MAX_ELEMS_PER_THREAD];

    // Per-head online softmax state in thread-private memory.
    // Only threads with simd_lane_id < TILE_SIZE use these (one row each).
    float head_max_val[MAX_GROUP_RATIO];
    float head_sum_exp[MAX_GROUP_RATIO];

    // Initialize per-head state
    for (uint g = 0; g < group_ratio && g < MAX_GROUP_RATIO; g++) {
        for (uint e = 0; e < elems_per_thread; e++) {
            output_acc[g][e] = 0.0f;
        }
        head_max_val[g] = -INFINITY;
        head_sum_exp[g] = 0.0f;
    }

    // === Outer loop: K/V tiles ===
    // K and V are loaded ONCE per tile and reused across all Q heads in the group.
    for (uint k_tile_idx = 0; k_tile_idx < num_k_tiles; k_tile_idx++) {
        const uint k_start = k_tile_idx * TILE_SIZE;
        const uint k_end = min(k_start + TILE_SIZE, seq_len_k);
        const uint k_tile_size = k_end - k_start;

        // Load K tile ONCE into shared_K
        for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32) {
            uint row = idx / head_dim;
            uint col = idx % head_dim;
            uint k_idx = k_start + row;
            if (k_idx < seq_len_k && row < k_tile_size) {
                shared_K[row * head_dim + col] = K[kv_offset + k_idx * head_dim + col];
            } else {
                shared_K[row * head_dim + col] = half(0.0f);
            }
        }

        // Load V tile ONCE into shared_V (separate buffer from K)
        for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32) {
            uint row = idx / head_dim;
            uint col = idx % head_dim;
            uint v_idx = k_start + row;
            if (v_idx < seq_len_k && row < k_tile_size) {
                shared_V[row * head_dim + col] = V[kv_offset + v_idx * head_dim + col];
            } else {
                shared_V[row * head_dim + col] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Inner loop: Q heads ===
        // Process each Q head against the loaded K/V tile.
        for (uint g = 0; g < group_ratio && g < MAX_GROUP_RATIO; g++) {
            const uint q_head = kv_head * group_ratio + g;
            if (q_head >= num_heads) continue;

            const uint q_offset = (b * num_heads + q_head) * seq_len_q * head_dim;

            // Load Q tile for this head
            for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32) {
                uint row = idx / head_dim;
                uint col = idx % head_dim;
                uint q_idx = q_start + row;
                if (q_idx < seq_len_q) {
                    shared_Q[row * head_dim + col] = Q[q_offset + q_idx * head_dim + col];
                } else {
                    shared_Q[row * head_dim + col] = half(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- Compute Q @ K^T using FP16 SIMD matrix ops ---
            simdgroup_half8x8 scores_00, scores_01, scores_10, scores_11;
            scores_00 = simdgroup_half8x8(half(0.0f));
            scores_01 = simdgroup_half8x8(half(0.0f));
            scores_10 = simdgroup_half8x8(half(0.0f));
            scores_11 = simdgroup_half8x8(half(0.0f));

            for (uint d = 0; d < head_dim; d += 8) {
                simdgroup_half8x8 q0, q1, k0, k1;
                simdgroup_load(q0, shared_Q + 0 * head_dim + d, head_dim);
                simdgroup_load(q1, shared_Q + 8 * head_dim + d, head_dim);
                simdgroup_load(k0, shared_K + 0 * head_dim + d, head_dim);
                simdgroup_load(k1, shared_K + 8 * head_dim + d, head_dim);

                simdgroup_multiply_accumulate(scores_00, q0, k0, scores_00);
                simdgroup_multiply_accumulate(scores_01, q0, k1, scores_01);
                simdgroup_multiply_accumulate(scores_10, q1, k0, scores_10);
                simdgroup_multiply_accumulate(scores_11, q1, k1, scores_11);
            }

            // Store FP16 scores to temp buffer, then convert to FP32
            threadgroup half temp_scores_half[TILE_SIZE * TILE_SIZE];
            simdgroup_store(scores_00, temp_scores_half + 0 * TILE_SIZE + 0, TILE_SIZE);
            simdgroup_store(scores_01, temp_scores_half + 0 * TILE_SIZE + 8, TILE_SIZE);
            simdgroup_store(scores_10, temp_scores_half + 8 * TILE_SIZE + 0, TILE_SIZE);
            simdgroup_store(scores_11, temp_scores_half + 8 * TILE_SIZE + 8, TILE_SIZE);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Convert to FP32, apply scaling and mask
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

            // --- Online softmax update ---
            // Threads 0..15 each handle one row's softmax state (private per head).
            threadgroup float rescale_factors[TILE_SIZE];
            threadgroup bool needs_rescale[TILE_SIZE];

            if (simd_lane_id < TILE_SIZE) {
                uint q_idx = q_start + simd_lane_id;
                if (q_idx < seq_len_q) {
                    float old_max = head_max_val[g];
                    float new_max = old_max;
                    float new_sum = head_sum_exp[g];

                    for (uint k_local = 0; k_local < k_tile_size; k_local++) {
                        float score = shared_scores[simd_lane_id * TILE_SIZE + k_local];
                        float prev_max = new_max;
                        new_max = max(new_max, score);
                        float exp_diff = exp(prev_max - new_max);
                        new_sum = new_sum * exp_diff + exp(score - new_max);
                    }

                    head_max_val[g] = new_max;
                    head_sum_exp[g] = new_sum;

                    needs_rescale[simd_lane_id] = (new_max > old_max);
                    if (needs_rescale[simd_lane_id]) {
                        rescale_factors[simd_lane_id] = exp(old_max - new_max);
                    }
                } else {
                    needs_rescale[simd_lane_id] = false;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Rescale per-head output accumulators in private memory
            {
                uint elem = 0;
                for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32, elem++) {
                    uint row = idx / head_dim;
                    if (needs_rescale[row]) {
                        output_acc[g][elem] *= rescale_factors[row];
                    }
                }
            }

            // --- Compute softmax weights ---
            // Broadcast per-row max_val to shared memory so all 32 threads can read it.
            threadgroup float row_max_vals[TILE_SIZE];
            if (simd_lane_id < TILE_SIZE) {
                row_max_vals[simd_lane_id] = head_max_val[g];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint idx = simd_lane_id; idx < TILE_SIZE * TILE_SIZE; idx += 32) {
                uint row = idx / TILE_SIZE;
                uint col = idx % TILE_SIZE;
                uint q_idx = q_start + row;

                if (q_idx < seq_len_q && col < k_tile_size) {
                    float score = shared_scores[row * TILE_SIZE + col];
                    shared_scores[row * TILE_SIZE + col] = exp(score - row_max_vals[row]);
                } else {
                    shared_scores[row * TILE_SIZE + col] = 0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- Accumulate weighted V using SIMD matrix ops ---
            // V is in shared_V (persistent). Use shared_output as scratch buffer for
            // the SIMD matmul result, then read back into private output accumulators.

            // Convert FP32 weights to FP16 once (reused across d_blocks)
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

            for (uint d_block = 0; d_block < head_dim; d_block += 16) {
                simdgroup_float8x8 out_00, out_01, out_10, out_11;
                out_00 = simdgroup_float8x8(0.0f);
                out_01 = simdgroup_float8x8(0.0f);
                out_10 = simdgroup_float8x8(0.0f);
                out_11 = simdgroup_float8x8(0.0f);

                simdgroup_half8x8 v0_0, v0_1, v1_0, v1_1;
                simdgroup_load(v0_0, shared_V + 0 * head_dim + d_block + 0, head_dim);
                simdgroup_load(v0_1, shared_V + 0 * head_dim + d_block + 8, head_dim);
                simdgroup_load(v1_0, shared_V + 8 * head_dim + d_block + 0, head_dim);
                simdgroup_load(v1_1, shared_V + 8 * head_dim + d_block + 8, head_dim);

                // Mixed precision: FP16 multiply with FP32 accumulate
                simdgroup_multiply_accumulate(out_00, w00, v0_0, out_00);
                simdgroup_multiply_accumulate(out_00, w01, v1_0, out_00);
                simdgroup_multiply_accumulate(out_01, w00, v0_1, out_01);
                simdgroup_multiply_accumulate(out_01, w01, v1_1, out_01);
                simdgroup_multiply_accumulate(out_10, w10, v0_0, out_10);
                simdgroup_multiply_accumulate(out_10, w11, v1_0, out_10);
                simdgroup_multiply_accumulate(out_11, w10, v0_1, out_11);
                simdgroup_multiply_accumulate(out_11, w11, v1_1, out_11);

                // Store SIMD result to shared_output scratch
                simdgroup_store(out_00, shared_output + 0 * head_dim + d_block + 0, head_dim);
                simdgroup_store(out_01, shared_output + 0 * head_dim + d_block + 8, head_dim);
                simdgroup_store(out_10, shared_output + 8 * head_dim + d_block + 0, head_dim);
                simdgroup_store(out_11, shared_output + 8 * head_dim + d_block + 8, head_dim);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Accumulate from shared_output scratch into private output_acc.
                // Use the natural strided pattern: thread t reads elements at
                // flat indices t, t+32, t+64, ... that fall within this d_block's columns.
                for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32) {
                    uint col = idx % head_dim;
                    if (col >= d_block && col < d_block + 16) {
                        uint elem = idx / 32;
                        output_acc[g][elem] += shared_output[idx];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } // end inner loop (Q heads)
    } // end outer loop (K/V tiles)

    // === Final normalization and write output as FP16 ===
    for (uint g = 0; g < group_ratio && g < MAX_GROUP_RATIO; g++) {
        const uint q_head = kv_head * group_ratio + g;
        if (q_head >= num_heads) continue;
        const uint q_offset = (b * num_heads + q_head) * seq_len_q * head_dim;

        // Broadcast per-row sum_exp to shared memory for all threads to read
        threadgroup float row_sum_exps[TILE_SIZE];
        if (simd_lane_id < TILE_SIZE) {
            row_sum_exps[simd_lane_id] = head_sum_exp[g];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint elem = 0;
        for (uint idx = simd_lane_id; idx < TILE_SIZE * head_dim; idx += 32, elem++) {
            uint row = idx / head_dim;
            uint col = idx % head_dim;
            uint q_idx = q_start + row;

            if (q_idx < seq_len_q) {
                float normalized = output_acc[g][elem] / row_sum_exps[row];
                output[q_offset + q_idx * head_dim + col] = half(normalized);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/// FlashAttention kernel using 2 SIMD groups (64 threads) for FP16
/// Each SIMD group handles 8 rows of the 16x16 tile independently
///
/// Memory layout (same as single-SIMD tile16):
/// - shared_Q: [16, head_dim] FP16
/// - shared_KV: [16, head_dim] FP16 (K then V, aliased)
/// - shared_scores: [16, 16] FP32 for numerical stability
/// - shared_output: [16, head_dim] FP32 accumulation
/// Total: 17,408 bytes (fits in 32KB limit)
///
/// Threadgroup: 64 threads (two SIMD groups)
/// Grid: [num_q_tiles, num_heads, batch]
///
/// Parallelization:
/// - SIMD group 0: Q rows 0-7 (top half of score matrix)
/// - SIMD group 1: Q rows 8-15 (bottom half of score matrix)
/// - Both groups cooperatively load K/V tiles (2x faster)
/// - Each group independently computes its 8x16 score block
/// - No merge needed: each group handles its rows independently
kernel void flash_attention_simd_kernel_fp16_multi(
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

    // Cooperative loading of Q tile (64 threads)
    for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_SIZE * head_dim; idx += 64) {
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
    for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_SIZE * head_dim; idx += 64) {
        shared_output[idx] = 0.0f;
    }

    // Online softmax state for each SIMD group's rows
    threadgroup OnlineSoftmaxState softmax_states[TILE_SIZE];
    const uint row_start = simd_group_id * 8;  // Each group handles 8 rows

    if (simd_lane_id < 8) {
        uint row_idx = row_start + simd_lane_id;
        softmax_states[row_idx].max_val = -INFINITY;
        softmax_states[row_idx].sum_exp = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_k_tiles = (seq_len_k + TILE_SIZE - 1) / TILE_SIZE;

    for (uint k_tile_idx = 0; k_tile_idx < num_k_tiles; k_tile_idx++) {
        const uint k_start = k_tile_idx * TILE_SIZE;
        const uint k_end = min(k_start + TILE_SIZE, seq_len_k);
        const uint k_tile_size = k_end - k_start;

        // Cooperative load K tile (64 threads)
        for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_SIZE * head_dim; idx += 64) {
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

        // Compute Q @ K^T: each SIMD group handles one 8x16 block (1x2 in 8x8 tiles)
        // SIMD group 0: rows 0-7, SIMD group 1: rows 8-15
        simdgroup_half8x8 scores_0, scores_1;
        scores_0 = simdgroup_half8x8(half(0.0f));
        scores_1 = simdgroup_half8x8(half(0.0f));

        for (uint d = 0; d < head_dim; d += 8) {
            simdgroup_half8x8 q_block, k0, k1;
            simdgroup_load(q_block, shared_Q + row_start * head_dim + d, head_dim);
            simdgroup_load(k0, shared_KV + 0 * head_dim + d, head_dim);
            simdgroup_load(k1, shared_KV + 8 * head_dim + d, head_dim);

            simdgroup_multiply_accumulate(scores_0, q_block, k0, scores_0);
            simdgroup_multiply_accumulate(scores_1, q_block, k1, scores_1);
        }

        // Store scores to shared memory via temporary half buffer
        threadgroup half temp_scores_half[TILE_SIZE * TILE_SIZE];
        simdgroup_store(scores_0, temp_scores_half + row_start * TILE_SIZE + 0, TILE_SIZE);
        simdgroup_store(scores_1, temp_scores_half + row_start * TILE_SIZE + 8, TILE_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Convert to FP32 and apply scaling/masking (each group handles its rows)
        for (uint local_row = 0; local_row < 8; local_row++) {
            uint row = row_start + local_row;
            for (uint col = simd_lane_id; col < TILE_SIZE; col += 32) {
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
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update online softmax (each group handles its 8 rows)
        threadgroup float rescale_factors[TILE_SIZE];
        threadgroup bool needs_rescale[TILE_SIZE];

        if (simd_lane_id < 8) {
            uint row = row_start + simd_lane_id;
            uint q_idx = q_start + row;
            if (q_idx < seq_len_q) {
                float old_max = softmax_states[row].max_val;

                for (uint k_local = 0; k_local < k_tile_size; k_local++) {
                    float score = shared_scores[row * TILE_SIZE + k_local];
                    softmax_states[row] = update_online_softmax(softmax_states[row], score);
                }

                needs_rescale[row] = (softmax_states[row].max_val > old_max);
                if (needs_rescale[row]) {
                    rescale_factors[row] = exp(old_max - softmax_states[row].max_val);
                }
            } else {
                needs_rescale[row] = false;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale using ALL 64 threads
        for (uint row = 0; row < TILE_SIZE; row++) {
            if (needs_rescale[row]) {
                uint thread_id = simd_group_id * 32 + simd_lane_id;
                for (uint d = thread_id; d < head_dim; d += 64) {
                    shared_output[row * head_dim + d] *= rescale_factors[row];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute softmax weights (each group handles its rows)
        for (uint local_row = 0; local_row < 8; local_row++) {
            uint row = row_start + local_row;
            for (uint col = simd_lane_id; col < TILE_SIZE; col += 32) {
                uint q_idx = q_start + row;

                if (q_idx < seq_len_q && col < k_tile_size) {
                    float score = shared_scores[row * TILE_SIZE + col];
                    shared_scores[row * TILE_SIZE + col] = exp(score - softmax_states[row].max_val);
                } else {
                    shared_scores[row * TILE_SIZE + col] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative load V tile (64 threads)
        for (uint idx = simd_group_id * 32 + simd_lane_id; idx < TILE_SIZE * head_dim; idx += 64) {
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

        // Accumulate weighted V (each SIMD group handles its 8 rows)
        // Process head_dim in blocks of 16 (2x 8x8 blocks)
        for (uint d_block = 0; d_block < head_dim; d_block += 16) {
            simdgroup_float8x8 out_0, out_1;

            simdgroup_load(out_0, shared_output + row_start * head_dim + d_block + 0, head_dim);
            simdgroup_load(out_1, shared_output + row_start * head_dim + d_block + 8, head_dim);

            // Convert weights to FP16
            threadgroup half temp_weights_half[TILE_SIZE * TILE_SIZE];
            for (uint i = simd_group_id * 32 + simd_lane_id; i < TILE_SIZE * TILE_SIZE; i += 64) {
                temp_weights_half[i] = half(shared_scores[i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_half8x8 w0, w1;
            simdgroup_load(w0, temp_weights_half + row_start * TILE_SIZE + 0, TILE_SIZE);
            simdgroup_load(w1, temp_weights_half + row_start * TILE_SIZE + 8, TILE_SIZE);

            simdgroup_half8x8 v0_0, v0_1, v1_0, v1_1;
            simdgroup_load(v0_0, shared_KV + 0 * head_dim + d_block + 0, head_dim);
            simdgroup_load(v0_1, shared_KV + 0 * head_dim + d_block + 8, head_dim);
            simdgroup_load(v1_0, shared_KV + 8 * head_dim + d_block + 0, head_dim);
            simdgroup_load(v1_1, shared_KV + 8 * head_dim + d_block + 8, head_dim);

            // Mixed precision: FP16 multiply with FP32 accumulate
            simdgroup_multiply_accumulate(out_0, w0, v0_0, out_0);
            simdgroup_multiply_accumulate(out_0, w1, v1_0, out_0);
            simdgroup_multiply_accumulate(out_1, w0, v0_1, out_1);
            simdgroup_multiply_accumulate(out_1, w1, v1_1, out_1);

            simdgroup_store(out_0, shared_output + row_start * head_dim + d_block + 0, head_dim);
            simdgroup_store(out_1, shared_output + row_start * head_dim + d_block + 8, head_dim);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Final normalization and write output as FP16 (each group handles its rows)
    for (uint local_row = 0; local_row < 8; local_row++) {
        uint row = row_start + local_row;
        for (uint col = simd_lane_id; col < head_dim; col += 32) {
            uint q_idx = q_start + row;

            if (q_idx < seq_len_q) {
                float normalized = shared_output[row * head_dim + col] / softmax_states[row].sum_exp;
                output[qkv_offset + q_idx * head_dim + col] = half(normalized);
            }
        }
    }
}

/// FlashAttention kernel using 4 SIMD groups (128 threads) for FP16
/// Each SIMD group handles 4 rows of the 16x16 tile independently
///
/// Memory layout:
/// - shared_Q: [16, head_dim] FP16 = 16 * 128 * 2 = 4,096 bytes
/// - shared_KV: [16, head_dim] FP16 = 16 * 128 * 2 = 4,096 bytes (K then V, aliased)
/// - shared_scores: [16, 16] FP32 = 16 * 16 * 4 = 1,024 bytes
/// - shared_output: [16, head_dim] FP32 = 16 * 128 * 4 = 8,192 bytes
/// - softmax_states: 16 * 8 = 128 bytes
/// Total: ~17,536 bytes (fits in 32KB limit)
///
/// Threadgroup: 128 threads (four SIMD groups)
/// Grid: [num_q_tiles, num_heads, batch]
///
/// Parallelization Strategy (Row-Split):
/// - SIMD group 0: Q rows 0-3
/// - SIMD group 1: Q rows 4-7
/// - SIMD group 2: Q rows 8-11
/// - SIMD group 3: Q rows 12-15
/// - All 128 threads cooperatively load K/V tiles (2x faster than 2-SIMD)
/// - Each group independently computes attention for its 4 rows
/// - No inter-SIMD reduction needed (fully independent)
///
/// Expected speedup: 1.1-1.3x over 2-SIMD variant due to:
/// - 2x faster K/V loading (128 threads vs 64)
/// - Better occupancy with more threads per threadgroup
kernel void flash_attention_4simd_fp16(
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

    // Cooperative loading of Q tile (128 threads)
    const uint thread_id = simd_group_id * 32 + simd_lane_id;
    for (uint idx = thread_id; idx < TILE_SIZE * head_dim; idx += 128) {
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

    // Initialize FP32 output accumulator (128 threads)
    for (uint idx = thread_id; idx < TILE_SIZE * head_dim; idx += 128) {
        shared_output[idx] = 0.0f;
    }

    // Online softmax state for each SIMD group's rows
    threadgroup OnlineSoftmaxState softmax_states[TILE_SIZE];
    const uint row_start = simd_group_id * 4;  // Each group handles 4 rows

    if (simd_lane_id < 4) {
        uint row_idx = row_start + simd_lane_id;
        softmax_states[row_idx].max_val = -INFINITY;
        softmax_states[row_idx].sum_exp = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_k_tiles = (seq_len_k + TILE_SIZE - 1) / TILE_SIZE;

    for (uint k_tile_idx = 0; k_tile_idx < num_k_tiles; k_tile_idx++) {
        const uint k_start = k_tile_idx * TILE_SIZE;
        const uint k_end = min(k_start + TILE_SIZE, seq_len_k);
        const uint k_tile_size = k_end - k_start;

        // Cooperative load K tile (128 threads - 2x faster than 2-SIMD)
        for (uint idx = thread_id; idx < TILE_SIZE * head_dim; idx += 128) {
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

        // Compute Q @ K^T using SIMD matrix ops - each group computes one 8x8 quadrant
        // Group 0: scores[0:8, 0:8], Group 1: scores[0:8, 8:16]
        // Group 2: scores[8:16, 0:8], Group 3: scores[8:16, 8:16]
        {
            threadgroup half temp_scores_h[TILE_SIZE * TILE_SIZE];

            simdgroup_half8x8 score_block = simdgroup_half8x8(0.0h);
            uint q_row_offset = (simd_group_id < 2) ? 0 : 8;
            uint k_row_offset = (simd_group_id % 2 == 0) ? 0 : 8;

            for (uint d = 0; d < head_dim; d += 8) {
                simdgroup_half8x8 q_block, k_block;
                simdgroup_load(q_block, shared_Q + q_row_offset * head_dim + d, head_dim);
                simdgroup_load(k_block, shared_KV + k_row_offset * head_dim + d, head_dim);
                simdgroup_multiply_accumulate(score_block, q_block, k_block, score_block);
            }

            simdgroup_store(score_block, temp_scores_h + q_row_offset * TILE_SIZE + k_row_offset, TILE_SIZE);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Convert FP16 scores to FP32, apply scaling and mask (all 128 threads)
            for (uint idx = thread_id; idx < TILE_SIZE * TILE_SIZE; idx += 128) {
                uint row = idx / TILE_SIZE;
                uint col = idx % TILE_SIZE;
                uint q_idx = q_start + row;
                uint k_idx = k_start + col;

                if (q_idx < seq_len_q && col < k_tile_size) {
                    float score = float(temp_scores_h[row * TILE_SIZE + col]) * scale;
                    if (has_mask) {
                        score += float(mask[b * seq_len_q * seq_len_k + q_idx * seq_len_k + k_idx]);
                    }
                    shared_scores[row * TILE_SIZE + col] = score;
                } else {
                    shared_scores[row * TILE_SIZE + col] = -INFINITY;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax + weight computation using SIMD-parallel reduction (4-SIMD FP16)
        // Each SIMD group handles 4 rows; all 32 lanes do parallel max/sum
        threadgroup float rescale_factors[TILE_SIZE];
        threadgroup bool needs_rescale[TILE_SIZE];

        for (uint r = 0; r < 4; r++) {
            uint row = row_start + r;
            if (row < TILE_SIZE) {
                uint q_idx = q_start + row;
                if (q_idx < seq_len_q) {
                    float old_max = softmax_states[row].max_val;

                    float my_score = (simd_lane_id < k_tile_size) ?
                        shared_scores[row * TILE_SIZE + simd_lane_id] : -INFINITY;

                    float tile_max = simd_max(my_score);
                    float my_exp = (simd_lane_id < k_tile_size) ?
                        exp(my_score - tile_max) : 0.0f;
                    float tile_sum = simd_sum(my_exp);

                    if (simd_lane_id == 0) {
                        float new_max = max(old_max, tile_max);
                        float old_scale = exp(old_max - new_max);
                        float tile_scale = exp(tile_max - new_max);
                        float new_sum = softmax_states[row].sum_exp * old_scale +
                                        tile_sum * tile_scale;
                        softmax_states[row].max_val = new_max;
                        softmax_states[row].sum_exp = new_sum;
                        needs_rescale[row] = (new_max > old_max);
                        if (needs_rescale[row]) {
                            rescale_factors[row] = old_scale;
                        }
                    }

                    float new_max_bcast = simd_broadcast_first(
                        (simd_lane_id == 0) ? softmax_states[row].max_val : 0.0f);
                    if (simd_lane_id < TILE_SIZE) {
                        shared_scores[row * TILE_SIZE + simd_lane_id] =
                            (simd_lane_id < k_tile_size) ?
                                exp(my_score - new_max_bcast) : 0.0f;
                    }
                } else {
                    if (simd_lane_id == 0) {
                        needs_rescale[row] = false;
                    }
                    if (simd_lane_id < TILE_SIZE) {
                        shared_scores[row * TILE_SIZE + simd_lane_id] = 0.0f;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale using ALL 128 threads
        for (uint row = 0; row < TILE_SIZE; row++) {
            if (needs_rescale[row]) {
                for (uint d = thread_id; d < head_dim; d += 128) {
                    shared_output[row * head_dim + d] *= rescale_factors[row];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative load V tile (128 threads - 2x faster than 2-SIMD)
        for (uint idx = thread_id; idx < TILE_SIZE * head_dim; idx += 128) {
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

        // Accumulate weighted V using SIMD matrix ops
        // Convert softmax weights from FP32 to FP16 for simdgroup_half8x8 multiply
        {
            threadgroup half temp_weights_h[TILE_SIZE * TILE_SIZE];
            for (uint idx = thread_id; idx < TILE_SIZE * TILE_SIZE; idx += 128) {
                temp_weights_h[idx] = half(shared_scores[idx]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Load weight blocks as half8x8
            simdgroup_half8x8 w00, w01, w10, w11;
            simdgroup_load(w00, temp_weights_h + 0 * TILE_SIZE + 0, TILE_SIZE);
            simdgroup_load(w01, temp_weights_h + 0 * TILE_SIZE + 8, TILE_SIZE);
            simdgroup_load(w10, temp_weights_h + 8 * TILE_SIZE + 0, TILE_SIZE);
            simdgroup_load(w11, temp_weights_h + 8 * TILE_SIZE + 8, TILE_SIZE);

            // Each group handles different d_block columns (stride = 4 groups * 16 cols = 64)
            for (uint d_block = simd_group_id * 16; d_block < head_dim; d_block += 4 * 16) {
                // Load V blocks (FP16)
                simdgroup_half8x8 v0_0, v0_1, v1_0, v1_1;
                simdgroup_load(v0_0, shared_KV + 0 * head_dim + d_block + 0, head_dim);
                simdgroup_load(v0_1, shared_KV + 0 * head_dim + d_block + 8, head_dim);
                simdgroup_load(v1_0, shared_KV + 8 * head_dim + d_block + 0, head_dim);
                simdgroup_load(v1_1, shared_KV + 8 * head_dim + d_block + 8, head_dim);

                // Compute weights @ V in FP16, store to temp, convert to FP32 for accumulation
                simdgroup_half8x8 r_00 = simdgroup_half8x8(0.0h);
                simdgroup_half8x8 r_01 = simdgroup_half8x8(0.0h);
                simdgroup_half8x8 r_10 = simdgroup_half8x8(0.0h);
                simdgroup_half8x8 r_11 = simdgroup_half8x8(0.0h);

                // output[0:8] += weights[0:8,0:8] @ V[0:8] + weights[0:8,8:16] @ V[8:16]
                simdgroup_multiply_accumulate(r_00, w00, v0_0, r_00);
                simdgroup_multiply_accumulate(r_00, w01, v1_0, r_00);
                simdgroup_multiply_accumulate(r_01, w00, v0_1, r_01);
                simdgroup_multiply_accumulate(r_01, w01, v1_1, r_01);

                // output[8:16] += weights[8:16,0:8] @ V[0:8] + weights[8:16,8:16] @ V[8:16]
                simdgroup_multiply_accumulate(r_10, w10, v0_0, r_10);
                simdgroup_multiply_accumulate(r_10, w11, v1_0, r_10);
                simdgroup_multiply_accumulate(r_11, w10, v0_1, r_11);
                simdgroup_multiply_accumulate(r_11, w11, v1_1, r_11);

                // Store FP16 results to per-group scratch area
                // Each group gets its own 16x16 temp area (no race with V reads)
                threadgroup half temp_v_all[4 * TILE_SIZE * 16];
                threadgroup half* temp_v = temp_v_all + simd_group_id * TILE_SIZE * 16;
                simdgroup_store(r_00, temp_v + 0 * 16 + 0, 16);
                simdgroup_store(r_01, temp_v + 0 * 16 + 8, 16);
                simdgroup_store(r_10, temp_v + 8 * 16 + 0, 16);
                simdgroup_store(r_11, temp_v + 8 * 16 + 8, 16);

                // Convert to FP32 and add to output (only this group's lanes)
                for (uint idx = simd_lane_id; idx < TILE_SIZE * 16; idx += 32) {
                    uint row = idx / 16;
                    uint col = idx % 16;
                    shared_output[row * head_dim + d_block + col] += float(temp_v[row * 16 + col]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and write output as FP16 (each group handles its 4 rows)
    for (uint local_row = 0; local_row < 4; local_row++) {
        uint row = row_start + local_row;
        for (uint col = simd_lane_id; col < head_dim; col += 32) {
            uint q_idx = q_start + row;

            if (q_idx < seq_len_q) {
                float normalized = shared_output[row * head_dim + col] / softmax_states[row].sum_exp;
                output[qkv_offset + q_idx * head_dim + col] = half(normalized);
            }
        }
    }
}

/// FlashAttention 4-SIMD FP32 kernel (128 threads)
/// 4 SIMD groups, each handles 4 rows of the 16x16 Q tile
///
/// Memory layout:
/// - shared_Q: [16, head_dim] FP32
/// - shared_KV: [16, head_dim] FP32 (K then V, aliased)
/// - shared_scores: [16, 16] FP32
/// - shared_output: [16, head_dim] FP32
/// Total at head_dim=128: 16,384+16,384+1,024+16,384 = 50,176 bytes (fits 64KB)
///
/// Threadgroup: 128 threads (four SIMD groups)
/// Grid: [num_q_tiles, num_heads, batch]
kernel void flash_attention_4simd_fp32(
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
    threadgroup float* shared_KV        [[threadgroup(1)]],
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

    // Cooperative loading of Q tile (128 threads)
    const uint thread_id = simd_group_id * 32 + simd_lane_id;
    for (uint idx = thread_id; idx < TILE_SIZE * head_dim; idx += 128) {
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

    // Initialize FP32 output accumulator (128 threads)
    for (uint idx = thread_id; idx < TILE_SIZE * head_dim; idx += 128) {
        shared_output[idx] = 0.0f;
    }

    // Online softmax state for each SIMD group's rows
    threadgroup OnlineSoftmaxState softmax_states[TILE_SIZE];
    const uint row_start = simd_group_id * 4;  // Each group handles 4 rows

    if (simd_lane_id < 4) {
        uint row_idx = row_start + simd_lane_id;
        softmax_states[row_idx].max_val = -INFINITY;
        softmax_states[row_idx].sum_exp = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_k_tiles = (seq_len_k + TILE_SIZE - 1) / TILE_SIZE;

    for (uint k_tile_idx = 0; k_tile_idx < num_k_tiles; k_tile_idx++) {
        const uint k_start = k_tile_idx * TILE_SIZE;
        const uint k_end = min(k_start + TILE_SIZE, seq_len_k);
        const uint k_tile_size = k_end - k_start;

        // Cooperative load K tile (128 threads)
        for (uint idx = thread_id; idx < TILE_SIZE * head_dim; idx += 128) {
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

        // Compute Q @ K^T using SIMD matrix ops - each group computes one 8x8 quadrant
        // Group 0: scores[0:8, 0:8], Group 1: scores[0:8, 8:16]
        // Group 2: scores[8:16, 0:8], Group 3: scores[8:16, 8:16]
        {
            simdgroup_float8x8 score_block = simdgroup_float8x8(0.0f);
            uint q_row_offset = (simd_group_id < 2) ? 0 : 8;
            uint k_row_offset = (simd_group_id % 2 == 0) ? 0 : 8;

            for (uint d = 0; d < head_dim; d += 8) {
                simdgroup_float8x8 q_block, k_block;
                simdgroup_load(q_block, shared_Q + q_row_offset * head_dim + d, head_dim);
                simdgroup_load(k_block, shared_KV + k_row_offset * head_dim + d, head_dim);
                simdgroup_multiply_accumulate(score_block, q_block, k_block, score_block);
            }

            simdgroup_store(score_block, shared_scores + q_row_offset * TILE_SIZE + k_row_offset, TILE_SIZE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply scaling and masking (all 128 threads cooperatively)
        for (uint idx = thread_id; idx < TILE_SIZE * TILE_SIZE; idx += 128) {
            uint row = idx / TILE_SIZE;
            uint col = idx % TILE_SIZE;
            uint q_idx = q_start + row;
            uint k_idx = k_start + col;

            if (q_idx < seq_len_q && col < k_tile_size) {
                float scaled_score = shared_scores[row * TILE_SIZE + col] * scale;
                if (has_mask) {
                    scaled_score += mask[b * seq_len_q * seq_len_k + q_idx * seq_len_k + k_idx];
                }
                shared_scores[row * TILE_SIZE + col] = scaled_score;
            } else {
                shared_scores[row * TILE_SIZE + col] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax + weight computation using SIMD-parallel reduction (4-SIMD FP32)
        // Each SIMD group handles 4 rows; all 32 lanes do parallel max/sum
        threadgroup float rescale_factors[TILE_SIZE];
        threadgroup bool needs_rescale[TILE_SIZE];

        for (uint r = 0; r < 4; r++) {
            uint row = row_start + r;
            if (row < TILE_SIZE) {
                uint q_idx = q_start + row;
                if (q_idx < seq_len_q) {
                    float old_max = softmax_states[row].max_val;

                    float my_score = (simd_lane_id < k_tile_size) ?
                        shared_scores[row * TILE_SIZE + simd_lane_id] : -INFINITY;

                    float tile_max = simd_max(my_score);
                    float my_exp = (simd_lane_id < k_tile_size) ?
                        exp(my_score - tile_max) : 0.0f;
                    float tile_sum = simd_sum(my_exp);

                    if (simd_lane_id == 0) {
                        float new_max = max(old_max, tile_max);
                        float old_scale = exp(old_max - new_max);
                        float tile_scale = exp(tile_max - new_max);
                        float new_sum = softmax_states[row].sum_exp * old_scale +
                                        tile_sum * tile_scale;
                        softmax_states[row].max_val = new_max;
                        softmax_states[row].sum_exp = new_sum;
                        needs_rescale[row] = (new_max > old_max);
                        if (needs_rescale[row]) {
                            rescale_factors[row] = old_scale;
                        }
                    }

                    float new_max_bcast = simd_broadcast_first(
                        (simd_lane_id == 0) ? softmax_states[row].max_val : 0.0f);
                    if (simd_lane_id < TILE_SIZE) {
                        shared_scores[row * TILE_SIZE + simd_lane_id] =
                            (simd_lane_id < k_tile_size) ?
                                exp(my_score - new_max_bcast) : 0.0f;
                    }
                } else {
                    if (simd_lane_id == 0) {
                        needs_rescale[row] = false;
                    }
                    if (simd_lane_id < TILE_SIZE) {
                        shared_scores[row * TILE_SIZE + simd_lane_id] = 0.0f;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale using ALL 128 threads
        for (uint row = 0; row < TILE_SIZE; row++) {
            if (needs_rescale[row]) {
                for (uint d = thread_id; d < head_dim; d += 128) {
                    shared_output[row * head_dim + d] *= rescale_factors[row];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative load V tile (128 threads)
        for (uint idx = thread_id; idx < TILE_SIZE * head_dim; idx += 128) {
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

        // Accumulate weighted V using SIMD matrix ops (FP32)
        // Load weight blocks (all groups load the same 16x16 weight matrix as four 8x8 quadrants)
        {
            simdgroup_float8x8 w00, w01, w10, w11;
            simdgroup_load(w00, shared_scores + 0 * TILE_SIZE + 0, TILE_SIZE);
            simdgroup_load(w01, shared_scores + 0 * TILE_SIZE + 8, TILE_SIZE);
            simdgroup_load(w10, shared_scores + 8 * TILE_SIZE + 0, TILE_SIZE);
            simdgroup_load(w11, shared_scores + 8 * TILE_SIZE + 8, TILE_SIZE);

            // Each group handles different d_block columns (stride = 4 groups * 16 cols = 64)
            for (uint d_block = simd_group_id * 16; d_block < head_dim; d_block += 4 * 16) {
                // Load V blocks: V[0:8, d_block:d_block+16] and V[8:16, d_block:d_block+16]
                simdgroup_float8x8 v0_0, v0_1, v1_0, v1_1;
                simdgroup_load(v0_0, shared_KV + 0 * head_dim + d_block + 0, head_dim);
                simdgroup_load(v0_1, shared_KV + 0 * head_dim + d_block + 8, head_dim);
                simdgroup_load(v1_0, shared_KV + 8 * head_dim + d_block + 0, head_dim);
                simdgroup_load(v1_1, shared_KV + 8 * head_dim + d_block + 8, head_dim);

                // Load current output blocks
                simdgroup_float8x8 out_00, out_01, out_10, out_11;
                simdgroup_load(out_00, shared_output + 0 * head_dim + d_block + 0, head_dim);
                simdgroup_load(out_01, shared_output + 0 * head_dim + d_block + 8, head_dim);
                simdgroup_load(out_10, shared_output + 8 * head_dim + d_block + 0, head_dim);
                simdgroup_load(out_11, shared_output + 8 * head_dim + d_block + 8, head_dim);

                // output[0:8] += weights[0:8,0:8] @ V[0:8] + weights[0:8,8:16] @ V[8:16]
                simdgroup_multiply_accumulate(out_00, w00, v0_0, out_00);
                simdgroup_multiply_accumulate(out_00, w01, v1_0, out_00);
                simdgroup_multiply_accumulate(out_01, w00, v0_1, out_01);
                simdgroup_multiply_accumulate(out_01, w01, v1_1, out_01);

                // output[8:16] += weights[8:16,0:8] @ V[0:8] + weights[8:16,8:16] @ V[8:16]
                simdgroup_multiply_accumulate(out_10, w10, v0_0, out_10);
                simdgroup_multiply_accumulate(out_10, w11, v1_0, out_10);
                simdgroup_multiply_accumulate(out_11, w10, v0_1, out_11);
                simdgroup_multiply_accumulate(out_11, w11, v1_1, out_11);

                // Store back
                simdgroup_store(out_00, shared_output + 0 * head_dim + d_block + 0, head_dim);
                simdgroup_store(out_01, shared_output + 0 * head_dim + d_block + 8, head_dim);
                simdgroup_store(out_10, shared_output + 8 * head_dim + d_block + 0, head_dim);
                simdgroup_store(out_11, shared_output + 8 * head_dim + d_block + 8, head_dim);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and write output as FP32 (each group handles its 4 rows)
    for (uint local_row = 0; local_row < 4; local_row++) {
        uint row = row_start + local_row;
        for (uint col = simd_lane_id; col < head_dim; col += 32) {
            uint q_idx = q_start + row;

            if (q_idx < seq_len_q) {
                float normalized = shared_output[row * head_dim + col] / softmax_states[row].sum_exp;
                output[qkv_offset + q_idx * head_dim + col] = normalized;
            }
        }
    }
}

// ============================================================================
// Multi-SIMD 256-thread FlashAttention kernels (8 SIMD groups)
// ============================================================================

// Multi-SIMD attention constants
constant constexpr uint ATTN_LARGE_THREADS = 256;
constant constexpr uint ATTN_LARGE_SIMD_SIZE = 32;
constant constexpr uint ATTN_LARGE_NUM_SIMD = ATTN_LARGE_THREADS / ATTN_LARGE_SIMD_SIZE; // 8
constant constexpr uint ATTN_LARGE_TILE = 16;  // Keep same tile size for threadgroup memory compatibility

/// FlashAttention kernel using 8 SIMD groups (256 threads) for FP32
/// Maximizes GPU occupancy (~25% vs ~3% with 32-thread variant)
///
/// Memory layout (same as tile16 FP32):
/// - shared_Q: [16, head_dim] FP32 = 16 * 128 * 4 = 8,192 bytes
/// - shared_KV: [16, head_dim] FP32 = 16 * 128 * 4 = 8,192 bytes (K then V, aliased)
/// - shared_scores: [16, 16] FP32 = 16 * 16 * 4 = 1,024 bytes
/// - shared_output: [16, head_dim] FP32 = 16 * 128 * 4 = 8,192 bytes
/// - shared_reduction: 16 floats (8 max + 8 sum for inter-SIMD reduction) = 64 bytes
/// Total: 25,664 bytes (fits in 32KB limit)
///
/// Threadgroup: 256 threads (eight SIMD groups)
/// Grid: [num_q_tiles, num_heads, batch]
///
/// Parallelization Strategy:
/// - All 256 threads cooperatively load Q/K/V tiles (8x faster than 32 threads)
/// - SIMD groups 0-3 parallelize Q@K^T (each computes one 8x8 quadrant)
/// - All 8 SIMD groups parallelize V accumulation (each handles 16-column blocks)
/// - All 256 threads participate in scaling, masking, online softmax, rescaling, normalization
/// - Inter-SIMD reduction via shared_reduction buffer for softmax row max/sum
kernel void flash_attention_large_fp32(
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
    threadgroup float* shared_KV        [[threadgroup(1)]],
    threadgroup float* shared_scores    [[threadgroup(2)]],
    threadgroup float* shared_output    [[threadgroup(3)]],
    threadgroup float* shared_reduction [[threadgroup(4)]],  // For inter-SIMD reduction
    uint3 gid                           [[threadgroup_position_in_grid]],
    uint  tid                           [[thread_index_in_threadgroup]],
    uint  simd_lane_id                  [[thread_index_in_simdgroup]],
    uint  simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    const uint b = gid.z;
    const uint h = gid.y;
    const uint q_tile_idx = gid.x;

    if (b >= batch || h >= num_heads) return;

    const uint q_start = q_tile_idx * ATTN_LARGE_TILE;
    const uint qkv_offset = (b * num_heads + h) * seq_len_q * head_dim;
    const uint k_offset = (b * num_heads + h) * seq_len_k * head_dim;

    // Cooperative loading of Q tile (256 threads -- 8x faster than 32-thread kernel)
    for (uint idx = tid; idx < ATTN_LARGE_TILE * head_dim; idx += ATTN_LARGE_THREADS) {
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

    // Initialize output accumulator (256 threads)
    for (uint idx = tid; idx < ATTN_LARGE_TILE * head_dim; idx += ATTN_LARGE_THREADS) {
        shared_output[idx] = 0.0f;
    }

    // Initialize online softmax state
    threadgroup OnlineSoftmaxState softmax_states[ATTN_LARGE_TILE];
    if (tid < ATTN_LARGE_TILE) {
        softmax_states[tid].max_val = -INFINITY;
        softmax_states[tid].sum_exp = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_k_tiles = (seq_len_k + ATTN_LARGE_TILE - 1) / ATTN_LARGE_TILE;

    // Process K/V tiles
    for (uint k_tile_idx = 0; k_tile_idx < num_k_tiles; k_tile_idx++) {
        const uint k_start = k_tile_idx * ATTN_LARGE_TILE;
        const uint k_end = min(k_start + ATTN_LARGE_TILE, seq_len_k);
        const uint k_tile_size = k_end - k_start;

        // Cooperative load K tile (256 threads)
        for (uint idx = tid; idx < ATTN_LARGE_TILE * head_dim; idx += ATTN_LARGE_THREADS) {
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

        // Compute Q @ K^T using SIMD matrix ops (groups 0-3 each compute one quadrant)
        // Groups 0-3 parallelize the 4 quadrants; groups 4-7 idle here
        if (simd_group_id < 4) {
            simdgroup_float8x8 score_block = simdgroup_float8x8(0.0f);
            uint q_row_offset = (simd_group_id < 2) ? 0 : 8;
            uint k_row_offset = (simd_group_id % 2 == 0) ? 0 : 8;

            for (uint d = 0; d < head_dim; d += 8) {
                simdgroup_float8x8 q_block, k_block;
                simdgroup_load(q_block, shared_Q + q_row_offset * head_dim + d, head_dim);
                simdgroup_load(k_block, shared_KV + k_row_offset * head_dim + d, head_dim);
                simdgroup_multiply_accumulate(score_block, q_block, k_block, score_block);
            }

            simdgroup_store(score_block, shared_scores + q_row_offset * ATTN_LARGE_TILE + k_row_offset, ATTN_LARGE_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply scaling and mask (ALL 256 threads)
        for (uint idx = tid; idx < ATTN_LARGE_TILE * ATTN_LARGE_TILE; idx += ATTN_LARGE_THREADS) {
            uint row = idx / ATTN_LARGE_TILE;
            uint col = idx % ATTN_LARGE_TILE;
            uint q_idx = q_start + row;
            uint k_idx = k_start + col;

            if (q_idx < seq_len_q && col < k_tile_size) {
                float score = shared_scores[row * ATTN_LARGE_TILE + col] * scale;
                if (has_mask) {
                    score += mask[b * seq_len_q * seq_len_k + q_idx * seq_len_k + k_idx];
                }
                shared_scores[row * ATTN_LARGE_TILE + col] = score;
            } else {
                shared_scores[row * ATTN_LARGE_TILE + col] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax + weight computation using SIMD-parallel reduction (FP32)
        // Each SIMD group handles 2 rows; all 32 lanes participate in parallel
        // max/sum reduction via simd_max/simd_sum (replaces serial lane-0 loop)
        threadgroup float rescale_factors[ATTN_LARGE_TILE];
        threadgroup bool needs_rescale[ATTN_LARGE_TILE];

        {
            const uint rows_per_simd = 2;  // 16 rows / 8 SIMD groups
            const uint my_row_start = simd_group_id * rows_per_simd;

            for (uint r = 0; r < rows_per_simd; r++) {
                uint row = my_row_start + r;
                if (row < ATTN_LARGE_TILE) {
                    uint q_idx = q_start + row;
                    if (q_idx < seq_len_q) {
                        float old_max = softmax_states[row].max_val;

                        // Each lane reads one score (16 cols; lanes 16-31 get -INF)
                        float my_score = (simd_lane_id < k_tile_size) ?
                            shared_scores[row * ATTN_LARGE_TILE + simd_lane_id] : -INFINITY;

                        // Parallel max across all 32 lanes
                        float tile_max = simd_max(my_score);

                        // Each lane computes exp(score - tile_max)
                        float my_exp = (simd_lane_id < k_tile_size) ?
                            exp(my_score - tile_max) : 0.0f;

                        // Parallel sum across all 32 lanes
                        float tile_sum = simd_sum(my_exp);

                        // Lane 0 merges with running softmax state
                        if (simd_lane_id == 0) {
                            float new_max = max(old_max, tile_max);
                            float old_scale = exp(old_max - new_max);
                            float tile_scale = exp(tile_max - new_max);
                            float new_sum = softmax_states[row].sum_exp * old_scale +
                                            tile_sum * tile_scale;

                            softmax_states[row].max_val = new_max;
                            softmax_states[row].sum_exp = new_sum;

                            needs_rescale[row] = (new_max > old_max);
                            if (needs_rescale[row]) {
                                rescale_factors[row] = old_scale;
                            }
                        }

                        // All lanes write softmax weights directly
                        float new_max_bcast = simd_broadcast_first(
                            (simd_lane_id == 0) ? softmax_states[row].max_val : 0.0f);
                        if (simd_lane_id < ATTN_LARGE_TILE) {
                            shared_scores[row * ATTN_LARGE_TILE + simd_lane_id] =
                                (simd_lane_id < k_tile_size) ?
                                    exp(my_score - new_max_bcast) : 0.0f;
                        }
                    } else {
                        if (simd_lane_id == 0) {
                            needs_rescale[row] = false;
                        }
                        if (simd_lane_id < ATTN_LARGE_TILE) {
                            shared_scores[row * ATTN_LARGE_TILE + simd_lane_id] = 0.0f;
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale output using ALL 256 threads
        for (uint row = 0; row < ATTN_LARGE_TILE; row++) {
            if (needs_rescale[row]) {
                for (uint d = tid; d < head_dim; d += ATTN_LARGE_THREADS) {
                    shared_output[row * head_dim + d] *= rescale_factors[row];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative load V tile (256 threads)
        for (uint idx = tid; idx < ATTN_LARGE_TILE * head_dim; idx += ATTN_LARGE_THREADS) {
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

        // Accumulate weighted V using SIMD matrix ops (FP32)
        // Parallelize across all 8 SIMD groups - each handles 16-column blocks
        // For head_dim=128: 8 d_blocks, one per SIMD group (perfect load balance)
        // For head_dim=64: 4 d_blocks, 4 SIMD groups active
        // Hoist weight loads - same weights for all d_blocks
        {
            simdgroup_float8x8 w00, w01, w10, w11;
            simdgroup_load(w00, shared_scores + 0 * ATTN_LARGE_TILE + 0, ATTN_LARGE_TILE);
            simdgroup_load(w01, shared_scores + 0 * ATTN_LARGE_TILE + 8, ATTN_LARGE_TILE);
            simdgroup_load(w10, shared_scores + 8 * ATTN_LARGE_TILE + 0, ATTN_LARGE_TILE);
            simdgroup_load(w11, shared_scores + 8 * ATTN_LARGE_TILE + 8, ATTN_LARGE_TILE);

            for (uint d_block = simd_group_id * 16; d_block < head_dim; d_block += ATTN_LARGE_NUM_SIMD * 16) {
                simdgroup_float8x8 out_00, out_01, out_10, out_11;
                simdgroup_float8x8 v0_0, v0_1, v1_0, v1_1;

                simdgroup_load(out_00, shared_output + 0 * head_dim + d_block + 0, head_dim);
                simdgroup_load(out_01, shared_output + 0 * head_dim + d_block + 8, head_dim);
                simdgroup_load(out_10, shared_output + 8 * head_dim + d_block + 0, head_dim);
                simdgroup_load(out_11, shared_output + 8 * head_dim + d_block + 8, head_dim);

                simdgroup_load(v0_0, shared_KV + 0 * head_dim + d_block + 0, head_dim);
                simdgroup_load(v0_1, shared_KV + 0 * head_dim + d_block + 8, head_dim);
                simdgroup_load(v1_0, shared_KV + 8 * head_dim + d_block + 0, head_dim);
                simdgroup_load(v1_1, shared_KV + 8 * head_dim + d_block + 8, head_dim);

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
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and write output (ALL 256 threads)
    for (uint idx = tid; idx < ATTN_LARGE_TILE * head_dim; idx += ATTN_LARGE_THREADS) {
        uint row = idx / head_dim;
        uint col = idx % head_dim;
        uint q_idx = q_start + row;

        if (q_idx < seq_len_q) {
            float normalized = shared_output[row * head_dim + col] / softmax_states[row].sum_exp;
            output[qkv_offset + q_idx * head_dim + col] = normalized;
        }
    }
}

/// FlashAttention kernel using 8 SIMD groups (256 threads) for FP16
/// Same structure as flash_attention_large_fp32 but with half input/output, FP32 accumulation
///
/// Memory layout (same as tile16 FP16):
/// - shared_Q: [16, head_dim] FP16 = 16 * 128 * 2 = 4,096 bytes
/// - shared_KV: [16, head_dim] FP16 = 16 * 128 * 2 = 4,096 bytes (K then V, aliased)
/// - shared_scores: [16, 16] FP32 = 16 * 16 * 4 = 1,024 bytes
/// - shared_output: [16, head_dim] FP32 = 16 * 128 * 4 = 8,192 bytes (FP32 accumulation)
/// - shared_reduction: 16 floats = 64 bytes (inter-SIMD reduction)
/// Total: 17,472 bytes (fits in 32KB limit)
///
/// Threadgroup: 256 threads (eight SIMD groups)
/// Grid: [num_q_tiles, num_heads, batch]
///
/// Parallelization Strategy:
/// - All 256 threads cooperatively load Q/K/V tiles (8x faster than 32 threads)
/// - SIMD group 0 performs simdgroup_half8x8 matrix ops for Q@K^T
/// - All 8 SIMD groups parallelize V accumulation (each handles 16-column blocks)
/// - All 256 threads participate in scaling, masking, online softmax, rescaling, normalization
/// - Mixed precision: FP16 inputs, FP32 accumulation, FP16 output
kernel void flash_attention_large_fp16(
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
    threadgroup half* shared_KV         [[threadgroup(1)]],
    threadgroup float* shared_scores    [[threadgroup(2)]],  // FP32 for numerical stability
    threadgroup float* shared_output    [[threadgroup(3)]],  // FP32 accumulation
    threadgroup float* shared_reduction [[threadgroup(4)]],  // For inter-SIMD reduction
    uint3 gid                           [[threadgroup_position_in_grid]],
    uint  tid                           [[thread_index_in_threadgroup]],
    uint  simd_lane_id                  [[thread_index_in_simdgroup]],
    uint  simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    const uint b = gid.z;
    const uint h = gid.y;
    const uint q_tile_idx = gid.x;

    if (b >= batch || h >= num_heads) return;

    const uint q_start = q_tile_idx * ATTN_LARGE_TILE;
    const uint qkv_offset = (b * num_heads + h) * seq_len_q * head_dim;
    const uint k_offset = (b * num_heads + h) * seq_len_k * head_dim;

    // Cooperative loading of Q tile (256 threads)
    for (uint idx = tid; idx < ATTN_LARGE_TILE * head_dim; idx += ATTN_LARGE_THREADS) {
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

    // Initialize FP32 output accumulator (256 threads)
    for (uint idx = tid; idx < ATTN_LARGE_TILE * head_dim; idx += ATTN_LARGE_THREADS) {
        shared_output[idx] = 0.0f;
    }

    // Initialize online softmax state
    threadgroup OnlineSoftmaxState softmax_states[ATTN_LARGE_TILE];
    if (tid < ATTN_LARGE_TILE) {
        softmax_states[tid].max_val = -INFINITY;
        softmax_states[tid].sum_exp = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_k_tiles = (seq_len_k + ATTN_LARGE_TILE - 1) / ATTN_LARGE_TILE;

    for (uint k_tile_idx = 0; k_tile_idx < num_k_tiles; k_tile_idx++) {
        const uint k_start = k_tile_idx * ATTN_LARGE_TILE;
        const uint k_end = min(k_start + ATTN_LARGE_TILE, seq_len_k);
        const uint k_tile_size = k_end - k_start;

        // Cooperative load K tile (256 threads)
        for (uint idx = tid; idx < ATTN_LARGE_TILE * head_dim; idx += ATTN_LARGE_THREADS) {
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

        // Compute Q @ K^T with FP16 SIMD matrices (groups 0-3 each compute one quadrant)
        threadgroup half temp_scores_half[ATTN_LARGE_TILE * ATTN_LARGE_TILE];

        if (simd_group_id < 4) {
            simdgroup_half8x8 score_block = simdgroup_half8x8(half(0.0f));
            uint q_row_offset = (simd_group_id < 2) ? 0 : 8;
            uint k_row_offset = (simd_group_id % 2 == 0) ? 0 : 8;

            for (uint d = 0; d < head_dim; d += 8) {
                simdgroup_half8x8 q_block, k_block;
                simdgroup_load(q_block, shared_Q + q_row_offset * head_dim + d, head_dim);
                simdgroup_load(k_block, shared_KV + k_row_offset * head_dim + d, head_dim);
                simdgroup_multiply_accumulate(score_block, q_block, k_block, score_block);
            }

            simdgroup_store(score_block, temp_scores_half + q_row_offset * ATTN_LARGE_TILE + k_row_offset, ATTN_LARGE_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Convert to FP32 and apply scaling/masking (ALL 256 threads)
        for (uint idx = tid; idx < ATTN_LARGE_TILE * ATTN_LARGE_TILE; idx += ATTN_LARGE_THREADS) {
            uint row = idx / ATTN_LARGE_TILE;
            uint col = idx % ATTN_LARGE_TILE;
            uint q_idx = q_start + row;
            uint k_idx = k_start + col;

            if (q_idx < seq_len_q && col < k_tile_size) {
                float score = float(temp_scores_half[row * ATTN_LARGE_TILE + col]) * scale;
                if (has_mask) {
                    score += float(mask[b * seq_len_q * seq_len_k + q_idx * seq_len_k + k_idx]);
                }
                shared_scores[row * ATTN_LARGE_TILE + col] = score;
            } else {
                shared_scores[row * ATTN_LARGE_TILE + col] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax + weight computation using SIMD-parallel reduction (FP16 kernel)
        // Each SIMD group handles 2 rows; all 32 lanes participate in parallel
        // max/sum reduction via simd_max/simd_sum (replaces serial lane-0 loop)
        threadgroup float rescale_factors[ATTN_LARGE_TILE];
        threadgroup bool needs_rescale[ATTN_LARGE_TILE];

        {
            const uint rows_per_simd = 2;  // 16 rows / 8 SIMD groups
            const uint my_row_start = simd_group_id * rows_per_simd;

            for (uint r = 0; r < rows_per_simd; r++) {
                uint row = my_row_start + r;
                if (row < ATTN_LARGE_TILE) {
                    uint q_idx = q_start + row;
                    if (q_idx < seq_len_q) {
                        float old_max = softmax_states[row].max_val;

                        // Each lane reads one score (16 cols; lanes 16-31 get -INF)
                        float my_score = (simd_lane_id < k_tile_size) ?
                            shared_scores[row * ATTN_LARGE_TILE + simd_lane_id] : -INFINITY;

                        // Parallel max across all 32 lanes
                        float tile_max = simd_max(my_score);

                        // Each lane computes exp(score - tile_max)
                        float my_exp = (simd_lane_id < k_tile_size) ?
                            exp(my_score - tile_max) : 0.0f;

                        // Parallel sum across all 32 lanes
                        float tile_sum = simd_sum(my_exp);

                        // Lane 0 merges with running softmax state
                        if (simd_lane_id == 0) {
                            float new_max = max(old_max, tile_max);
                            float old_scale = exp(old_max - new_max);
                            float tile_scale = exp(tile_max - new_max);
                            float new_sum = softmax_states[row].sum_exp * old_scale +
                                            tile_sum * tile_scale;

                            softmax_states[row].max_val = new_max;
                            softmax_states[row].sum_exp = new_sum;

                            needs_rescale[row] = (new_max > old_max);
                            if (needs_rescale[row]) {
                                rescale_factors[row] = old_scale;
                            }
                        }

                        // All lanes write softmax weights directly
                        float new_max_bcast = simd_broadcast_first(
                            (simd_lane_id == 0) ? softmax_states[row].max_val : 0.0f);
                        if (simd_lane_id < ATTN_LARGE_TILE) {
                            shared_scores[row * ATTN_LARGE_TILE + simd_lane_id] =
                                (simd_lane_id < k_tile_size) ?
                                    exp(my_score - new_max_bcast) : 0.0f;
                        }
                    } else {
                        if (simd_lane_id == 0) {
                            needs_rescale[row] = false;
                        }
                        if (simd_lane_id < ATTN_LARGE_TILE) {
                            shared_scores[row * ATTN_LARGE_TILE + simd_lane_id] = 0.0f;
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale output using ALL 256 threads
        for (uint row = 0; row < ATTN_LARGE_TILE; row++) {
            if (needs_rescale[row]) {
                for (uint d = tid; d < head_dim; d += ATTN_LARGE_THREADS) {
                    shared_output[row * head_dim + d] *= rescale_factors[row];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative load V tile (256 threads)
        for (uint idx = tid; idx < ATTN_LARGE_TILE * head_dim; idx += ATTN_LARGE_THREADS) {
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

        // Accumulate weighted V using SIMD matrix ops (FP16 kernel)
        // Convert FP32 weights to FP16 once, then iterate d_blocks
        {
            threadgroup half temp_weights_half[ATTN_LARGE_TILE * ATTN_LARGE_TILE];
            for (uint i = tid; i < ATTN_LARGE_TILE * ATTN_LARGE_TILE; i += ATTN_LARGE_THREADS) {
                temp_weights_half[i] = half(shared_scores[i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Hoist weight loads - same weights for all d_blocks
            simdgroup_half8x8 w00, w01, w10, w11;
            simdgroup_load(w00, temp_weights_half + 0 * ATTN_LARGE_TILE + 0, ATTN_LARGE_TILE);
            simdgroup_load(w01, temp_weights_half + 0 * ATTN_LARGE_TILE + 8, ATTN_LARGE_TILE);
            simdgroup_load(w10, temp_weights_half + 8 * ATTN_LARGE_TILE + 0, ATTN_LARGE_TILE);
            simdgroup_load(w11, temp_weights_half + 8 * ATTN_LARGE_TILE + 8, ATTN_LARGE_TILE);

            // Parallelize across all 8 SIMD groups - each handles 16-column blocks
            // For head_dim=128: 8 d_blocks, one per SIMD group (perfect load balance)
            // For head_dim=64: 4 d_blocks, 4 SIMD groups active
            for (uint d_block = simd_group_id * 16; d_block < head_dim; d_block += ATTN_LARGE_NUM_SIMD * 16) {
                simdgroup_float8x8 out_00, out_01, out_10, out_11;

                simdgroup_load(out_00, shared_output + 0 * head_dim + d_block + 0, head_dim);
                simdgroup_load(out_01, shared_output + 0 * head_dim + d_block + 8, head_dim);
                simdgroup_load(out_10, shared_output + 8 * head_dim + d_block + 0, head_dim);
                simdgroup_load(out_11, shared_output + 8 * head_dim + d_block + 8, head_dim);

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
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and write output as FP16 (ALL 256 threads)
    for (uint idx = tid; idx < ATTN_LARGE_TILE * head_dim; idx += ATTN_LARGE_THREADS) {
        uint row = idx / head_dim;
        uint col = idx % head_dim;
        uint q_idx = q_start + row;

        if (q_idx < seq_len_q) {
            float normalized = shared_output[row * head_dim + col] / softmax_states[row].sum_exp;
            output[qkv_offset + q_idx * head_dim + col] = half(normalized);
        }
    }
}

