/// @file embedding_kernel.metal
/// @brief Metal kernels for embedding and positional encoding operations.

#include <metal_stdlib>
#include "common/metal_types.h"

using namespace metal;

// ---------------------------------------------------------------------------
// Embedding lookup kernels
// ---------------------------------------------------------------------------

/// Embedding table lookup for FP32.
///
/// Each thread handles one index lookup, copying an entire embedding vector.
kernel void embedding_lookup_fp32(
    device const int32_t* indices     [[buffer(0)]],  // [num_indices]
    device const float*   weight      [[buffer(1)]],  // [vocab_size, embed_dim]
    device float*         output      [[buffer(2)]],  // [num_indices, embed_dim]
    constant uint32_t*    constants   [[buffer(3)]],  // [num_indices, vocab_size, embed_dim]
    uint gid [[thread_position_in_grid]])
{
    const uint32_t num_indices = constants[0];
    const uint32_t vocab_size  = constants[1];
    const uint32_t embed_dim   = constants[2];

    if (gid >= num_indices) return;

    // Read index and validate bounds
    const int32_t idx = indices[gid];
    if (idx < 0 || idx >= int32_t(vocab_size)) {
        // Out of bounds - fill with zeros
        for (uint32_t d = 0; d < embed_dim; ++d) {
            output[gid * embed_dim + d] = 0.0f;
        }
        return;
    }

    // Copy embedding vector
    const uint32_t weight_offset = uint32_t(idx) * embed_dim;
    const uint32_t output_offset = gid * embed_dim;

    for (uint32_t d = 0; d < embed_dim; ++d) {
        output[output_offset + d] = weight[weight_offset + d];
    }
}

/// Embedding table lookup for FP16.
kernel void embedding_lookup_fp16(
    device const int32_t* indices     [[buffer(0)]],
    device const half*    weight      [[buffer(1)]],
    device half*          output      [[buffer(2)]],
    constant uint32_t*    constants   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint32_t num_indices = constants[0];
    const uint32_t vocab_size  = constants[1];
    const uint32_t embed_dim   = constants[2];

    if (gid >= num_indices) return;

    const int32_t idx = indices[gid];
    if (idx < 0 || idx >= int32_t(vocab_size)) {
        for (uint32_t d = 0; d < embed_dim; ++d) {
            output[gid * embed_dim + d] = half(0.0h);
        }
        return;
    }

    const uint32_t weight_offset = uint32_t(idx) * embed_dim;
    const uint32_t output_offset = gid * embed_dim;

    for (uint32_t d = 0; d < embed_dim; ++d) {
        output[output_offset + d] = weight[weight_offset + d];
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding (RoPE) kernels
// ---------------------------------------------------------------------------

/// RoPE embedding for FP32.
///
/// RoPE rotates pairs of elements in the embedding dimension using precomputed
/// sin/cos tables indexed by position. Each thread handles one rotation pair.
///
/// Input shape: [batch, seq_len, num_heads, head_dim]
/// Each thread processes one (batch, seq, head, dim_pair) element.
kernel void rope_embedding_fp32(
    device const float*   input        [[buffer(0)]],  // [batch, seq_len, num_heads, head_dim]
    device const float*   cos_cache    [[buffer(1)]],  // [max_seq_len, head_dim]
    device const float*   sin_cache    [[buffer(2)]],  // [max_seq_len, head_dim]
    device const int32_t* position_ids [[buffer(3)]],  // [batch, seq_len]
    device float*         output       [[buffer(4)]],  // [batch, seq_len, num_heads, head_dim]
    constant uint32_t*    constants    [[buffer(5)]],  // [batch, seq_len, num_heads, head_dim]
    uint gid [[thread_position_in_grid]])
{
    const uint32_t batch     = constants[0];
    const uint32_t seq_len   = constants[1];
    const uint32_t num_heads = constants[2];
    const uint32_t head_dim  = constants[3];

    const uint32_t half_dim = head_dim / 2;
    const uint32_t total_pairs = batch * seq_len * num_heads * half_dim;

    if (gid >= total_pairs) return;

    // Decode thread index
    uint32_t idx = gid;
    const uint32_t pair_idx = idx % half_dim;
    idx /= half_dim;
    const uint32_t head_idx = idx % num_heads;
    idx /= num_heads;
    const uint32_t seq_idx = idx % seq_len;
    const uint32_t batch_idx = idx / seq_len;

    // Get position for this batch/sequence element
    const int32_t pos = position_ids[batch_idx * seq_len + seq_idx];

    // Load sin/cos values for this position and dimension
    const uint32_t cache_offset = uint32_t(pos) * head_dim;
    const float cos_val = cos_cache[cache_offset + pair_idx];
    const float sin_val = sin_cache[cache_offset + pair_idx];

    // Load input pair (x, y) at positions [2*pair_idx] and [2*pair_idx + 1]
    const uint32_t input_base = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    const float x = input[input_base + 2 * pair_idx];
    const float y = input[input_base + 2 * pair_idx + 1];

    // Apply rotation: (x', y') = (x*cos - y*sin, x*sin + y*cos)
    const float x_rotated = x * cos_val - y * sin_val;
    const float y_rotated = x * sin_val + y * cos_val;

    // Write output
    const uint32_t output_base = input_base;
    output[output_base + 2 * pair_idx]     = x_rotated;
    output[output_base + 2 * pair_idx + 1] = y_rotated;
}

/// RoPE embedding for FP16.
kernel void rope_embedding_fp16(
    device const half*    input        [[buffer(0)]],
    device const half*    cos_cache    [[buffer(1)]],
    device const half*    sin_cache    [[buffer(2)]],
    device const int32_t* position_ids [[buffer(3)]],
    device half*          output       [[buffer(4)]],
    constant uint32_t*    constants    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint32_t batch     = constants[0];
    const uint32_t seq_len   = constants[1];
    const uint32_t num_heads = constants[2];
    const uint32_t head_dim  = constants[3];

    const uint32_t half_dim = head_dim / 2;
    const uint32_t total_pairs = batch * seq_len * num_heads * half_dim;

    if (gid >= total_pairs) return;

    uint32_t idx = gid;
    const uint32_t pair_idx = idx % half_dim;
    idx /= half_dim;
    const uint32_t head_idx = idx % num_heads;
    idx /= num_heads;
    const uint32_t seq_idx = idx % seq_len;
    const uint32_t batch_idx = idx / seq_len;

    const int32_t pos = position_ids[batch_idx * seq_len + seq_idx];

    const uint32_t cache_offset = uint32_t(pos) * head_dim;
    const half cos_val = cos_cache[cache_offset + pair_idx];
    const half sin_val = sin_cache[cache_offset + pair_idx];

    const uint32_t input_base = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    const half x = input[input_base + 2 * pair_idx];
    const half y = input[input_base + 2 * pair_idx + 1];

    const half x_rotated = x * cos_val - y * sin_val;
    const half y_rotated = x * sin_val + y * cos_val;

    const uint32_t output_base = input_base;
    output[output_base + 2 * pair_idx]     = x_rotated;
    output[output_base + 2 * pair_idx + 1] = y_rotated;
}
