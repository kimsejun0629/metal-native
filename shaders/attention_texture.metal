#include <metal_stdlib>
#include "common/metal_types.h"
using namespace metal;

// Tile size for texture-backed attention
constant constexpr uint TEX_TILE_SIZE = 16;

/// FlashAttention with texture-backed score cache (FP32 MHA only)
/// Uses texture2d for QK^T scores to leverage texture cache hardware.
/// Q, K, V are still in device buffers; only scores use texture.
kernel void flash_attention_texture_fp32(
    device const float* Q          [[buffer(0)]],
    device const float* K          [[buffer(1)]],
    device const float* V          [[buffer(2)]],
    device const float* mask       [[buffer(3)]],
    device float*       output     [[buffer(4)]],
    constant uint&      batch      [[buffer(5)]],
    constant uint&      num_heads  [[buffer(6)]],
    constant uint&      seq_len_q  [[buffer(7)]],
    constant uint&      seq_len_k  [[buffer(8)]],
    constant uint&      head_dim   [[buffer(9)]],
    constant float&     scale      [[buffer(10)]],
    constant bool&      has_mask   [[buffer(11)]],
    texture2d<float, access::read_write> score_tex [[texture(0)]],
    threadgroup float*  shared_Q       [[threadgroup(0)]],
    threadgroup float*  shared_K       [[threadgroup(1)]],
    threadgroup float*  shared_output  [[threadgroup(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    // Grid: [num_q_tiles, num_heads, batch]
    const uint q_tile = tgid.x;
    const uint head = tgid.y;
    const uint b = tgid.z;

    const uint q_start = q_tile * TEX_TILE_SIZE;
    if (q_start >= seq_len_q) return;

    const uint q_tile_size = min(TEX_TILE_SIZE, seq_len_q - q_start);

    // Offset to this head's Q, K, V, output
    const uint qkv_stride = seq_len_q * head_dim;
    const uint kv_stride = seq_len_k * head_dim;
    const uint head_offset_q = (b * num_heads + head) * qkv_stride;
    const uint head_offset_k = (b * num_heads + head) * kv_stride;
    const uint head_offset_o = head_offset_q;

    // Initialize output accumulator and running max/sum for online softmax
    float row_max[TEX_TILE_SIZE];
    float row_sum[TEX_TILE_SIZE];

    // Init shared_output to 0
    for (uint i = tid; i < TEX_TILE_SIZE * head_dim; i += 32) {
        shared_output[i] = 0.0f;
    }
    for (uint i = 0; i < TEX_TILE_SIZE; i++) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load Q tile into shared memory
    for (uint i = tid; i < q_tile_size * head_dim; i += 32) {
        uint row = i / head_dim;
        uint col = i % head_dim;
        shared_Q[row * head_dim + col] = Q[head_offset_q + (q_start + row) * head_dim + col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate over K tiles
    for (uint k_start = 0; k_start < seq_len_k; k_start += TEX_TILE_SIZE) {
        uint k_tile_size = min(TEX_TILE_SIZE, seq_len_k - k_start);

        // Load K tile
        for (uint i = tid; i < k_tile_size * head_dim; i += 32) {
            uint row = i / head_dim;
            uint col = i % head_dim;
            shared_K[row * head_dim + col] = K[head_offset_k + (k_start + row) * head_dim + col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute Q*K^T scores and write to texture
        for (uint qr = tid; qr < q_tile_size; qr += 32) {
            for (uint kr = 0; kr < k_tile_size; kr++) {
                float dot = 0.0f;
                for (uint d = 0; d < head_dim; d++) {
                    dot += shared_Q[qr * head_dim + d] * shared_K[kr * head_dim + d];
                }
                dot *= scale;

                // Apply mask if present
                if (has_mask) {
                    float mask_val = mask[(b * num_heads + head) * seq_len_q * seq_len_k +
                                        (q_start + qr) * seq_len_k + (k_start + kr)];
                    dot += mask_val;
                }

                // Write score to texture (leverages texture cache for subsequent read)
                score_tex.write(float4(dot, 0, 0, 0), uint2(kr, qr));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax: update max and rescale
        for (uint qr = tid; qr < q_tile_size; qr += 32) {
            float prev_max = row_max[qr];
            float new_max = prev_max;

            // Find new max from texture
            for (uint kr = 0; kr < k_tile_size; kr++) {
                float s = score_tex.read(uint2(kr, qr)).x;
                new_max = max(new_max, s);
            }

            // Rescale previous sum
            float rescale = exp(prev_max - new_max);
            row_sum[qr] *= rescale;
            row_max[qr] = new_max;

            // Rescale existing output
            for (uint d = 0; d < head_dim; d++) {
                shared_output[qr * head_dim + d] *= rescale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load V tile (reuse shared_K memory since scores are now in texture)
        for (uint i = tid; i < k_tile_size * head_dim; i += 32) {
            uint row = i / head_dim;
            uint col = i % head_dim;
            shared_K[row * head_dim + col] = V[head_offset_k + (k_start + row) * head_dim + col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate: exp(score - max) * V
        for (uint qr = tid; qr < q_tile_size; qr += 32) {
            for (uint kr = 0; kr < k_tile_size; kr++) {
                float s = score_tex.read(uint2(kr, qr)).x;
                float w = exp(s - row_max[qr]);
                row_sum[qr] += w;

                for (uint d = 0; d < head_dim; d++) {
                    shared_output[qr * head_dim + d] += w * shared_K[kr * head_dim + d];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize output by row_sum and write to global memory
    for (uint qr = tid; qr < q_tile_size; qr += 32) {
        float inv_sum = (row_sum[qr] > 0.0f) ? (1.0f / row_sum[qr]) : 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            output[head_offset_o + (q_start + qr) * head_dim + d] = shared_output[qr * head_dim + d] * inv_sum;
        }
    }
}
