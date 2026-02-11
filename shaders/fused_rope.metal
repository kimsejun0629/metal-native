//
// fused_rope.metal
// Metal Shading Language kernel for Fused Rotary Position Embedding (RoPE)
//
// RoPE applies rotary embeddings to Q and K tensors simultaneously in-place.
// For each position pos and dimension pair (2i, 2i+1):
//   cos_val = cos_cache[pos * half_dim + i]
//   sin_val = sin_cache[pos * half_dim + i]
//
//   q[..., 2*i]   = q_real * cos_val - q_imag * sin_val
//   q[..., 2*i+1] = q_real * sin_val + q_imag * cos_val
//   (same for K tensor)
//
// Grid dispatch: [head_dim/2, seq_len, batch_size * num_heads]
//

#include <metal_stdlib>
using namespace metal;

//
// FP32 Fused RoPE Kernel
//
// Applies rotary position embeddings to Q and K tensors in a single pass.
// Q and K have shape [batch, heads, seq_len, head_dim] and are modified in-place.
// cos_cache and sin_cache have shape [max_seq_len, head_dim/2].
//
kernel void fused_rope_fp32(
    device float* q [[buffer(0)]],                  // in-place Q tensor
    device float* k [[buffer(1)]],                  // in-place K tensor
    device const float* cos_cache [[buffer(2)]],    // precomputed cosines
    device const float* sin_cache [[buffer(3)]],    // precomputed sines
    constant uint& batch_size [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& start_pos [[buffer(8)]],         // offset into cache
    uint3 tid [[thread_position_in_grid]])
{
    const uint pair_idx = tid.x;         // dimension pair index [0, head_dim/2)
    const uint seq_idx = tid.y;          // sequence position [0, seq_len)
    const uint bh_idx = tid.z;           // flattened batch-head index

    const uint half_dim = head_dim / 2;

    // Bounds check
    if (pair_idx >= half_dim || seq_idx >= seq_len || bh_idx >= batch_size * num_heads)
        return;

    // Cache lookup: absolute position in sequence
    const uint pos = start_pos + seq_idx;
    const float cos_val = cos_cache[pos * half_dim + pair_idx];
    const float sin_val = sin_cache[pos * half_dim + pair_idx];

    // Calculate base offset for this (batch, head, seq_pos) slice
    // Layout: [batch, heads, seq_len, head_dim] - row-major contiguous
    const uint base = bh_idx * seq_len * head_dim + seq_idx * head_dim;
    const uint dim0 = base + 2 * pair_idx;      // real component
    const uint dim1 = base + 2 * pair_idx + 1;  // imaginary component

    // Apply RoPE rotation to Q tensor
    const float q_real = q[dim0];
    const float q_imag = q[dim1];
    q[dim0] = q_real * cos_val - q_imag * sin_val;
    q[dim1] = q_real * sin_val + q_imag * cos_val;

    // Apply RoPE rotation to K tensor
    const float k_real = k[dim0];
    const float k_imag = k[dim1];
    k[dim0] = k_real * cos_val - k_imag * sin_val;
    k[dim1] = k_real * sin_val + k_imag * cos_val;
}

//
// FP16 Fused RoPE Kernel
//
// Same algorithm as FP32, but performs rotation in FP32 for numerical stability.
// Input/output tensors are half-precision.
//
kernel void fused_rope_fp16(
    device half* q [[buffer(0)]],                   // in-place Q tensor
    device half* k [[buffer(1)]],                   // in-place K tensor
    device const half* cos_cache [[buffer(2)]],     // precomputed cosines
    device const half* sin_cache [[buffer(3)]],     // precomputed sines
    constant uint& batch_size [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& start_pos [[buffer(8)]],         // offset into cache
    uint3 tid [[thread_position_in_grid]])
{
    const uint pair_idx = tid.x;         // dimension pair index [0, head_dim/2)
    const uint seq_idx = tid.y;          // sequence position [0, seq_len)
    const uint bh_idx = tid.z;           // flattened batch-head index

    const uint half_dim = head_dim / 2;

    // Bounds check
    if (pair_idx >= half_dim || seq_idx >= seq_len || bh_idx >= batch_size * num_heads)
        return;

    // Cache lookup: absolute position in sequence
    const uint pos = start_pos + seq_idx;
    // Promote to FP32 for numerical stability
    const float cos_val = float(cos_cache[pos * half_dim + pair_idx]);
    const float sin_val = float(sin_cache[pos * half_dim + pair_idx]);

    // Calculate base offset for this (batch, head, seq_pos) slice
    // Layout: [batch, heads, seq_len, head_dim] - row-major contiguous
    const uint base = bh_idx * seq_len * head_dim + seq_idx * head_dim;
    const uint dim0 = base + 2 * pair_idx;      // real component
    const uint dim1 = base + 2 * pair_idx + 1;  // imaginary component

    // Apply RoPE rotation to Q tensor (compute in FP32, write as half)
    const float q_real = float(q[dim0]);
    const float q_imag = float(q[dim1]);
    q[dim0] = half(q_real * cos_val - q_imag * sin_val);
    q[dim1] = half(q_real * sin_val + q_imag * cos_val);

    // Apply RoPE rotation to K tensor (compute in FP32, write as half)
    const float k_real = float(k[dim0]);
    const float k_imag = float(k[dim1]);
    k[dim0] = half(k_real * cos_val - k_imag * sin_val);
    k[dim1] = half(k_real * sin_val + k_imag * cos_val);
}
