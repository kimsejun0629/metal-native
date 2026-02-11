//
// fused_qkv_rope.metal
// Metal Shading Language kernel for Fused QKV Split + Reshape + RoPE
//
// This kernel takes a combined QKV projection tensor and performs three operations in one pass:
// 1. Split into separate Q, K, V tensors
// 2. Reshape from [batch, seq_len, 3 * num_heads * head_dim] to [batch, num_heads, seq_len, head_dim]
// 3. Apply RoPE (Rotary Position Embedding) to Q and K (V is not rotated)
//
// Input:
//   - qkv_proj: [batch, seq_len, 3 * num_heads * head_dim] - combined QKV projection
//   - cos_table: [max_seq_len, head_dim/2] - precomputed cosine values
//   - sin_table: [max_seq_len, head_dim/2] - precomputed sine values
//
// Output (3 separate buffers):
//   - Q: [batch, num_heads, seq_len, head_dim] with RoPE applied
//   - K: [batch, num_heads, seq_len, head_dim] with RoPE applied
//   - V: [batch, num_heads, seq_len, head_dim] (no RoPE, just reshaped)
//
// RoPE Algorithm:
//   For each dimension pair (2i, 2i+1):
//     cos_val = cos_table[start_pos + seq][i]
//     sin_val = sin_table[start_pos + seq][i]
//     x0 = input[..., 2*i]
//     x1 = input[..., 2*i+1]
//     output[..., 2*i]   = x0 * cos_val - x1 * sin_val
//     output[..., 2*i+1] = x0 * sin_val + x1 * cos_val
//
// Grid dispatch: [head_dim/2, seq_len, batch * num_heads]
//   - Each thread handles one (batch, head, seq, dim_pair) tuple
//   - tid.x = dim_pair index [0, head_dim/2)
//   - tid.y = sequence position [0, seq_len)
//   - tid.z = flattened batch-head index [0, batch * num_heads)
//

#include <metal_stdlib>
using namespace metal;

//
// FP32 Fused QKV Split + Reshape + RoPE Kernel
//
// Splits combined QKV projection, reshapes to attention format, and applies RoPE to Q and K.
// V is reshaped but not rotated.
//
kernel void fused_qkv_rope_fp32(
    device const float* qkv_proj [[buffer(0)]],     // [batch, seq_len, 3*num_heads*head_dim]
    device const float* cos_table [[buffer(1)]],    // [max_seq_len, head_dim/2]
    device const float* sin_table [[buffer(2)]],    // [max_seq_len, head_dim/2]
    device float* q_out [[buffer(3)]],              // [batch, num_heads, seq_len, head_dim]
    device float* k_out [[buffer(4)]],              // [batch, num_heads, seq_len, head_dim]
    device float* v_out [[buffer(5)]],              // [batch, num_heads, seq_len, head_dim]
    constant uint& batch [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& num_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& start_pos [[buffer(10)]],        // KV cache offset
    uint3 tid [[thread_position_in_grid]])
{
    const uint pair_idx = tid.x;    // dimension pair index [0, head_dim/2)
    const uint seq_idx = tid.y;     // sequence position [0, seq_len)
    const uint bh_idx = tid.z;      // flattened batch-head index [0, batch*num_heads)

    const uint half_dim = head_dim / 2;

    // Bounds check
    if (pair_idx >= half_dim || seq_idx >= seq_len || bh_idx >= batch * num_heads)
        return;

    // Decompose flattened batch-head index
    const uint b = bh_idx / num_heads;
    const uint h = bh_idx % num_heads;

    // Input layout: [batch, seq_len, 3 * num_heads * head_dim]
    // The 3 * num_heads * head_dim dimension is structured as: [Q0...Qn, K0...Kn, V0...Vn]
    // where each Qi, Ki, Vi has shape [head_dim] for head i
    const uint qkv_dim = 3 * num_heads * head_dim;
    const uint qkv_base = b * seq_len * qkv_dim + seq_idx * qkv_dim;

    // Offsets for Q, K, V within the combined projection
    const uint q_offset = qkv_base + h * head_dim;
    const uint k_offset = qkv_base + num_heads * head_dim + h * head_dim;
    const uint v_offset = qkv_base + 2 * num_heads * head_dim + h * head_dim;

    // Output layout: [batch, num_heads, seq_len, head_dim]
    const uint out_base = bh_idx * seq_len * head_dim + seq_idx * head_dim;
    const uint dim0 = out_base + 2 * pair_idx;      // real component
    const uint dim1 = out_base + 2 * pair_idx + 1;  // imaginary component

    // Load Q values from input
    const float q_real = qkv_proj[q_offset + 2 * pair_idx];
    const float q_imag = qkv_proj[q_offset + 2 * pair_idx + 1];

    // Load K values from input
    const float k_real = qkv_proj[k_offset + 2 * pair_idx];
    const float k_imag = qkv_proj[k_offset + 2 * pair_idx + 1];

    // Load V values from input (no rotation needed)
    const float v_real = qkv_proj[v_offset + 2 * pair_idx];
    const float v_imag = qkv_proj[v_offset + 2 * pair_idx + 1];

    // Get RoPE values
    const uint pos = start_pos + seq_idx;
    const float cos_val = cos_table[pos * half_dim + pair_idx];
    const float sin_val = sin_table[pos * half_dim + pair_idx];

    // Apply RoPE rotation to Q and write
    q_out[dim0] = q_real * cos_val - q_imag * sin_val;
    q_out[dim1] = q_real * sin_val + q_imag * cos_val;

    // Apply RoPE rotation to K and write
    k_out[dim0] = k_real * cos_val - k_imag * sin_val;
    k_out[dim1] = k_real * sin_val + k_imag * cos_val;

    // Write V without rotation (just reshape)
    v_out[dim0] = v_real;
    v_out[dim1] = v_imag;
}

//
// FP16 Fused QKV Split + Reshape + RoPE Kernel
//
// Same algorithm as FP32, but performs RoPE rotation in FP32 for numerical stability.
// Input/output tensors are half-precision.
//
kernel void fused_qkv_rope_fp16(
    device const half* qkv_proj [[buffer(0)]],      // [batch, seq_len, 3*num_heads*head_dim]
    device const half* cos_table [[buffer(1)]],     // [max_seq_len, head_dim/2]
    device const half* sin_table [[buffer(2)]],     // [max_seq_len, head_dim/2]
    device half* q_out [[buffer(3)]],               // [batch, num_heads, seq_len, head_dim]
    device half* k_out [[buffer(4)]],               // [batch, num_heads, seq_len, head_dim]
    device half* v_out [[buffer(5)]],               // [batch, num_heads, seq_len, head_dim]
    constant uint& batch [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& num_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& start_pos [[buffer(10)]],        // KV cache offset
    uint3 tid [[thread_position_in_grid]])
{
    const uint pair_idx = tid.x;    // dimension pair index [0, head_dim/2)
    const uint seq_idx = tid.y;     // sequence position [0, seq_len)
    const uint bh_idx = tid.z;      // flattened batch-head index [0, batch*num_heads)

    const uint half_dim = head_dim / 2;

    // Bounds check
    if (pair_idx >= half_dim || seq_idx >= seq_len || bh_idx >= batch * num_heads)
        return;

    // Decompose flattened batch-head index
    const uint b = bh_idx / num_heads;
    const uint h = bh_idx % num_heads;

    // Input layout: [batch, seq_len, 3 * num_heads * head_dim]
    const uint qkv_dim = 3 * num_heads * head_dim;
    const uint qkv_base = b * seq_len * qkv_dim + seq_idx * qkv_dim;

    // Offsets for Q, K, V within the combined projection
    const uint q_offset = qkv_base + h * head_dim;
    const uint k_offset = qkv_base + num_heads * head_dim + h * head_dim;
    const uint v_offset = qkv_base + 2 * num_heads * head_dim + h * head_dim;

    // Output layout: [batch, num_heads, seq_len, head_dim]
    const uint out_base = bh_idx * seq_len * head_dim + seq_idx * head_dim;
    const uint dim0 = out_base + 2 * pair_idx;      // real component
    const uint dim1 = out_base + 2 * pair_idx + 1;  // imaginary component

    // Load Q values from input and promote to FP32
    const float q_real = float(qkv_proj[q_offset + 2 * pair_idx]);
    const float q_imag = float(qkv_proj[q_offset + 2 * pair_idx + 1]);

    // Load K values from input and promote to FP32
    const float k_real = float(qkv_proj[k_offset + 2 * pair_idx]);
    const float k_imag = float(qkv_proj[k_offset + 2 * pair_idx + 1]);

    // Load V values from input (no rotation needed)
    const half v_real = qkv_proj[v_offset + 2 * pair_idx];
    const half v_imag = qkv_proj[v_offset + 2 * pair_idx + 1];

    // Get RoPE values and promote to FP32
    const uint pos = start_pos + seq_idx;
    const float cos_val = float(cos_table[pos * half_dim + pair_idx]);
    const float sin_val = float(sin_table[pos * half_dim + pair_idx]);

    // Apply RoPE rotation to Q (compute in FP32, write as half)
    q_out[dim0] = half(q_real * cos_val - q_imag * sin_val);
    q_out[dim1] = half(q_real * sin_val + q_imag * cos_val);

    // Apply RoPE rotation to K (compute in FP32, write as half)
    k_out[dim0] = half(k_real * cos_val - k_imag * sin_val);
    k_out[dim1] = half(k_real * sin_val + k_imag * cos_val);

    // Write V without rotation (just reshape)
    v_out[dim0] = v_real;
    v_out[dim1] = v_imag;
}

//
// FP32 Vectorized (float4) Fused QKV Split + Reshape + RoPE Kernel
//
// Optimized version that processes 2 dimension pairs (4 elements) at a time using float4 loads.
// Requires head_dim to be divisible by 4.
//
kernel void fused_qkv_rope_fp32_vec4(
    device const float4* qkv_proj [[buffer(0)]],    // [batch, seq_len, 3*num_heads*head_dim] as float4
    device const float4* cos_table [[buffer(1)]],   // [max_seq_len, head_dim/4] as float4
    device const float4* sin_table [[buffer(2)]],   // [max_seq_len, head_dim/4] as float4
    device float4* q_out [[buffer(3)]],             // [batch, num_heads, seq_len, head_dim] as float4
    device float4* k_out [[buffer(4)]],             // [batch, num_heads, seq_len, head_dim] as float4
    device float4* v_out [[buffer(5)]],             // [batch, num_heads, seq_len, head_dim] as float4
    constant uint& batch [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& num_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& start_pos [[buffer(10)]],
    uint3 tid [[thread_position_in_grid]])
{
    const uint vec4_idx = tid.x;    // float4 index [0, head_dim/4)
    const uint seq_idx = tid.y;     // sequence position [0, seq_len)
    const uint bh_idx = tid.z;      // flattened batch-head index [0, batch*num_heads)

    const uint head_dim_div4 = head_dim / 4;

    // Bounds check
    if (vec4_idx >= head_dim_div4 || seq_idx >= seq_len || bh_idx >= batch * num_heads)
        return;

    // Decompose flattened batch-head index
    const uint b = bh_idx / num_heads;
    const uint h = bh_idx % num_heads;

    // Input layout: [batch, seq_len, 3 * num_heads * head_dim]
    const uint qkv_dim_div4 = (3 * num_heads * head_dim) / 4;
    const uint qkv_base = b * seq_len * qkv_dim_div4 + seq_idx * qkv_dim_div4;

    // Offsets for Q, K, V within the combined projection (in float4 units)
    const uint q_offset = qkv_base + h * head_dim_div4 + vec4_idx;
    const uint k_offset = qkv_base + num_heads * head_dim_div4 + h * head_dim_div4 + vec4_idx;
    const uint v_offset = qkv_base + 2 * num_heads * head_dim_div4 + h * head_dim_div4 + vec4_idx;

    // Output layout: [batch, num_heads, seq_len, head_dim] (in float4 units)
    const uint out_offset = bh_idx * seq_len * head_dim_div4 + seq_idx * head_dim_div4 + vec4_idx;

    // Load Q, K, V values from input (4 elements each)
    const float4 q_vec = qkv_proj[q_offset];
    const float4 k_vec = qkv_proj[k_offset];
    const float4 v_vec = qkv_proj[v_offset];

    // Load RoPE values (4 elements: 2 cos/sin pairs)
    const uint pos = start_pos + seq_idx;
    const float4 cos_vec = cos_table[pos * head_dim_div4 + vec4_idx];
    const float4 sin_vec = sin_table[pos * head_dim_div4 + vec4_idx];

    // Apply RoPE rotation to Q
    // q_vec contains [x0, x1, x2, x3] representing 2 pairs: (x0,x1) and (x2,x3)
    // cos_vec contains [c0, c0, c1, c1] - cos values repeated for each pair
    // sin_vec contains [s0, s0, s1, s1] - sin values repeated for each pair
    // But actually, the RoPE formula applies per pair:
    //   pair0: (x0, x1) -> (x0*c0 - x1*s0, x0*s0 + x1*c0)
    //   pair1: (x2, x3) -> (x2*c1 - x3*s1, x2*s1 + x3*c1)

    // Extract pairs
    const float q0 = q_vec.x;
    const float q1 = q_vec.y;
    const float q2 = q_vec.z;
    const float q3 = q_vec.w;

    const float k0 = k_vec.x;
    const float k1 = k_vec.y;
    const float k2 = k_vec.z;
    const float k3 = k_vec.w;

    // cos_vec and sin_vec contain values for 2 dimension pairs
    // Layout: [cos_pair0, cos_pair1] or [sin_pair0, sin_pair1]
    const float c0 = cos_vec.x;
    const float c1 = cos_vec.z;
    const float s0 = sin_vec.x;
    const float s1 = sin_vec.z;

    // Apply RoPE to Q
    float4 q_rotated;
    q_rotated.x = q0 * c0 - q1 * s0;
    q_rotated.y = q0 * s0 + q1 * c0;
    q_rotated.z = q2 * c1 - q3 * s1;
    q_rotated.w = q2 * s1 + q3 * c1;

    // Apply RoPE to K
    float4 k_rotated;
    k_rotated.x = k0 * c0 - k1 * s0;
    k_rotated.y = k0 * s0 + k1 * c0;
    k_rotated.z = k2 * c1 - k3 * s1;
    k_rotated.w = k2 * s1 + k3 * c1;

    // Write outputs
    q_out[out_offset] = q_rotated;
    k_out[out_offset] = k_rotated;
    v_out[out_offset] = v_vec;  // V is not rotated
}

//
// FP16 Vectorized (half4) Fused QKV Split + Reshape + RoPE Kernel
//
// Optimized version that processes 2 dimension pairs (4 elements) at a time using half4 loads.
// Rotations are computed in FP32 for numerical stability.
// Requires head_dim to be divisible by 4.
//
kernel void fused_qkv_rope_fp16_vec4(
    device const half4* qkv_proj [[buffer(0)]],     // [batch, seq_len, 3*num_heads*head_dim] as half4
    device const half4* cos_table [[buffer(1)]],    // [max_seq_len, head_dim/4] as half4
    device const half4* sin_table [[buffer(2)]],    // [max_seq_len, head_dim/4] as half4
    device half4* q_out [[buffer(3)]],              // [batch, num_heads, seq_len, head_dim] as half4
    device half4* k_out [[buffer(4)]],              // [batch, num_heads, seq_len, head_dim] as half4
    device half4* v_out [[buffer(5)]],              // [batch, num_heads, seq_len, head_dim] as half4
    constant uint& batch [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& num_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& start_pos [[buffer(10)]],
    uint3 tid [[thread_position_in_grid]])
{
    const uint vec4_idx = tid.x;    // half4 index [0, head_dim/4)
    const uint seq_idx = tid.y;     // sequence position [0, seq_len)
    const uint bh_idx = tid.z;      // flattened batch-head index [0, batch*num_heads)

    const uint head_dim_div4 = head_dim / 4;

    // Bounds check
    if (vec4_idx >= head_dim_div4 || seq_idx >= seq_len || bh_idx >= batch * num_heads)
        return;

    // Decompose flattened batch-head index
    const uint b = bh_idx / num_heads;
    const uint h = bh_idx % num_heads;

    // Input layout: [batch, seq_len, 3 * num_heads * head_dim]
    const uint qkv_dim_div4 = (3 * num_heads * head_dim) / 4;
    const uint qkv_base = b * seq_len * qkv_dim_div4 + seq_idx * qkv_dim_div4;

    // Offsets for Q, K, V within the combined projection (in half4 units)
    const uint q_offset = qkv_base + h * head_dim_div4 + vec4_idx;
    const uint k_offset = qkv_base + num_heads * head_dim_div4 + h * head_dim_div4 + vec4_idx;
    const uint v_offset = qkv_base + 2 * num_heads * head_dim_div4 + h * head_dim_div4 + vec4_idx;

    // Output layout: [batch, num_heads, seq_len, head_dim] (in half4 units)
    const uint out_offset = bh_idx * seq_len * head_dim_div4 + seq_idx * head_dim_div4 + vec4_idx;

    // Load Q, K, V values from input (4 elements each)
    const half4 q_vec = qkv_proj[q_offset];
    const half4 k_vec = qkv_proj[k_offset];
    const half4 v_vec = qkv_proj[v_offset];

    // Load RoPE values and promote to FP32
    const uint pos = start_pos + seq_idx;
    const half4 cos_vec_h = cos_table[pos * head_dim_div4 + vec4_idx];
    const half4 sin_vec_h = sin_table[pos * head_dim_div4 + vec4_idx];

    // Extract and promote to FP32 for computation
    const float q0 = float(q_vec.x);
    const float q1 = float(q_vec.y);
    const float q2 = float(q_vec.z);
    const float q3 = float(q_vec.w);

    const float k0 = float(k_vec.x);
    const float k1 = float(k_vec.y);
    const float k2 = float(k_vec.z);
    const float k3 = float(k_vec.w);

    const float c0 = float(cos_vec_h.x);
    const float c1 = float(cos_vec_h.z);
    const float s0 = float(sin_vec_h.x);
    const float s1 = float(sin_vec_h.z);

    // Apply RoPE to Q (compute in FP32)
    half4 q_rotated;
    q_rotated.x = half(q0 * c0 - q1 * s0);
    q_rotated.y = half(q0 * s0 + q1 * c0);
    q_rotated.z = half(q2 * c1 - q3 * s1);
    q_rotated.w = half(q2 * s1 + q3 * c1);

    // Apply RoPE to K (compute in FP32)
    half4 k_rotated;
    k_rotated.x = half(k0 * c0 - k1 * s0);
    k_rotated.y = half(k0 * s0 + k1 * c0);
    k_rotated.z = half(k2 * c1 - k3 * s1);
    k_rotated.w = half(k2 * s1 + k3 * c1);

    // Write outputs
    q_out[out_offset] = q_rotated;
    k_out[out_offset] = k_rotated;
    v_out[out_offset] = v_vec;  // V is not rotated
}
