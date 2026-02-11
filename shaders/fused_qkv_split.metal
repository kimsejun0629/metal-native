#include <metal_stdlib>
using namespace metal;

/// Split and reshape QKV from [batch, seq_len, 3*num_heads*head_dim]
/// to Q, K, V each [batch, num_heads, seq_len, head_dim]
///
/// The combined tensor layout is:
///   For each (batch, seq_pos): [Q_h0, Q_h1, ..., K_h0, K_h1, ..., V_h0, V_h1, ...]
///   Each Q_hi/K_hi/V_hi has head_dim elements

kernel void qkv_split_reshape_fp32(
    device const float* qkv_combined  [[buffer(0)]],  // [batch, seq, 3*num_heads*head_dim]
    device float*       q_out         [[buffer(1)]],   // [batch, num_heads, seq, head_dim]
    device float*       k_out         [[buffer(2)]],   // [batch, num_heads, seq, head_dim]
    device float*       v_out         [[buffer(3)]],   // [batch, num_heads, seq, head_dim]
    constant uint&      batch_size    [[buffer(4)]],
    constant uint&      seq_len       [[buffer(5)]],
    constant uint&      num_heads     [[buffer(6)]],
    constant uint&      head_dim      [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]])
{
    // tid.x = element within head_dim, tid.y = seq position, tid.z = batch * num_heads
    uint d = tid.x;          // head_dim index
    uint s = tid.y;          // seq_len index
    uint bh = tid.z;         // batch * num_heads

    uint b = bh / num_heads;
    uint h = bh % num_heads;

    if (b >= batch_size || s >= seq_len || d >= head_dim) return;

    uint qkv_stride = 3 * num_heads * head_dim;

    // Source index in combined tensor [batch, seq, 3*num_heads*head_dim]
    uint src_base = b * seq_len * qkv_stride + s * qkv_stride;
    uint q_src = src_base + h * head_dim + d;                              // Q section
    uint k_src = src_base + num_heads * head_dim + h * head_dim + d;       // K section
    uint v_src = src_base + 2 * num_heads * head_dim + h * head_dim + d;   // V section

    // Destination index [batch, num_heads, seq, head_dim]
    uint dst_idx = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim + d;

    q_out[dst_idx] = qkv_combined[q_src];
    k_out[dst_idx] = qkv_combined[k_src];
    v_out[dst_idx] = qkv_combined[v_src];
}

kernel void qkv_split_reshape_fp16(
    device const half* qkv_combined  [[buffer(0)]],
    device half*       q_out         [[buffer(1)]],
    device half*       k_out         [[buffer(2)]],
    device half*       v_out         [[buffer(3)]],
    constant uint&     batch_size    [[buffer(4)]],
    constant uint&     seq_len       [[buffer(5)]],
    constant uint&     num_heads     [[buffer(6)]],
    constant uint&     head_dim      [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]])
{
    uint d = tid.x;
    uint s = tid.y;
    uint bh = tid.z;

    uint b = bh / num_heads;
    uint h = bh % num_heads;

    if (b >= batch_size || s >= seq_len || d >= head_dim) return;

    uint qkv_stride = 3 * num_heads * head_dim;
    uint src_base = b * seq_len * qkv_stride + s * qkv_stride;
    uint q_src = src_base + h * head_dim + d;
    uint k_src = src_base + num_heads * head_dim + h * head_dim + d;
    uint v_src = src_base + 2 * num_heads * head_dim + h * head_dim + d;

    uint dst_idx = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim + d;

    q_out[dst_idx] = qkv_combined[q_src];
    k_out[dst_idx] = qkv_combined[k_src];
    v_out[dst_idx] = qkv_combined[v_src];
}
