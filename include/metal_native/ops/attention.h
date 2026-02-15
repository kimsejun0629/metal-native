#pragma once

/// @file attention.h
/// @brief FlashAttention implementation for Apple Silicon GPUs.
///
/// Provides an optimized scaled dot-product attention implementation using
/// tiled computation in threadgroup memory with online softmax for numerical
/// stability. Supports causal masking and variable sequence lengths.

#include <cstddef>
#include <memory>

namespace metal_native {

class MNTensor;
class KVCache;

/// FlashAttention: memory-efficient scaled dot-product attention.
///
/// Computes: softmax(Q @ K^T / scale) @ V
///
/// Uses tiled computation in threadgroup memory to minimize global memory
/// traffic. Online softmax ensures numerical stability in a single pass.
/// Optimized for Apple Silicon GPU architecture.
///
/// Supports Grouped Query Attention (GQA), Multi-Query Attention (MQA),
/// and standard Multi-Head Attention (MHA):
/// - MHA: num_heads == num_kv_heads (e.g., 32 Q heads, 32 KV heads)
/// - GQA: num_heads > num_kv_heads, multiple Q heads share KV heads
///        (e.g., Llama 3: 32 Q heads, 8 KV heads, group_ratio=4)
/// - MQA: num_kv_heads == 1, all Q heads share single KV head
///        (e.g., Falcon: 32 Q heads, 1 KV head, group_ratio=32)
///
/// GQA reduces KV cache size by group_ratio (4x for Llama 3, 8x for Mistral).
///
/// @param query        Query tensor: [batch, num_heads, seq_len_q, head_dim]
/// @param key          Key tensor: [batch, num_kv_heads, seq_len_k, head_dim]
/// @param value        Value tensor: [batch, num_kv_heads, seq_len_v, head_dim]
/// @param mask         Optional attention mask: [batch, 1, seq_len_q, seq_len_k]
///                     or nullptr. Use for causal masking (upper-triangular).
/// @param scale        Scale factor (typically 1.0 / sqrt(head_dim)).
/// @return             Output tensor: [batch, num_heads, seq_len_q, head_dim]
///
/// @throws MNException(InvalidArgument) if shapes are incompatible,
///         if seq_len_k != seq_len_v, if num_heads % num_kv_heads != 0,
///         or if GQA is used with Float32 (currently FP16 only).
MNTensor flash_attention(const MNTensor& query,
                         const MNTensor& key,
                         const MNTensor& value,
                         const MNTensor* mask,
                         float scale);

/// FlashAttention with pre-allocated KV cache (autoregressive mode)
///
/// Uses cached key/value tensors from previous tokens for efficient
/// autoregressive generation. The caller should append new K/V to the
/// cache before calling this function.
///
/// @param query        Query tensor for current token: [1, num_heads, 1, head_dim]
/// @param kv_cache     Pre-allocated KV cache containing all previous K/V pairs
/// @param layer        Layer index to read from cache
/// @param position     Current token position (used for updating cache)
/// @param mask         Optional attention mask
/// @param scale        Scale factor (typically 1.0 / sqrt(head_dim))
/// @return             Output tensor: [1, num_heads, 1, head_dim]
MNTensor flash_attention_with_kv_cache(
    const MNTensor& query,
    KVCache& kv_cache,
    size_t layer,
    size_t position,
    const MNTensor* mask,
    float scale);

/// Fused QKV Split + Reshape + RoPE (Rotary Position Embedding)
///
/// Takes a combined QKV projection tensor and performs three operations in one pass:
/// 1. Split into separate Q, K, V tensors
/// 2. Reshape from [batch, seq_len, 3*num_heads*head_dim] to [batch, num_heads, seq_len, head_dim]
/// 3. Apply RoPE to Q and K (V is not rotated, just reshaped)
///
/// This fused kernel eliminates intermediate memory traffic and is significantly
/// faster than separate operations.
///
/// @param qkv_proj     Combined QKV projection: [batch, seq_len, 3*num_heads*head_dim]
/// @param cos_table    Precomputed cosine values: [max_seq_len, head_dim/2]
/// @param sin_table    Precomputed sine values: [max_seq_len, head_dim/2]
/// @param q_out        Output Q tensor: [batch, num_heads, seq_len, head_dim] (pre-allocated)
/// @param k_out        Output K tensor: [batch, num_heads, seq_len, head_dim] (pre-allocated)
/// @param v_out        Output V tensor: [batch, num_heads, seq_len, head_dim] (pre-allocated)
/// @param batch        Batch size
/// @param seq_len      Sequence length
/// @param num_heads    Number of attention heads
/// @param head_dim     Dimension per head (must be even)
/// @param start_pos    Starting position for KV cache offset (0 for prefill)
///
/// @throws MNException(InvalidArgument) if shapes are incompatible or head_dim is odd
void fused_qkv_split_rope(
    const MNTensor& qkv_proj,
    const MNTensor& cos_table,
    const MNTensor& sin_table,
    MNTensor& q_out,
    MNTensor& k_out,
    MNTensor& v_out,
    int64_t batch,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t start_pos);

/// Enable or disable texture-backed attention (experimental)
///
/// When enabled, FlashAttention uses Metal texture2d for the attention score
/// matrix (QK^T) instead of threadgroup memory, leveraging Apple Silicon's
/// texture cache hardware. Currently FP32 MHA only.
///
/// @param enable  true to enable texture attention, false for standard path
void set_texture_attention(bool enable);

/// Check if texture-backed attention is enabled
///
/// @return true if texture attention is enabled, false otherwise
bool texture_attention_enabled();

} // namespace metal_native
