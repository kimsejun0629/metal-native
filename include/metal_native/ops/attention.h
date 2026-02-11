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
/// @param query        Query tensor: [batch, num_heads, seq_len_q, head_dim]
/// @param key          Key tensor: [batch, num_heads, seq_len_k, head_dim]
/// @param value        Value tensor: [batch, num_heads, seq_len_v, head_dim]
/// @param mask         Optional attention mask: [batch, 1, seq_len_q, seq_len_k]
///                     or nullptr. Use for causal masking (upper-triangular).
/// @param scale        Scale factor (typically 1.0 / sqrt(head_dim)).
/// @return             Output tensor: [batch, num_heads, seq_len_q, head_dim]
///
/// @throws MNException(InvalidArgument) if shapes are incompatible or
///         if seq_len_k != seq_len_v.
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

} // namespace metal_native
