#pragma once

/// @file embedding.h
/// @brief Embedding and positional encoding operations.
///
/// Provides table lookup embedding and Rotary Position Embedding (RoPE)
/// for transformer models. All operations are GPU-accelerated using Metal.

#include <cstddef>
#include <memory>

#include "metal_native/core/dtype.h"
#include "metal_native/core/shape.h"

namespace metal_native {

class MNTensor;
class MNDevice;

/// Embedding table lookup operation.
///
/// Given a tensor of integer indices and a weight table, returns embeddings
/// by looking up each index in the table.
///
/// @param indices  Integer tensor of shape [batch, seq_len] or [seq_len].
/// @param weight   Embedding table of shape [vocab_size, embed_dim].
/// @param device   Device to perform the operation on.
/// @return         Embedded tensor of shape [..., embed_dim].
/// @throws         MNException if indices are out of bounds or shapes mismatch.
MNTensor embedding(const MNTensor& indices,
                   const MNTensor& weight,
                   MNDevice& device);

/// Rotary Position Embedding (RoPE) operation.
///
/// Applies rotary position embeddings to the input tensor using precomputed
/// sin/cos tables. This is a key component in modern transformer architectures
/// like LLaMA and GPT-NeoX.
///
/// @param input         Input tensor of shape [batch, seq_len, num_heads, head_dim].
/// @param cos_cache     Cosine cache of shape [max_seq_len, head_dim].
/// @param sin_cache     Sine cache of shape [max_seq_len, head_dim].
/// @param position_ids  Position indices of shape [batch, seq_len].
/// @param device        Device to perform the operation on.
/// @return              Tensor with RoPE applied, same shape as input.
/// @throws              MNException on shape or dtype mismatch.
MNTensor rope_embedding(const MNTensor& input,
                        const MNTensor& cos_cache,
                        const MNTensor& sin_cache,
                        const MNTensor& position_ids,
                        MNDevice& device);

} // namespace metal_native
