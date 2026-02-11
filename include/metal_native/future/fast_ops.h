#pragma once

/// @file fast_ops.h
/// @brief High-performance fused operation library (MLX-inspired).
///
/// `metal_native::fast` provides optimized fused kernels for common
/// deep learning operations. These bypass the general-purpose op dispatch
/// and execute hand-tuned Metal shaders directly.
///
/// Inspired by MLX's `mlx.core.fast` module which achieves 10-50x speedups
/// over unfused implementations.

#include <cstdint>
#include <vector>

namespace metal_native {

class MNTensor;
class MNDevice;

namespace fast {

/// Fused RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
/// @param input   Input tensor [..., hidden_size].
/// @param weight  Scale weights [hidden_size].
/// @param eps     Numerical stability constant.
/// @return        Normalized tensor.
MNTensor rms_norm(const MNTensor& input, const MNTensor& weight, float eps = 1e-6f);

/// Fused LayerNorm with optional affine: (x - mean) / sqrt(var + eps) * gamma + beta
MNTensor layer_norm(const MNTensor& input,
                    const MNTensor& weight,
                    const MNTensor& bias,
                    float eps = 1e-5f);

/// Rotary Position Embedding (RoPE)
/// Applies rotary embeddings to Q and K tensors in-place.
/// @param q         Query tensor [batch, heads, seq, head_dim].
/// @param k         Key tensor [batch, heads, seq, head_dim].
/// @param cos_cache Cosine cache [max_seq, head_dim/2].
/// @param sin_cache Sine cache [max_seq, head_dim/2].
/// @param start_pos Starting position for incremental decoding.
void rope(MNTensor& q, MNTensor& k,
          const MNTensor& cos_cache, const MNTensor& sin_cache,
          int64_t start_pos = 0);

/// Fused Scaled Dot-Product Attention (Flash Attention v2)
/// @param query  [batch, heads, seq_q, head_dim]
/// @param key    [batch, heads, seq_k, head_dim]
/// @param value  [batch, heads, seq_k, head_dim]
/// @param scale  Scaling factor (typically 1/sqrt(head_dim)).
/// @param causal Apply causal mask.
/// @return       Attention output [batch, heads, seq_q, head_dim].
MNTensor scaled_dot_product_attention(const MNTensor& query,
                                       const MNTensor& key,
                                       const MNTensor& value,
                                       float scale,
                                       bool causal = false);

/// Fused SiLU Gate: SiLU(gate) * up (Llama/Mistral FFN pattern)
/// @param gate  Gate tensor.
/// @param up    Up-projection tensor.
/// @return      SiLU(gate) * up.
MNTensor swiglu(const MNTensor& gate, const MNTensor& up);

} // namespace fast
} // namespace metal_native
