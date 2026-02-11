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

// Forward declarations
namespace metal_native {
class MNDevice;
}

// Need full definition for QKVResult struct
#include "metal_native/core/tensor.h"

namespace metal_native {

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

/// Fused QKV Projection: compute Q, K, V from input in single matmul.
/// Instead of 3 separate matmuls: Q=X@Wq, K=X@Wk, V=X@Wv
/// Performs: QKV = X @ W_qkv (where W_qkv = concat(Wq, Wk, Wv))
/// Then splits into Q, K, V tensors.
///
/// @param input    Input tensor [batch, seq_len, hidden_dim].
/// @param w_qkv    Concatenated weight matrix [hidden_dim, 3*head_dim*num_heads].
/// @param num_heads Number of attention heads.
/// @param head_dim  Dimension per head.
/// @return          Tuple-like struct with Q, K, V tensors.
struct QKVResult {
    MNTensor q;  ///< [batch, num_heads, seq_len, head_dim]
    MNTensor k;  ///< [batch, num_heads, seq_len, head_dim]
    MNTensor v;  ///< [batch, num_heads, seq_len, head_dim]
};

QKVResult fused_qkv_projection(const MNTensor& input,
                                const MNTensor& w_qkv,
                                int64_t num_heads,
                                int64_t head_dim);

/// Fused SiLU Gate: SiLU(gate) * up (Llama/Mistral FFN pattern)
/// @param gate  Gate tensor.
/// @param up    Up-projection tensor.
/// @return      SiLU(gate) * up.
MNTensor swiglu(const MNTensor& gate, const MNTensor& up);

/// Fused Bias + GELU activation: GELU(input + bias)
/// Typically used after MatMul: result = GELU(MatMul(X, W) + bias)
/// @param input  Input tensor from matmul [..., hidden_size].
/// @param bias   Bias vector [hidden_size].
/// @return       GELU-activated tensor.
MNTensor fused_bias_gelu(const MNTensor& input, const MNTensor& bias);

/// Fused Residual Addition + RMSNorm: rms_norm(input + residual) * weight
/// Common in Llama/Mistral transformer blocks.
/// @param input     Current layer output.
/// @param residual  Residual connection tensor (same shape as input).
/// @param weight    RMSNorm weight [hidden_size].
/// @param eps       Numerical stability constant.
/// @return          Normalized tensor.
MNTensor fused_residual_norm(const MNTensor& input, const MNTensor& residual,
                              const MNTensor& weight, float eps = 1e-6f);

} // namespace fast
} // namespace metal_native
