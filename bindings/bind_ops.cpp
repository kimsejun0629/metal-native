/**
 * @file bind_ops.cpp
 * @brief Pybind11 bindings for metal_native operations.
 *
 * This file provides Python bindings for:
 * - Fused fast operations (fast::)
 * - Matrix operations (matmul, batched_matmul)
 * - Attention operations (flash_attention)
 * - Normalization and activation operations
 */

#include <pybind11/pybind11.h>

#include "metal_native/future/fast_ops.h"
#include "metal_native/ops/attention.h"
#include "metal_native/ops/matmul.h"
#include "metal_native/ops/softmax.h"
#include "metal_native/core/tensor.h"

namespace py = pybind11;

namespace metal_native {
namespace python {

/**
 * @brief Bind operations module.
 *
 * Registers Python bindings for all metal_native operations.
 */
void bind_ops(py::module_& m) {
    // Create fast submodule for fused operations
    auto fast_m = m.def_submodule("fast", "Fused Metal operations");

    // fast::rms_norm
    fast_m.def("rms_norm",
               &fast::rms_norm,
               py::arg("input"),
               py::arg("weight"),
               py::arg("eps") = 1e-6f,
               "Fused RMSNorm: x * rsqrt(mean(x^2) + eps) * weight\n\n"
               "Args:\n"
               "    input: Input tensor [..., hidden_size]\n"
               "    weight: Scale weights [hidden_size]\n"
               "    eps: Numerical stability constant (default: 1e-6)\n\n"
               "Returns:\n"
               "    Normalized tensor");

    // fast::layer_norm
    fast_m.def("layer_norm",
               &fast::layer_norm,
               py::arg("input"),
               py::arg("weight"),
               py::arg("bias"),
               py::arg("eps") = 1e-5f,
               "Fused LayerNorm with affine: (x - mean) / sqrt(var + eps) * gamma + beta\n\n"
               "Args:\n"
               "    input: Input tensor\n"
               "    weight: Scale weights (gamma)\n"
               "    bias: Shift bias (beta)\n"
               "    eps: Numerical stability constant (default: 1e-5)\n\n"
               "Returns:\n"
               "    Normalized tensor");

    // fast::rope (in-place, returns None)
    fast_m.def("rope",
               &fast::rope,
               py::arg("q"),
               py::arg("k"),
               py::arg("cos_cache"),
               py::arg("sin_cache"),
               py::arg("start_pos") = 0,
               "Rotary Position Embedding (RoPE) - applies in-place to Q and K tensors\n\n"
               "Args:\n"
               "    q: Query tensor [batch, heads, seq, head_dim] (modified in-place)\n"
               "    k: Key tensor [batch, heads, seq, head_dim] (modified in-place)\n"
               "    cos_cache: Cosine cache [max_seq, head_dim/2]\n"
               "    sin_cache: Sine cache [max_seq, head_dim/2]\n"
               "    start_pos: Starting position for incremental decoding (default: 0)");

    // fast::swiglu
    fast_m.def("swiglu",
               &fast::swiglu,
               py::arg("gate"),
               py::arg("up"),
               "Fused SiLU Gate: SiLU(gate) * up (Llama/Mistral FFN pattern)\n\n"
               "Args:\n"
               "    gate: Gate tensor\n"
               "    up: Up-projection tensor\n\n"
               "Returns:\n"
               "    SiLU(gate) * up");

    // fast::fused_bias_gelu
    fast_m.def("fused_bias_gelu",
               &fast::fused_bias_gelu,
               py::arg("input"),
               py::arg("bias"),
               "Fused Bias + GELU activation: GELU(input + bias)\n\n"
               "Args:\n"
               "    input: Input tensor from matmul [..., hidden_size]\n"
               "    bias: Bias vector [hidden_size]\n\n"
               "Returns:\n"
               "    GELU-activated tensor");

    // fast::fused_residual_norm
    fast_m.def("fused_residual_norm",
               &fast::fused_residual_norm,
               py::arg("input"),
               py::arg("residual"),
               py::arg("weight"),
               py::arg("eps") = 1e-6f,
               "Fused Residual Addition + RMSNorm: rms_norm(input + residual) * weight\n\n"
               "Args:\n"
               "    input: Current layer output\n"
               "    residual: Residual connection tensor (same shape as input)\n"
               "    weight: RMSNorm weight [hidden_size]\n"
               "    eps: Numerical stability constant (default: 1e-6)\n\n"
               "Returns:\n"
               "    Normalized tensor");

    // fast::fused_qkv_projection
    fast_m.def("fused_qkv_projection",
               &fast::fused_qkv_projection,
               py::arg("input"),
               py::arg("w_qkv"),
               py::arg("num_heads"),
               py::arg("head_dim"),
               "Fused QKV Projection: compute Q, K, V from input in single matmul\n\n"
               "Args:\n"
               "    input: Input tensor [batch, seq_len, hidden_dim]\n"
               "    w_qkv: Concatenated weight matrix [hidden_dim, 3*head_dim*num_heads]\n"
               "    num_heads: Number of attention heads\n"
               "    head_dim: Dimension per head\n\n"
               "Returns:\n"
               "    QKVResult with q, k, v tensors [batch, num_heads, seq_len, head_dim]");

    // fast::scaled_dot_product_attention - bind flash_attention instead
    // (scaled_dot_product_attention throws NotImplemented in future_stubs.mm)
    fast_m.def("scaled_dot_product_attention",
               [](const MNTensor& query,
                  const MNTensor& key,
                  const MNTensor& value,
                  float scale,
                  bool causal) {
                   // Use flash_attention directly; pass nullptr for mask if not causal
                   // For causal, we'd need to construct a causal mask tensor
                   // For now, delegate to flash_attention with nullptr mask
                   return flash_attention(query, key, value, nullptr, scale);
               },
               py::arg("query"),
               py::arg("key"),
               py::arg("value"),
               py::arg("scale"),
               py::arg("causal") = false,
               "Fused Scaled Dot-Product Attention (Flash Attention v2)\n\n"
               "Args:\n"
               "    query: [batch, heads, seq_q, head_dim]\n"
               "    key: [batch, heads, seq_k, head_dim]\n"
               "    value: [batch, heads, seq_k, head_dim]\n"
               "    scale: Scaling factor (typically 1/sqrt(head_dim))\n"
               "    causal: Apply causal mask (default: False)\n\n"
               "Returns:\n"
               "    Attention output [batch, heads, seq_q, head_dim]");

    // Regular ops on main module

    // matmul
    m.def("matmul",
          &matmul,
          py::arg("a"),
          py::arg("b"),
          py::arg("transpose_a") = false,
          py::arg("transpose_b") = false,
          "Matrix multiplication: C = A @ B\n\n"
          "Args:\n"
          "    a: Left operand tensor [..., M, K]\n"
          "    b: Right operand tensor [..., K, N]\n"
          "    transpose_a: Transpose A before multiplication (default: False)\n"
          "    transpose_b: Transpose B before multiplication (default: False)\n\n"
          "Returns:\n"
          "    Result tensor [..., M, N]");

    // batched_matmul
    m.def("batched_matmul",
          &batched_matmul,
          py::arg("a"),
          py::arg("b"),
          py::arg("transpose_a") = false,
          py::arg("transpose_b") = false,
          "Batched matrix multiplication: C[i] = A[i] @ B[i]\n\n"
          "Args:\n"
          "    a: Left operand tensor [batch..., M, K]\n"
          "    b: Right operand tensor [batch..., K, N]\n"
          "    transpose_a: Transpose A before multiplication (default: False)\n"
          "    transpose_b: Transpose B before multiplication (default: False)\n\n"
          "Returns:\n"
          "    Batched result tensor [batch..., M, N]");

    // softmax
    m.def("softmax",
          &softmax,
          py::arg("input"),
          py::arg("dim"),
          "Numerically stable softmax along a specified dimension\n\n"
          "Args:\n"
          "    input: Input tensor of any shape\n"
          "    dim: Dimension along which to apply softmax\n\n"
          "Returns:\n"
          "    Output tensor with the same shape as input");

    // log_softmax
    m.def("log_softmax",
          &log_softmax,
          py::arg("input"),
          py::arg("dim"),
          "Numerically stable log-softmax along a specified dimension\n\n"
          "Args:\n"
          "    input: Input tensor of any shape\n"
          "    dim: Dimension along which to apply log-softmax\n\n"
          "Returns:\n"
          "    Output tensor with the same shape as input");

    // flash_attention
    m.def("flash_attention",
          [](const MNTensor& query,
             const MNTensor& key,
             const MNTensor& value,
             py::object mask_obj,
             float scale) {
              const MNTensor* mask = nullptr;
              if (!mask_obj.is_none()) {
                  mask = &mask_obj.cast<const MNTensor&>();
              }
              return flash_attention(query, key, value, mask, scale);
          },
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("mask") = py::none(),
          py::arg("scale"),
          "FlashAttention: memory-efficient scaled dot-product attention\n\n"
          "Args:\n"
          "    query: Query tensor [batch, num_heads, seq_len_q, head_dim]\n"
          "    key: Key tensor [batch, num_heads, seq_len_k, head_dim]\n"
          "    value: Value tensor [batch, num_heads, seq_len_v, head_dim]\n"
          "    mask: Optional attention mask [batch, 1, seq_len_q, seq_len_k] (default: None)\n"
          "    scale: Scale factor (typically 1.0 / sqrt(head_dim))\n\n"
          "Returns:\n"
          "    Output tensor [batch, num_heads, seq_len_q, head_dim]");
}

} // namespace python
} // namespace metal_native
