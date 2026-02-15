<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# ops
## Purpose
High-level neural network operations. Provides matmul, attention, convolution, element-wise ops, reductions, softmax, normalization, pooling, embedding, loss functions, mixed precision control, operator registry, batch size advisor, and speculative decoding.
## Key Files
| File | Description |
|------|-------------|
| matmul.h | Matrix multiplication (standard and batched) with transpose flags |
| attention.h | FlashAttention with KV cache support and fused QKV+RoPE |
| conv.h | 1D/2D/3D convolution (2D implemented, others stubbed) |
| elementwise.h | Binary ops (add/sub/mul/div), unary math (exp/log/sqrt/abs/neg), conditional ops |
| reduction.h | Sum, mean, max, min, argmax, argmin with keepdim support |
| softmax.h | Numerically stable softmax and log-softmax |
| normalization.h | LayerNorm, RMSNorm, BatchNorm, GroupNorm with Welford's algorithm |
| pooling.h | Max/avg/adaptive pooling operations |
| embedding.h | Embedding table lookup and RoPE (Rotary Position Embedding) |
| loss.h | Cross-entropy, MSE, L1 loss with reduction modes |
| mixed_precision.h | AMPController: automatic FP16/FP32 dtype selection for ops |
| op_registry.h | OpRegistry: global dispatch table for all operators |
| batch_advisor.h | BatchSizeAdvisor: recommend batch size based on memory |
| speculative.h | Speculative decoding verification kernel |
| speculative_buffers.h | Pre-allocated buffer pool for speculative decoding |
## For AI Agents
### Working In This Directory
- **matmul** routes large matrices (M,N,K >= 128) to MPSGraph, small matrices to MSL kernels
- **attention** implements FlashAttention with online softmax and tiled threadgroup memory
- **softmax** uses three-pass algorithm (max, exp-sum, normalize) with FP32 accumulation for FP16
- **normalization** uses Welford's algorithm for stable mean/variance
- **AMPController** automatically selects FP16 for compute-bound ops, FP32 for precision-sensitive ops
- **OpRegistry** enables dynamic dispatch and extensibility
### Common Patterns
```cpp
// Matrix multiplication
MNTensor a = ..., b = ...;
MNTensor c = matmul(a, b, /*transpose_a=*/false, /*transpose_b=*/false);

// FlashAttention
MNTensor q = ..., k = ..., v = ...;
float scale = 1.0f / std::sqrt(head_dim);
MNTensor out = flash_attention(q, k, v, /*mask=*/nullptr, scale);

// Mixed precision
AMPController& amp = AMPController::instance();
amp.set_enabled(true);
MNDType dtype = amp.get_compute_dtype("matmul");  // returns FP16
MNTensor a_fp16 = amp.cast_if_needed(a, dtype);

// Batch size recommendation
BatchSizeRecommendation rec = BatchSizeAdvisor::recommend(
    model_bytes, seq_len, hidden_dim, MNDType::Float16);
size_t batch = rec.recommended_batch;
```
## Dependencies
### Internal
- core/ (MNTensor, MNDevice, MNDType, MNShape)
- memory/ (KVCache)
- kernels/ (KernelRegistry, kernel_launch)
- graph/ (GraphBuilder for MPSGraph operations)
### External
- Metal framework
- MetalPerformanceShadersGraph framework
<!-- MANUAL: -->
