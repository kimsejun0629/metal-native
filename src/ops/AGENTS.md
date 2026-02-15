<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# ops
## Purpose
Implementation files for high-level neural network operations. Implements matmul, attention, convolution, element-wise ops, reductions, softmax, normalization, pooling, embedding, loss, AMP, op registry, and speculative decoding.
## Key Files
| File | Description |
|------|-------------|
| CMakeLists.txt | Build configuration for ops module |
| matmul.mm | Matrix multiplication implementation (routes to MPSGraph for large, MSL for small) |
| attention.mm | FlashAttention implementation (tiled threadgroup memory, online softmax) |
| conv.mm | Convolution implementation (2D via MPSGraph, 1D/3D stubbed) |
| elementwise.mm | Element-wise ops (broadcasting with MPSGraph, custom kernels for simple cases) |
| reduction.mm | Reduction ops (multi-stage parallel reduction with SIMD shuffles) |
| softmax.mm | Softmax implementation (three-pass: max, exp-sum, normalize) |
| normalization.mm | Normalization ops (Welford's algorithm for LayerNorm/BatchNorm) |
| pooling.mm | Pooling ops (MPSGraph pooling operations) |
| embedding.mm | Embedding table lookup and RoPE implementation |
| loss.mm | Loss functions (cross-entropy with fused log-softmax, MSE, L1) |
| mixed_precision.cpp | AMPController implementation (dtype policy table, cast helpers) |
| op_registry.cpp | OpRegistry implementation (dispatch table, forward/backward functions) |
| batch_advisor.mm | BatchSizeAdvisor implementation (calculates based on memory footprint) |
| speculative.mm | Speculative decoding verification kernel |
| speculative_buffers.mm | Pre-allocated buffer pool for speculative decoding |
## For AI Agents
### Working In This Directory
- Most `.mm` files use MPSGraph for high-level operations
- **matmul**: Threshold at M,N,K >= 128 to decide MPSGraph vs direct MSL
- **attention**: Custom Metal kernel with threadgroup memory tiling (FlashAttention algorithm)
- **softmax**: Uses SIMD-group reductions (`simd_max`, `simd_sum`) for efficiency
- **normalization**: Welford's algorithm computes mean/variance in single pass
- **mixed_precision.cpp**: Pure C++ policy table (no Metal dependencies)
### Common Patterns
```objc
// MPSGraph usage
MPSGraph* graph = [[MPSGraph alloc] init];
MPSGraphTensor* a = [graph placeholderWithShape:shape dataType:dtype name:@"a"];
MPSGraphTensor* b = [graph placeholderWithShape:shape dataType:dtype name:@"b"];
MPSGraphTensor* c = [graph matrixMultiplicationWithPrimaryTensor:a
                                                 secondaryTensor:b
                                                            name:@"matmul"];
MPSGraphExecutable* exec = [graph compileWithDevice:device feeds:feeds];
```
## Dependencies
### Internal
- ../include/metal_native/ops/ (public headers)
- core/ (MNTensor, MNDevice, MNDType, MNShape)
- memory/ (KVCache)
- kernels/ (KernelRegistry, kernel_launch)
- graph/ (GraphBuilder)
### External
- Metal framework
- MetalPerformanceShadersGraph framework
<!-- MANUAL: -->
