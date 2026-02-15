<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# shaders
## Purpose
Metal Shading Language (MSL) kernel implementations for GPU-accelerated operations. Includes FlashAttention v2 with SIMD matrix ops, fused kernels (matmul+GELU, residual+norm, QKV+RoPE, SwiGLU), normalization, activation, reduction, and utility kernels. Optimized for Apple Silicon with threadgroup memory, SIMD groups, and AMX acceleration.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| common/ | Shared headers: metal_types.h (type aliases), math_utils.h (fast math functions), simd_utils.h (SIMD group operations) |

## Key Files
| File | Description |
|------|-------------|
| attention_kernel.metal | FlashAttention v2 implementation with online softmax. Uses SIMD matrix ops (simdgroup_float8x8) and AMX acceleration. Tile size 16 (configurable), threadgroup memory budget managed carefully (32KB limit). Supports FP32/FP16 with FP32 accumulation. Memory layout documented: Q/K/V tiles, score matrices, output accumulators. |
| fused_matmul_gelu.metal | Fused matrix multiplication + GELU activation. Avoids materializing intermediate matmul result. Uses SIMD for efficient compute, threadgroup memory for tile blocking. |
| fused_norm.metal | Fused normalization kernels: LayerNorm, RMSNorm with affine transforms in single pass. Reduces memory bandwidth vs separate ops. |
| fused_residual_norm.metal | Fused residual addition + RMSNorm. Common transformer pattern: `rms_norm(x + residual)`. Single kernel avoids round-trip to memory. |
| fused_qkv_split.metal | Fused QKV projection split: single matmul producing concatenated QKV, then split into separate Q, K, V tensors with reshaping for multi-head attention. Reduces kernel launches. |
| fused_qkv_rope.metal | Fused QKV split + Rotary Position Embedding (RoPE). Applies RoPE to Q and K in-place after splitting. Optimizes Llama/Mistral attention patterns. |
| fused_rope.metal | Standalone RoPE kernel. Applies rotary embeddings using precomputed cos/sin caches. In-place operation on Q and K tensors. Supports incremental decoding with start_pos offset. |
| fused_swiglu.metal | Fused SwiGLU activation: `SiLU(gate) * up`. Used in Llama/Mistral FFN. Single kernel vs separate SiLU + multiply. |
| normalization_kernel.metal | Standalone normalization: LayerNorm, RMSNorm, GroupNorm, InstanceNorm. Numerically stable with Welford's algorithm for variance. Configurable epsilon. |
| activation_kernels.metal | Activation functions: ReLU, GELU (tanh approx), SiLU (Swish), Softplus, Mish. Vectorized implementations. |
| softmax_kernel.metal | Numerically stable softmax using max subtraction. Log-softmax variant included. Threadgroup reductions for max/sum. |
| reduction_kernel.metal | Reduction operations: sum, mean, max, min along specified dimension. SIMD group reductions followed by threadgroup reductions. FP32 accumulation for stability. |
| elementwise_kernels.metal | Element-wise operations: add, sub, mul, div, pow, exp, log, sqrt, abs. Supports broadcasting. Vectorized with SIMD. |
| copy_kernels.metal | Tensor copy, permute, reshape, slice, concat. Memory transfer optimization with coalesced access patterns. |
| embedding_kernel.metal | Embedding lookup kernel. Indexed access with bounds checking. Supports padding_idx. |
| dequantize_kernel.metal | Dequantization kernels for quantized inference: int8→fp16/fp32, int4→fp16/fp32. Per-tensor or per-channel scaling. |
| speculative_kernel.metal | Speculative decoding kernels for accelerated LLM generation (experimental). Parallel candidate generation and verification. |
| CMakeLists.txt | Shader build configuration using compile_metal_shaders() macro. Compiles all .metal → .air → metal_native.metallib bundle. Includes common/ headers. |

## For AI Agents
### Working In This Directory
- All kernels are Metal Shading Language (MSL 3.0), compiled with `xcrun metal`
- Build: Handled by CMake via ../cmake/MetalShaderCompile.cmake, output: ../build/shaders/metal_native.metallib
- Test changes: `cd ../build && cmake --build . --target metal_native_shaders`
- FlashAttention memory calculations documented in attention_kernel.metal (must fit in 32KB threadgroup limit)
- Use `[[buffer(N)]]` for inputs/outputs, `[[threadgroup(N)]]` for shared memory
- SIMD patterns: `simdgroup_load`, `simdgroup_store`, `simdgroup_matrix_multiply_accumulate`

### Testing Requirements
- Compile test: `cd ../build && cmake --build . --target metal_native_shaders` (should produce .metallib)
- Runtime test: Run any benchmark or test that loads kernels: `cd ../benchmarks && python3 benchmark_matmul.py --sizes 512 --iterations 5`
- Check metallib: `xcrun -sdk macosx metal-dsymutil ../build/shaders/metal_native.metallib` (symbol dump)
- Validate includes: Ensure common/ headers are in INCLUDE_DIRS for compile_metal_shaders()

### Common Patterns
- Kernel signature: `kernel void kernel_name(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], ...)`
- Thread indexing: `uint gid = thread_position_in_grid`, `uint tid = thread_position_in_threadgroup`
- Threadgroup memory: `threadgroup float shared[SIZE];` allocated per-threadgroup
- SIMD groups: 32 threads on Apple Silicon, use `simd_sum()`, `simd_max()` for reductions
- Numerical stability: Subtract max before exp in softmax, use Welford for variance
- Memory alignment: Ensure buffer offsets are 16-byte aligned for SIMD loads

## Dependencies
### Internal
- common/metal_types.h: Type aliases (float32_t, float16_t)
- common/math_utils.h: Fast math (rsqrt, exp2_fast, etc.)
- common/simd_utils.h: SIMD group utilities

### External
- Metal Shading Language 3.0 (macOS 13.0+ / iOS 16.0+)
- Metal framework runtime (MTLDevice, MTLCommandQueue, MTLBuffer)
- xcrun metal compiler (Xcode Command Line Tools)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
