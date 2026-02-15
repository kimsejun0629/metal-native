<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# future
## Purpose
Forward-compatibility stubs for upcoming hardware and API features. Includes BFloat16 support (M3+), high-performance fused ops (MLX-inspired), Metal 4 API stubs, and Neural Engine integration stubs.
## Key Files
| File | Description |
|------|-------------|
| bfloat16.h | BFloat16 support detection and AMP policy configuration (M3+ GPUs) |
| fast_ops.h | High-performance fused kernels (RMSNorm, RoPE, FlashAttention, SwiGLU, etc.) |
| metal4.h | Metal 4 API stubs (MTLTensor, ML encoder, argument tables - macOS 27+) |
| neural_engine.h | Apple Neural Engine integration stubs (future M5 chips) |
## For AI Agents
### Working In This Directory
- **bfloat16.h**: Check `bfloat16_supported()` before using BF16; configure policy via `set_bf16_policy()`
- **fast_ops.h**: MLX-inspired fused kernels achieve 10-50x speedup vs unfused - use these when available
- **metal4.h**: Forward-compatible stubs for Metal 4 features (currently returns false/unavailable)
- **neural_engine.h**: Stubs for future dedicated NN accelerator access
### Common Patterns
```cpp
// BFloat16
if (metal_native::future::bfloat16_supported()) {
    metal_native::future::set_bf16_policy(BF16Policy::Full);
}

// Fast fused ops (metal_native::fast namespace)
using namespace metal_native::fast;
MNTensor normed = rms_norm(input, weight, 1e-6f);
MNTensor attn = scaled_dot_product_attention(q, k, v, scale, /*causal=*/true);
MNTensor swiglu_out = swiglu(gate, up);

// Metal 4 (future)
if (metal_native::future::metal4_available()) {
    auto caps = metal_native::future::query_metal4_capabilities();
    if (caps.has_native_tensors) { /* use MTLTensor */ }
}

// Neural Engine (future)
if (metal_native::future::neural_engine_available()) {
    // offload matmul to dedicated accelerator
}
```
## Dependencies
### Internal
- core/ (MNTensor, MNDevice, MNDType)
### External
- Metal framework (feature detection APIs)
<!-- MANUAL: -->
