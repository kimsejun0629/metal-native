<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# future
## Purpose
Implementation files for forward-compatibility features. Implements BFloat16 detection, high-performance fused ops, and stubs for Metal 4 / Neural Engine.
## Key Files
| File | Description |
|------|-------------|
| CMakeLists.txt | Build configuration for future module |
| future_stubs.mm | Unified implementation for bfloat16, fast_ops, metal4, neural_engine stubs |
## For AI Agents
### Working In This Directory
- **future_stubs.mm**: Single file containing all future feature implementations and stubs
- BFloat16 detection: queries MTLDevice `supportsFamily:MTLGPUFamilyApple9`
- Fast ops: hand-tuned Metal kernels for common fusion patterns (RMSNorm, RoPE, FlashAttention, SwiGLU)
- Metal 4 stubs: always return false/unavailable (forward-compatibility for macOS 27+)
- Neural Engine stubs: always return false/unavailable (forward-compatibility for future chips)
### Common Patterns
```objc
// BFloat16 detection
bool bfloat16_supported() {
    id<MTLDevice> device = MNDevice::instance().metal_device();
    if (@available(macOS 14.0, *)) {
        return [device supportsFamily:MTLGPUFamilyApple9];
    }
    return false;
}

// Fast op implementation (example: RMSNorm)
// Uses custom Metal kernel with SIMD-group reductions
id<MTLComputePipelineState> pipeline = get_pipeline("fast_rms_norm");
// ... encode kernel with optimized threadgroup config ...
```
## Dependencies
### Internal
- ../include/metal_native/future/ (public headers)
- core/ (MNTensor, MNDevice, MNDType)
- kernels/ (KernelRegistry for fast ops)
### External
- Metal framework (GPU family queries, kernel execution)
<!-- MANUAL: -->
