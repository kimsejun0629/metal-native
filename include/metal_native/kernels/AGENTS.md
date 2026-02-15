<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# kernels
## Purpose
Kernel compilation and dispatch infrastructure. Provides registry for Metal compute pipelines, LRU cache for compiled states, and launch configuration helpers for optimal threadgroup sizing.
## Key Files
| File | Description |
|------|-------------|
| kernel_registry.h | KernelRegistry: singleton mapping kernel names to MTLComputePipelineState |
| kernel_cache.h | KernelCache: LRU cache for pipeline states with hit-rate tracking |
| kernel_launch.h | LaunchConfig helpers: compute optimal grid/threadgroup sizes for 1D/2D kernels |
## For AI Agents
### Working In This Directory
- **KernelRegistry** loads .metallib files and creates pipelines on-demand
- **KernelCache** caches compiled pipelines to avoid recompilation (default: 512 entries)
- **kernel_launch.h** provides utilities for computing launch configurations that respect pipeline limits
### Common Patterns
```cpp
// Register and load kernels
KernelRegistry& registry = KernelRegistry::instance();
registry.load_library("/path/to/kernels.metallib");
registry.register_kernel("my_kernel", "my_kernel_function");

// Get pipeline (cached after first call)
id<MTLComputePipelineState> pipeline = registry.get_pipeline("my_kernel");

// Compute launch config
LaunchConfig config = compute_launch_config_1d(pipeline, 10000);

// Dispatch
id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
dispatch_kernel(encoder, pipeline, config);
[encoder endEncoding];
```
## Dependencies
### Internal
- core/ (MNDevice)
### External
- Metal framework (MTLComputePipelineState, MTLLibrary, MTLFunction)
<!-- MANUAL: -->
