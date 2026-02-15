<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# kernels
## Purpose
Implementation files for kernel compilation and dispatch. Implements kernel registry, pipeline cache, and launch configuration utilities.
## Key Files
| File | Description |
|------|-------------|
| CMakeLists.txt | Build configuration for kernels module |
| kernel_registry.mm | KernelRegistry implementation (loads .metallib, creates MTLComputePipelineState) |
| kernel_cache.mm | KernelCache implementation (LRU cache with std::list + unordered_map) |
| kernel_launch.mm | Launch configuration helpers (compute optimal threadgroup sizes) |
## For AI Agents
### Working In This Directory
- **kernel_registry.mm**: Loads MTLLibrary from path or default, creates functions, compiles pipelines
- **kernel_cache.mm**: Hash table + doubly-linked list for O(1) lookup/eviction
- **kernel_launch.mm**: Queries pipeline's maxTotalThreadsPerThreadgroup and threadExecutionWidth
### Common Patterns
```objc
// Load library
id<MTLDevice> device = ...;
id<MTLLibrary> library = [device newLibraryWithFile:path error:&error];

// Create pipeline
id<MTLFunction> function = [library newFunctionWithName:name];
id<MTLComputePipelineState> pipeline =
    [device newComputePipelineStateWithFunction:function error:&error];

// Cache using kernel name as key
cache.put(name, (__bridge void*)pipeline);

// Launch config
MTLSize threadgroup = MTLSizeMake(
    std::min(total, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);
MTLSize grid = MTLSizeMake((total + threadgroup.width - 1) / threadgroup.width, 1, 1);
```
## Dependencies
### Internal
- ../include/metal_native/kernels/ (public headers)
- core/ (MNDevice)
### External
- Metal framework (MTLLibrary, MTLFunction, MTLComputePipelineState)
<!-- MANUAL: -->
