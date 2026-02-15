<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# memory
## Purpose
Implementation files for GPU memory management and caching. Implements heap manager, smart allocator, budget controller, memory pressure monitoring, and specialized caches (KV, activation, quantized weights).
## Key Files
| File | Description |
|------|-------------|
| CMakeLists.txt | Build configuration for memory module |
| heap_manager.mm | HeapManager implementation (MTLHeap pools, best-fit allocation, coalescing) |
| allocator.mm | MetalSmartAllocator implementation (power-of-2 free lists, heap backing) |
| budget_controller.mm | MemoryBudgetController implementation (distributes budget across strategies) |
| memory_pressure.mm | MemoryPressureMonitor implementation (dispatch_source integration) |
| kv_cache.mm | KVCache implementation (pre-allocated K/V buffers for autoregressive generation) |
| activation_cache.mm | ActivationCache implementation (LRU cache for layer outputs) |
| quant_weight_cache.mm | QuantWeightCache implementation (INT4/INT8 storage with FP16 dequantization) |
| memory_snapshot.mm | MemorySnapshot implementation (serialize allocator state to JSON) |
## For AI Agents
### Working In This Directory
- **heap_manager.mm**: Creates MTLHeap objects in size tiers (4MB, 64MB, 256MB), sub-allocates with 16KB alignment
- **allocator.mm**: Buckets sizes into power-of-2 classes, maintains per-class free lists
- **budget_controller.mm**: Queries MTLDevice currentAllocatedSize and recommendedMaxWorkingSetSize
- **memory_pressure.mm**: Uses DISPATCH_SOURCE_TYPE_MEMORYPRESSURE for system events
- Caches resize or evict entries when pressure reaches Warning/Critical levels
### Common Patterns
- All `.mm` files use Objective-C++ for Metal API access
- Allocator uses std::unordered_map for free lists, std::mutex for thread safety
- Pressure monitor callback is invoked on dispatch queue, not main thread
- Caches use LRU eviction (std::list + unordered_map for O(1) operations)
## Dependencies
### Internal
- ../include/metal_native/memory/ (public headers)
- core/ (MNDevice, MNTensor, MNBuffer, MNDType)
### External
- Metal framework (MTLHeap, MTLBuffer, MTLDevice memory queries)
- libdispatch (dispatch_source for memory pressure)
<!-- MANUAL: -->
