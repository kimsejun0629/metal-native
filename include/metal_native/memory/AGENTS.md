<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# memory
## Purpose
GPU memory management and specialized caching. Includes heap manager, smart allocator with free-list caching, budget controller, KV cache, activation cache, quantized weight cache, memory snapshots, and system pressure monitoring.
## Key Files
| File | Description |
|------|-------------|
| allocator.h | MetalSmartAllocator: high-level allocator with size-class free lists |
| heap_manager.h | HeapManager: MTLHeap pool manager with best-fit allocation and coalescing |
| budget_controller.h | MemoryBudgetController: distributes memory budget across cache strategies |
| memory_pressure.h | MemoryPressureMonitor: macOS dispatch_source integration for pressure events |
| kv_cache.h | KVCache: pre-allocated key/value cache for autoregressive generation |
| activation_cache.h | ActivationCache: LRU cache for layer activations (prefix caching) |
| quant_weight_cache.h | QuantWeightCache: stores quantized weights with on-demand dequantization |
| memory_snapshot.h | MemorySnapshot: serializable memory state for debugging/profiling |
## For AI Agents
### Working In This Directory
- **MetalSmartAllocator** is the primary allocation API - uses free lists for fast reuse
- **HeapManager** backs the allocator with MTLHeap sub-allocation (16KB page-aligned)
- **MemoryBudgetController** grants cache budgets based on current pressure level
- **MemoryPressureMonitor** listens to macOS pressure events and triggers callbacks
- Caches (KV, activation, quant weight) shrink dynamically under pressure
### Common Patterns
```cpp
// Allocate from smart allocator
MetalSmartAllocator& alloc = device.allocator();
AllocatedBlock block = alloc.allocate(1024 * 1024, StorageMode::Shared);
// ... use block.buffer ...
alloc.deallocate(block);

// Request cache budget
MemoryBudgetController& budget = MemoryBudgetController::instance();
GrantedBudget grant = budget.request_budget(BudgetStrategy::KVCache, 256_MB);
if (grant.approved) {
    // allocate cache
}

// Monitor pressure
MemoryPressureMonitor& monitor = MemoryPressureMonitor::instance();
monitor.set_callback([](MemoryPressureLevel level) {
    if (level == MemoryPressureLevel::Critical) {
        // evict caches
    }
});
monitor.start();
```
## Dependencies
### Internal
- core/ (MNDevice, MNTensor, MNBuffer, MNDType)
### External
- Metal framework (MTLHeap, MTLBuffer)
- libdispatch (dispatch_source for pressure monitoring)
<!-- MANUAL: -->
