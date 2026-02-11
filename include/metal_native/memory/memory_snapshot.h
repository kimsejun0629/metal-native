#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>

namespace metal_native {

struct AllocationRecord {
    uint64_t address;
    size_t size;
    std::string tag;         // e.g., "tensor:float32:[1024,768]"
    std::chrono::steady_clock::time_point allocated_at;
    bool is_aliasable;
};

struct HeapSnapshot {
    size_t heap_size;
    size_t used_bytes;
    size_t free_bytes;
    size_t num_allocations;
    double fragmentation_ratio;  // free_bytes / heap_size (lower is better for used heaps)
};

struct MemorySnapshot {
    // Overall stats
    size_t total_allocated;
    size_t total_cached;
    size_t peak_allocated;
    size_t available_system;

    // Per-heap snapshots
    std::vector<HeapSnapshot> heaps;

    // Individual allocations
    std::vector<AllocationRecord> allocations;

    // Serialize to JSON string
    std::string to_json() const;

    // Summary string (human-readable)
    std::string summary() const;
};

// Take a snapshot of current memory state
// In the future, this will call into MetalSmartAllocator
// For now, provide a standalone implementation that queries MTLDevice
MemorySnapshot take_memory_snapshot();

} // namespace metal_native
