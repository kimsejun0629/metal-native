#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_native/memory/memory_snapshot.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include <sstream>
#include <iomanip>
#include <ctime>

namespace metal_native {

std::string MemorySnapshot::to_json() const {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"total_allocated\": " << total_allocated << ",\n";
    oss << "  \"total_cached\": " << total_cached << ",\n";
    oss << "  \"peak_allocated\": " << peak_allocated << ",\n";
    oss << "  \"available_system\": " << available_system << ",\n";

    // Heaps array
    oss << "  \"heaps\": [\n";
    for (size_t i = 0; i < heaps.size(); ++i) {
        const auto& heap = heaps[i];
        oss << "    {\n";
        oss << "      \"heap_size\": " << heap.heap_size << ",\n";
        oss << "      \"used_bytes\": " << heap.used_bytes << ",\n";
        oss << "      \"free_bytes\": " << heap.free_bytes << ",\n";
        oss << "      \"num_allocations\": " << heap.num_allocations << ",\n";
        oss << "      \"fragmentation_ratio\": " << std::fixed << std::setprecision(4) << heap.fragmentation_ratio << "\n";
        oss << "    }";
        if (i < heaps.size() - 1) oss << ",";
        oss << "\n";
    }
    oss << "  ],\n";

    // Allocations array
    oss << "  \"allocations\": [\n";
    for (size_t i = 0; i < allocations.size(); ++i) {
        const auto& alloc = allocations[i];
        oss << "    {\n";
        oss << "      \"address\": \"0x" << std::hex << alloc.address << std::dec << "\",\n";
        oss << "      \"size\": " << alloc.size << ",\n";
        oss << "      \"tag\": \"" << alloc.tag << "\",\n";

        // Convert time_point to milliseconds since epoch
        auto duration = alloc.allocated_at.time_since_epoch();
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        oss << "      \"allocated_at_ms\": " << millis << ",\n";

        oss << "      \"is_aliasable\": " << (alloc.is_aliasable ? "true" : "false") << "\n";
        oss << "    }";
        if (i < allocations.size() - 1) oss << ",";
        oss << "\n";
    }
    oss << "  ]\n";
    oss << "}";

    return oss.str();
}

std::string MemorySnapshot::summary() const {
    std::ostringstream oss;

    auto format_bytes = [](size_t bytes) -> std::string {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unit_idx = 0;
        double size = static_cast<double>(bytes);

        while (size >= 1024.0 && unit_idx < 4) {
            size /= 1024.0;
            ++unit_idx;
        }

        std::ostringstream s;
        s << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
        return s.str();
    };

    oss << "=== Memory Snapshot ===\n";
    oss << "Total Allocated: " << format_bytes(total_allocated) << "\n";
    oss << "Total Cached:    " << format_bytes(total_cached) << "\n";
    oss << "Peak Allocated:  " << format_bytes(peak_allocated) << "\n";
    oss << "Available:       " << format_bytes(available_system) << "\n";
    oss << "\n";

    oss << "Heaps: " << heaps.size() << "\n";
    for (size_t i = 0; i < heaps.size(); ++i) {
        const auto& heap = heaps[i];
        oss << "  Heap " << i << ": "
            << format_bytes(heap.used_bytes) << " / " << format_bytes(heap.heap_size)
            << " (" << heap.num_allocations << " allocs, "
            << std::fixed << std::setprecision(2) << (heap.fragmentation_ratio * 100.0) << "% frag)\n";
    }
    oss << "\n";

    oss << "Active Allocations: " << allocations.size() << "\n";

    return oss.str();
}

MemorySnapshot take_memory_snapshot() {
    MemorySnapshot snapshot;

    // Query system available memory (macOS compatible)
    @autoreleasepool {
        NSProcessInfo* info = [NSProcessInfo processInfo];
        snapshot.available_system = info.physicalMemory;
    }

    // Query Metal device for current allocation stats
    @autoreleasepool {
        id<MTLDevice> device = MNDevice::instance().metal_device();

        if (@available(macOS 10.13, iOS 11.0, *)) {
            snapshot.total_allocated = [device currentAllocatedSize];
        } else {
            snapshot.total_allocated = 0;
        }

        // For now, we don't have access to the allocator's internal state
        // These will be populated by the allocator in the future
        snapshot.total_cached = 0;
        snapshot.peak_allocated = snapshot.total_allocated; // Conservative estimate

        // No heap information available yet (will be filled by allocator)
        // For now, create a single synthetic heap representing device memory
        if (snapshot.total_allocated > 0) {
            HeapSnapshot heap;
            heap.heap_size = snapshot.total_allocated;
            heap.used_bytes = snapshot.total_allocated;
            heap.free_bytes = 0;
            heap.num_allocations = 1; // Unknown
            heap.fragmentation_ratio = 0.0; // Unknown
            snapshot.heaps.push_back(heap);
        }

        // No allocation records yet (will be filled by allocator)
    }

    return snapshot;
}

} // namespace metal_native
