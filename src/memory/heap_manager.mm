/// @file heap_manager.mm
/// @brief Objective-C++ implementation of HeapManager.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/memory/heap_manager.h"
#include "metal_native/core/buffer.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

#include <algorithm>
#include <mutex>
#include <vector>

namespace metal_native {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// GPU page size on Apple Silicon (16 KB).
static constexpr size_t kGPUPageSize = 16384;

/// Size tier thresholds.
static constexpr size_t kSmallThreshold  = 1UL * 1024 * 1024;   // 1 MB
static constexpr size_t kMediumThreshold = 16UL * 1024 * 1024;  // 16 MB

/// Heap sizes per tier.
static constexpr size_t kSmallHeapSize  = 4UL * 1024 * 1024;    // 4 MB
static constexpr size_t kMediumHeapSize = 64UL * 1024 * 1024;   // 64 MB
static constexpr size_t kLargeHeapSize  = 256UL * 1024 * 1024;  // 256 MB

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Round @p size up to the next multiple of kGPUPageSize.
static size_t align_to_page(size_t size) {
    return (size + kGPUPageSize - 1) & ~(kGPUPageSize - 1);
}

/// Choose the heap capacity for the given allocation size.
static size_t heap_size_for_alloc(size_t aligned_size) {
    if (aligned_size < kSmallThreshold) {
        return kSmallHeapSize;
    } else if (aligned_size <= kMediumThreshold) {
        return kMediumHeapSize;
    } else {
        // For very large allocations, the heap must be at least as large as
        // the allocation itself (rounded up to a multiple of kLargeHeapSize).
        size_t min_heap = std::max(kLargeHeapSize, aligned_size);
        return align_to_page(min_heap);
    }
}

/// Convert StorageMode to MTLStorageMode.
static MTLStorageMode to_mtl_storage_mode(StorageMode mode) {
    switch (mode) {
        case StorageMode::Shared:  return MTLStorageModeShared;
        case StorageMode::Private: return MTLStorageModePrivate;
    }
    return MTLStorageModeShared;
}

// ---------------------------------------------------------------------------
// FreeBlock -- a contiguous free region within a heap
// ---------------------------------------------------------------------------

struct FreeBlock {
    size_t offset = 0;
    size_t size   = 0;
};

// ---------------------------------------------------------------------------
// HeapEntry -- a single MTLHeap and its free-block list
// ---------------------------------------------------------------------------

struct HeapEntry {
    id<MTLHeap>          heap       = nil;
    size_t               capacity   = 0;
    StorageMode          mode       = StorageMode::Shared;
    std::vector<FreeBlock> free_blocks;
    size_t               alloc_count = 0; // live sub-allocations
};

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct HeapManager::Impl {
    MNDevice&              device;
    std::vector<HeapEntry> heaps;
    mutable std::mutex     mu;

    // Running counters for stats.
    size_t total_heap_bytes   = 0;
    size_t allocated_bytes    = 0;
    size_t allocation_count   = 0;

    explicit Impl(MNDevice& dev) : device(dev) {}

    // -- Internal helpers ----------------------------------------------------

    /// Try to find a free block in an existing heap using best-fit.
    /// Returns the heap index and free-block index, or (-1, -1) on failure.
    std::pair<size_t, size_t> find_best_fit(
            size_t aligned_size, StorageMode mode) {
        size_t best_heap  = SIZE_MAX;
        size_t best_block = SIZE_MAX;
        size_t best_waste = SIZE_MAX;

        for (size_t hi = 0; hi < heaps.size(); ++hi) {
            auto& entry = heaps[hi];
            if (entry.mode != mode) continue;

            for (size_t bi = 0; bi < entry.free_blocks.size(); ++bi) {
                auto& fb = entry.free_blocks[bi];
                if (fb.size >= aligned_size) {
                    size_t waste = fb.size - aligned_size;
                    if (waste < best_waste) {
                        best_waste = waste;
                        best_heap  = hi;
                        best_block = bi;
                        if (waste == 0) goto done;
                    }
                }
            }
        }
    done:
        return {best_heap, best_block};
    }

    /// Create a new MTLHeap of the given capacity and storage mode.
    size_t create_heap(size_t capacity, StorageMode mode) {
        @autoreleasepool {
            MTLHeapDescriptor* desc = [[MTLHeapDescriptor alloc] init];
            desc.size        = capacity;
            desc.storageMode = to_mtl_storage_mode(mode);
            desc.cpuCacheMode = MTLCPUCacheModeDefaultCache;
            desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
            desc.type = MTLHeapTypePlacement;

            id<MTLHeap> heap = [device.metal_device() newHeapWithDescriptor:desc];
            MN_CHECK(heap != nil,
                     MetalNativeError::OutOfMemory,
                     "HeapManager: failed to create MTLHeap of " +
                     std::to_string(capacity) + " bytes");

            HeapEntry entry;
            entry.heap     = heap;
            entry.capacity = capacity;
            entry.mode     = mode;
            // The entire heap starts as one free block.
            entry.free_blocks.push_back({0, capacity});

            heaps.push_back(std::move(entry));
            total_heap_bytes += capacity;

            return heaps.size() - 1;
        }
    }

    /// Coalesce adjacent free blocks in the given heap.
    void coalesce(HeapEntry& entry) {
        if (entry.free_blocks.size() < 2) return;

        std::sort(entry.free_blocks.begin(), entry.free_blocks.end(),
                  [](const FreeBlock& a, const FreeBlock& b) {
                      return a.offset < b.offset;
                  });

        std::vector<FreeBlock> merged;
        merged.reserve(entry.free_blocks.size());
        merged.push_back(entry.free_blocks[0]);

        for (size_t i = 1; i < entry.free_blocks.size(); ++i) {
            auto& prev = merged.back();
            auto& curr = entry.free_blocks[i];
            if (prev.offset + prev.size == curr.offset) {
                prev.size += curr.size;
            } else {
                merged.push_back(curr);
            }
        }

        entry.free_blocks = std::move(merged);
    }
};

// ---------------------------------------------------------------------------
// HeapManager public API
// ---------------------------------------------------------------------------

HeapManager::HeapManager(MNDevice& device)
    : impl_(std::make_unique<Impl>(device)) {}

HeapManager::~HeapManager() = default;

HeapBlock HeapManager::allocate(size_t size, StorageMode mode) {
    MN_CHECK(size > 0,
             MetalNativeError::InvalidArgument,
             "HeapManager::allocate: size must be > 0");

    const size_t aligned_size = align_to_page(size);

    std::lock_guard<std::mutex> lock(impl_->mu);

    // 1. Try best-fit in existing heaps.
    auto [hi, bi] = impl_->find_best_fit(aligned_size, mode);

    // 2. If no suitable block found, create a new heap.
    if (hi == SIZE_MAX) {
        size_t heap_cap = heap_size_for_alloc(aligned_size);
        hi = impl_->create_heap(heap_cap, mode);
        bi = 0; // The new heap has exactly one free block.
    }

    auto& entry = impl_->heaps[hi];
    auto& fb    = entry.free_blocks[bi];

    // Record the offset before we split.
    const size_t alloc_offset = fb.offset;

    // Split or consume the free block.
    if (fb.size > aligned_size) {
        fb.offset += aligned_size;
        fb.size   -= aligned_size;
    } else {
        entry.free_blocks.erase(entry.free_blocks.begin() +
                                static_cast<ptrdiff_t>(bi));
    }

    // Sub-allocate an MTLBuffer from the heap at the chosen offset.
    MTLResourceOptions options =
        static_cast<MTLResourceOptions>(
            static_cast<uint32_t>(mode)) << MTLResourceStorageModeShift;

    id<MTLBuffer> buffer = [entry.heap newBufferWithLength:aligned_size
                                                   options:options
                                                    offset:alloc_offset];
    MN_CHECK(buffer != nil,
             MetalNativeError::OutOfMemory,
             "HeapManager: MTLHeap newBufferWithLength returned nil "
             "(requested " + std::to_string(aligned_size) + " bytes)");

    entry.alloc_count += 1;
    impl_->allocated_bytes += aligned_size;
    impl_->allocation_count += 1;

    HeapBlock block;
    block.offset     = alloc_offset;
    block.size       = aligned_size;
    block.heap_index = hi;
    block.buffer     = buffer;
    block.heap       = entry.heap;

    return block;
}

void HeapManager::deallocate(HeapBlock& block) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    MN_CHECK(block.heap_index < impl_->heaps.size(),
             MetalNativeError::InvalidArgument,
             "HeapManager::deallocate: invalid heap_index");

    auto& entry = impl_->heaps[block.heap_index];

    // Return the region to the free list.
    entry.free_blocks.push_back({block.offset, block.size});

    // Coalesce adjacent free blocks.
    impl_->coalesce(entry);

    entry.alloc_count -= 1;
    impl_->allocated_bytes -= block.size;
    impl_->allocation_count -= 1;

    // Clear the block handle.
    block.buffer = nil;
    block.heap   = nil;
    block.offset = 0;
    block.size   = 0;
}

void HeapManager::release_empty_heaps() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    auto it = impl_->heaps.begin();
    while (it != impl_->heaps.end()) {
        if (it->alloc_count == 0) {
            impl_->total_heap_bytes -= it->capacity;
            it = impl_->heaps.erase(it);
        } else {
            ++it;
        }
    }

    // Re-index: after erasing, heap_index values in outstanding HeapBlocks
    // are invalidated.  Callers must not hold stale HeapBlocks across
    // release_empty_heaps().  This is acceptable because the allocator
    // layer (MetalSmartAllocator) ensures blocks are deallocated first.
}

HeapStats HeapManager::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mu);

    HeapStats s;
    s.total_heap_bytes  = impl_->total_heap_bytes;
    s.allocated_bytes   = impl_->allocated_bytes;
    s.heap_count        = impl_->heaps.size();
    s.allocation_count  = impl_->allocation_count;

    // Compute available bytes and largest free block.
    size_t largest_free = 0;
    for (auto& entry : impl_->heaps) {
        for (auto& fb : entry.free_blocks) {
            s.available_bytes += fb.size;
            if (fb.size > largest_free) {
                largest_free = fb.size;
            }
        }
    }

    // Fragmentation = 1 - (largest_free / available).
    if (s.available_bytes > 0) {
        s.fragmentation_ratio =
            1.0 - static_cast<double>(largest_free) /
                   static_cast<double>(s.available_bytes);
    }

    return s;
}

} // namespace metal_native
