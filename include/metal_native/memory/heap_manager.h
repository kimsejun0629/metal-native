#pragma once

/// @file heap_manager.h
/// @brief MTLHeap pool manager for efficient GPU sub-allocation.
///
/// HeapManager creates and manages pools of MTLHeap objects, organised by size
/// tier and storage mode.  Allocations are sub-allocated from heaps using a
/// best-fit strategy with 16 KB GPU page alignment.  Adjacent free blocks are
/// coalesced on deallocation to reduce fragmentation.
///
/// Size tiers:
///   Small  (< 1 MB)   -- heap size  4 MB
///   Medium (1-16 MB)   -- heap size 64 MB
///   Large  (> 16 MB)   -- heap size 256 MB
///
/// Thread-safe: all public methods are guarded by a mutex.

#include <cstddef>
#include <cstdint>
#include <memory>

#include "metal_native/core/buffer.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace metal_native {

/// Statistics snapshot for the heap manager.
struct HeapStats {
    size_t total_heap_bytes    = 0; ///< Sum of all heap capacities.
    size_t allocated_bytes     = 0; ///< Bytes currently sub-allocated.
    size_t available_bytes     = 0; ///< Bytes available in free blocks.
    size_t heap_count          = 0; ///< Number of live heaps.
    size_t allocation_count    = 0; ///< Number of live sub-allocations.
    double fragmentation_ratio = 0; ///< 1 - (largest_free / available).
};

/// Handle returned by HeapManager representing a sub-allocation.
struct HeapBlock {
    size_t offset = 0;   ///< Byte offset within the heap.
    size_t size   = 0;   ///< Usable size in bytes (aligned).
    size_t heap_index = 0; ///< Internal index of the owning heap.

#ifdef __OBJC__
    id<MTLBuffer> buffer = nil; ///< The sub-allocated MTLBuffer.
    id<MTLHeap>   heap   = nil; ///< The owning MTLHeap.
#else
    void* buffer = nullptr;
    void* heap   = nullptr;
#endif
};

/// Manages pools of MTLHeap objects for efficient sub-allocation.
class HeapManager {
public:
    /// Construct a HeapManager that allocates heaps from the given device.
    explicit HeapManager(MNDevice& device);
    ~HeapManager();

    // Non-copyable, non-movable.
    HeapManager(const HeapManager&) = delete;
    HeapManager& operator=(const HeapManager&) = delete;
    HeapManager(HeapManager&&) = delete;
    HeapManager& operator=(HeapManager&&) = delete;

    // -- Allocation ----------------------------------------------------------

    /// Sub-allocate a buffer of @p size bytes from a heap.
    ///
    /// The returned HeapBlock owns the MTLBuffer.  The allocation is aligned
    /// to the GPU page size (16 KB).
    ///
    /// @param size  Requested size in bytes (must be > 0).
    /// @param mode  Storage mode (Shared or Private).
    /// @return      A HeapBlock describing the allocation.
    /// @throws MNException on failure (OutOfMemory / InvalidArgument).
    HeapBlock allocate(size_t size, StorageMode mode);

    /// Return a sub-allocation to the free list.
    ///
    /// Adjacent free blocks are coalesced automatically.
    ///
    /// @param block  The HeapBlock previously returned by allocate().
    void deallocate(HeapBlock& block);

    // -- Cache management ----------------------------------------------------

    /// Release all empty heaps back to the system.
    /// Heaps with live allocations are kept.
    void release_empty_heaps();

    // -- Statistics ----------------------------------------------------------

    /// Return a snapshot of current heap statistics.
    HeapStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
