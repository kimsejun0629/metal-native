#pragma once

/// @file allocator.h
/// @brief High-level GPU memory allocator with free-list caching.
///
/// MetalSmartAllocator is the public allocation API for metal_native.  It
/// maintains per-size-class free lists for fast buffer reuse and falls back
/// to HeapManager for fresh allocations.
///
/// Size classes are power-of-2 buckets.  Freed buffers are cached in the
/// appropriate bucket and reused on the next matching allocation, avoiding
/// expensive MTLHeap sub-allocation round-trips.
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

/// Statistics snapshot for the allocator.
struct AllocatorStats {
    size_t allocated_bytes   = 0; ///< Bytes currently in use by the caller.
    size_t cached_bytes      = 0; ///< Bytes held in free lists (reusable).
    size_t peak_bytes        = 0; ///< High-water mark of allocated_bytes.
    size_t allocation_count  = 0; ///< Number of live allocations.
    size_t cache_hit_count   = 0; ///< Number of allocations served from cache.
    size_t cache_miss_count  = 0; ///< Number of allocations requiring new heap sub-allocation.
};

/// Handle returned by the allocator, wrapping a GPU buffer.
struct AllocatedBlock {
    size_t size          = 0; ///< Usable allocation size (may be rounded up).
    size_t requested     = 0; ///< Original requested size.
    size_t size_class    = 0; ///< Power-of-2 bucket this block belongs to.
    StorageMode mode;         ///< Storage mode of the allocation.
    bool   aliasable     = false; ///< True if makeAliasable has been called.

#ifdef __OBJC__
    id<MTLBuffer> buffer = nil;
#else
    void* buffer = nullptr;
#endif

    /// Internal bookkeeping index (opaque to callers).
    uint64_t internal_id = 0;
};

/// High-level GPU memory allocator with caching.
class MetalSmartAllocator {
public:
    /// Construct an allocator that uses the given device.
    explicit MetalSmartAllocator(MNDevice& device);
    ~MetalSmartAllocator();

    // Non-copyable, non-movable.
    MetalSmartAllocator(const MetalSmartAllocator&) = delete;
    MetalSmartAllocator& operator=(const MetalSmartAllocator&) = delete;
    MetalSmartAllocator(MetalSmartAllocator&&) = delete;
    MetalSmartAllocator& operator=(MetalSmartAllocator&&) = delete;

    // -- Allocation ----------------------------------------------------------

    /// Allocate a GPU buffer of at least @p nbytes bytes.
    ///
    /// The allocation is served from a cached free-list block if one of a
    /// suitable size class is available; otherwise a new block is
    /// sub-allocated from the HeapManager.
    ///
    /// @param nbytes  Requested size in bytes (must be > 0).
    /// @param mode    Storage mode (default: Shared).
    /// @return        An AllocatedBlock describing the allocation.
    /// @throws MNException on failure.
    AllocatedBlock allocate(size_t nbytes, StorageMode mode = StorageMode::Shared);

    /// Return a buffer to the free-list cache for future reuse.
    ///
    /// The buffer is not immediately released; it is placed in the
    /// appropriate size-class bucket.
    ///
    /// @param block  The AllocatedBlock previously returned by allocate().
    void deallocate(AllocatedBlock& block);

    /// Mark a buffer as aliasable so Metal can reuse its underlying storage.
    ///
    /// After calling this, the buffer contents are undefined.  The block
    /// remains valid and can be deallocated normally.
    ///
    /// @param block  The AllocatedBlock to mark.
    void make_aliasable(AllocatedBlock& block);

    // -- Cache management ----------------------------------------------------

    /// Drain all free-list caches and release empty heaps.
    ///
    /// Live allocations are not affected.
    void empty_cache();

    // -- Statistics ----------------------------------------------------------

    /// Return a snapshot of current allocator statistics.
    AllocatorStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
