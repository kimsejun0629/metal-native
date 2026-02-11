/// @file allocator.mm
/// @brief Objective-C++ implementation of MetalSmartAllocator.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/memory/allocator.h"
#include "metal_native/memory/heap_manager.h"
#include "metal_native/core/buffer.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

#include <cmath>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace metal_native {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum allocation size (one GPU page).
static constexpr size_t kMinAllocSize = 16384; // 16 KB

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Round @p size up to the next size class.
/// Size classes use power-of-2 with 1.5x intermediates:
/// 16KB, 24KB, 32KB, 48KB, 64KB, 96KB, 128KB, 192KB, 256KB, ...
/// This reduces worst-case internal fragmentation from ~50% to ~33%.
static size_t next_size_class(size_t size) {
    if (size <= kMinAllocSize) return kMinAllocSize;

    // Find the enclosing power-of-2 range
    size_t pow2 = kMinAllocSize;
    while (pow2 < size) {
        pow2 <<= 1;
    }

    // Check if the 1.5x intermediate of the previous power fits
    size_t half_pow2 = pow2 >> 1;
    size_t intermediate = half_pow2 + (half_pow2 >> 1); // 1.5 * half_pow2

    if (size <= intermediate && intermediate < pow2) {
        return intermediate;
    }
    return pow2;
}

/// Compute a bucket key from size class and storage mode.
/// We use the size class as the high bits and mode as the low bit.
static uint64_t bucket_key(size_t size_class, StorageMode mode) {
    return (static_cast<uint64_t>(size_class) << 1) |
            static_cast<uint64_t>(mode == StorageMode::Private ? 1 : 0);
}

// ---------------------------------------------------------------------------
// CachedBlock -- a free-list entry
// ---------------------------------------------------------------------------

struct CachedBlock {
    HeapBlock   heap_block;  ///< Underlying heap sub-allocation.
    size_t      size_class;  ///< Power-of-2 bucket.
    StorageMode mode;
};

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct MetalSmartAllocator::Impl {
    MNDevice&      device;
    HeapManager    heap_mgr;
    mutable std::mutex mu;

    /// Free lists keyed by bucket_key(size_class, mode).
    std::unordered_map<uint64_t, std::vector<CachedBlock>> free_lists;

    /// Map from internal_id to the underlying HeapBlock (for deallocate).
    std::unordered_map<uint64_t, HeapBlock> live_blocks;

    /// Monotonically increasing ID for each allocation.
    uint64_t next_id = 1;

    // Statistics.
    size_t allocated_bytes  = 0;
    size_t cached_bytes     = 0;
    size_t peak_bytes       = 0;
    size_t allocation_count = 0;
    size_t cache_hit_count  = 0;
    size_t cache_miss_count = 0;

    explicit Impl(MNDevice& dev) : device(dev), heap_mgr(dev) {}
};

// ---------------------------------------------------------------------------
// MetalSmartAllocator public API
// ---------------------------------------------------------------------------

MetalSmartAllocator::MetalSmartAllocator(MNDevice& device)
    : impl_(std::make_unique<Impl>(device)) {}

MetalSmartAllocator::~MetalSmartAllocator() {
    // Release all cached blocks before destroying the heap manager.
    empty_cache();
}

AllocatedBlock MetalSmartAllocator::allocate(size_t nbytes, StorageMode mode) {
    MN_CHECK(nbytes > 0,
             MetalNativeError::InvalidArgument,
             "MetalSmartAllocator::allocate: nbytes must be > 0");

    const size_t sc  = next_size_class(nbytes);
    const uint64_t key = bucket_key(sc, mode);

    std::lock_guard<std::mutex> lock(impl_->mu);

    HeapBlock heap_block;
    bool from_cache = false;

    // 1. Check free list for a cached block.
    auto it = impl_->free_lists.find(key);
    if (it != impl_->free_lists.end() && !it->second.empty()) {
        auto& bucket = it->second;
        CachedBlock cached = std::move(bucket.back());
        bucket.pop_back();

        heap_block = cached.heap_block;
        from_cache = true;

        impl_->cached_bytes -= sc;
        impl_->cache_hit_count += 1;
    }

    // 2. Fall back to heap sub-allocation.
    if (!from_cache) {
        heap_block = impl_->heap_mgr.allocate(sc, mode);
        impl_->cache_miss_count += 1;
    }

    // 3. Build the returned handle.
    const uint64_t id = impl_->next_id++;

    impl_->live_blocks[id] = heap_block;
    impl_->allocated_bytes += sc;
    impl_->allocation_count += 1;
    if (impl_->allocated_bytes > impl_->peak_bytes) {
        impl_->peak_bytes = impl_->allocated_bytes;
    }

    AllocatedBlock block;
    block.size        = sc;
    block.requested   = nbytes;
    block.size_class  = sc;
    block.mode        = mode;
    block.aliasable   = false;
    block.buffer      = heap_block.buffer;
    block.internal_id = id;

    return block;
}

void MetalSmartAllocator::deallocate(AllocatedBlock& block) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    auto it = impl_->live_blocks.find(block.internal_id);
    MN_CHECK(it != impl_->live_blocks.end(),
             MetalNativeError::InvalidArgument,
             "MetalSmartAllocator::deallocate: unknown block (double free?)");

    HeapBlock heap_block = it->second;
    impl_->live_blocks.erase(it);

    impl_->allocated_bytes -= block.size;
    impl_->allocation_count -= 1;

    // If the block was marked aliasable, we cannot reuse it from the
    // free list because Metal may have reclaimed the storage.  Return
    // it directly to the heap manager.
    if (block.aliasable) {
        impl_->heap_mgr.deallocate(heap_block);
    } else {
        // Place in the free-list cache for reuse.
        const uint64_t key = bucket_key(block.size_class, block.mode);
        CachedBlock cached;
        cached.heap_block = heap_block;
        cached.size_class = block.size_class;
        cached.mode       = block.mode;

        impl_->free_lists[key].push_back(std::move(cached));
        impl_->cached_bytes += block.size;
    }

    // Clear the caller's handle.
    block.buffer      = nil;
    block.size        = 0;
    block.requested   = 0;
    block.internal_id = 0;
}

void MetalSmartAllocator::make_aliasable(AllocatedBlock& block) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    MN_CHECK(block.buffer != nil,
             MetalNativeError::InvalidArgument,
             "MetalSmartAllocator::make_aliasable: block has no buffer");

    if (!block.aliasable) {
        [block.buffer makeAliasable];
        block.aliasable = true;
    }
}

void MetalSmartAllocator::empty_cache() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    // Return all cached blocks to the heap manager.
    for (auto& [key, bucket] : impl_->free_lists) {
        for (auto& cached : bucket) {
            impl_->heap_mgr.deallocate(cached.heap_block);
        }
    }

    impl_->free_lists.clear();
    impl_->cached_bytes = 0;

    // Release any heaps that are now fully empty.
    impl_->heap_mgr.release_empty_heaps();
}

AllocatorStats MetalSmartAllocator::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mu);

    AllocatorStats s;
    s.allocated_bytes  = impl_->allocated_bytes;
    s.cached_bytes     = impl_->cached_bytes;
    s.peak_bytes       = impl_->peak_bytes;
    s.allocation_count = impl_->allocation_count;
    s.cache_hit_count  = impl_->cache_hit_count;
    s.cache_miss_count = impl_->cache_miss_count;

    return s;
}

} // namespace metal_native
