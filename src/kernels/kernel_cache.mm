/// @file kernel_cache.mm
/// @brief Objective-C++ implementation of KernelCache (LRU).

#import <Metal/Metal.h>

#include "metal_native/kernels/kernel_cache.h"

#include <list>
#include <string>
#include <unordered_map>
#include <utility>

namespace metal_native {

// ---------------------------------------------------------------------------
// Impl -- LRU cache using std::list + std::unordered_map for O(1) ops
// ---------------------------------------------------------------------------

struct KernelCache::Impl {
    /// Each entry stores the key and pipeline state.
    using Entry = std::pair<std::string, id<MTLComputePipelineState>>;

    /// Doubly-linked list ordered from most-recently-used (front) to
    /// least-recently-used (back).
    std::list<Entry> order;

    /// Maps key -> iterator into the order list for O(1) lookup.
    std::unordered_map<std::string, std::list<Entry>::iterator> map;

    size_t   max_entries = 512;
    uint64_t hits        = 0;
    uint64_t misses      = 0;
};

// ---------------------------------------------------------------------------
// Constructor / destructor / move
// ---------------------------------------------------------------------------

KernelCache::KernelCache(size_t max_entries)
    : impl_(std::make_unique<Impl>()) {
    impl_->max_entries = (max_entries > 0) ? max_entries : 1;
}

KernelCache::~KernelCache() = default;

KernelCache::KernelCache(KernelCache&&) noexcept = default;
KernelCache& KernelCache::operator=(KernelCache&&) noexcept = default;

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

id<MTLComputePipelineState> KernelCache::get(const std::string& key) {
    auto it = impl_->map.find(key);
    if (it == impl_->map.end()) {
        ++impl_->misses;
        return nil;
    }

    // Move the accessed entry to the front (most-recently-used).
    impl_->order.splice(impl_->order.begin(), impl_->order, it->second);
    ++impl_->hits;
    return it->second->second;
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

void KernelCache::put(const std::string& key,
                      id<MTLComputePipelineState> pipeline) {
    auto it = impl_->map.find(key);
    if (it != impl_->map.end()) {
        // Update existing entry and promote to front.
        it->second->second = pipeline;
        impl_->order.splice(impl_->order.begin(), impl_->order, it->second);
        return;
    }

    // Evict the least-recently-used entry if at capacity.
    if (impl_->map.size() >= impl_->max_entries) {
        auto& back = impl_->order.back();
        impl_->map.erase(back.first);
        impl_->order.pop_back();
    }

    // Insert new entry at the front.
    impl_->order.emplace_front(key, pipeline);
    impl_->map[key] = impl_->order.begin();
}

// ---------------------------------------------------------------------------
// Clear / stats
// ---------------------------------------------------------------------------

void KernelCache::clear() {
    impl_->order.clear();
    impl_->map.clear();
    impl_->hits   = 0;
    impl_->misses = 0;
}

size_t KernelCache::size() const {
    return impl_->map.size();
}

size_t KernelCache::capacity() const {
    return impl_->max_entries;
}

double KernelCache::hit_rate() const {
    uint64_t total = impl_->hits + impl_->misses;
    if (total == 0) return 0.0;
    return static_cast<double>(impl_->hits) / static_cast<double>(total);
}

} // namespace metal_native
