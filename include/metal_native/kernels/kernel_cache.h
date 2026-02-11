#pragma once

/// @file kernel_cache.h
/// @brief LRU cache for MTLComputePipelineState objects.
///
/// KernelCache provides O(1) lookup and insertion with LRU eviction when
/// the cache reaches capacity.  Implemented with std::unordered_map for
/// fast key lookup and std::list for O(1) LRU ordering.

#include <cstddef>
#include <memory>
#include <string>

namespace metal_native {

class KernelCache {
public:
    /// Construct a cache with the given maximum number of entries.
    /// @param max_entries  Maximum cached pipeline states (default 512).
    explicit KernelCache(size_t max_entries = 512);
    ~KernelCache();

    // Move-only.
    KernelCache(KernelCache&&) noexcept;
    KernelCache& operator=(KernelCache&&) noexcept;
    KernelCache(const KernelCache&) = delete;
    KernelCache& operator=(const KernelCache&) = delete;

    // -- Lookup / insert -----------------------------------------------------

    /// Look up a cached pipeline state.  Returns nullptr if not found.
    /// Promotes the entry to most-recently-used on hit.
    /// The returned pointer is an id<MTLComputePipelineState> in ObjC++ contexts.
    void* get(const std::string& key);

    /// Insert a pipeline state.  If the cache is full, evicts the
    /// least-recently-used entry.
    /// @param pipeline  An id<MTLComputePipelineState> passed as void*.
    void put(const std::string& key, void* pipeline);

    /// Remove all cached entries.
    void clear();

    // -- Stats ---------------------------------------------------------------

    /// Number of entries currently in the cache.
    size_t size() const;

    /// Maximum number of entries the cache can hold.
    size_t capacity() const;

    /// Cache hit rate (hits / (hits + misses)).  Returns 0.0 if no lookups.
    double hit_rate() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
