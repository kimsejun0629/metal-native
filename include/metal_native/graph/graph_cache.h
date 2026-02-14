#pragma once

/// @file graph_cache.h
/// @brief Two-level cache for MPSGraphExecutable objects.
///
/// The GraphCache maintains an in-memory LRU cache (L1) and a disk-backed
/// persistent cache (L2) for compiled MPSGraph executables. Each cache entry
/// is keyed by topology hash, shape tuple, and data type.

#include "metal_native/core/dtype.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#ifdef __OBJC__
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#endif

namespace metal_native {

// ---------------------------------------------------------------------------
// GraphCacheKey
// ---------------------------------------------------------------------------

/// Cache key identifying a unique MPSGraph executable.
struct GraphCacheKey {
    /// Hash of the graph topology (operation structure).
    uint64_t topology_hash;

    /// Input/output tensor shapes.
    std::vector<size_t> shape_tuple;

    /// Data type of the graph's primary tensors.
    MNDType dtype;

    /// Equality comparison for use in unordered containers.
    bool operator==(const GraphCacheKey& other) const;
};

// ---------------------------------------------------------------------------
// GraphCacheStats
// ---------------------------------------------------------------------------

/// Statistics for cache hit/miss rates.
struct GraphCacheStats {
    uint64_t l1_hits   = 0;  ///< L1 (in-memory) cache hits
    uint64_t l1_misses = 0;  ///< L1 cache misses
    uint64_t l2_hits   = 0;  ///< L2 (disk) cache hits
    uint64_t l2_misses = 0;  ///< L2 cache misses

    /// L1 hit rate (0.0 to 1.0).
    double l1_hit_rate() const;

    /// Overall hit rate including L2 (0.0 to 1.0).
    double overall_hit_rate() const;
};

} // namespace metal_native

// ---------------------------------------------------------------------------
// Hash function for GraphCacheKey
// ---------------------------------------------------------------------------

namespace std {
template <>
struct hash<metal_native::GraphCacheKey> {
    size_t operator()(const metal_native::GraphCacheKey& key) const noexcept;
};
} // namespace std

namespace metal_native {

// ---------------------------------------------------------------------------
// GraphCache
// ---------------------------------------------------------------------------

/// Two-level cache for MPSGraphExecutable objects.
///
/// L1 is an in-memory LRU cache with a configurable maximum size (default 128).
/// L2 is a disk-backed cache storing serialized .mpsgraphpackage files in
/// ~/.cache/metal_native/graph_cache/.
///
/// The cache is thread-safe and supports automatic eviction of stale entries
/// via a background timer.
class GraphCache {
public:
    // -- Constructor / destructor --------------------------------------------

    /// Create a GraphCache with the specified L1 capacity.
    /// @param max_l1_entries  Maximum number of entries in the L1 cache (default: 128).
    explicit GraphCache(size_t max_l1_entries = 128);

    ~GraphCache();

    // -- Non-copyable / non-movable ------------------------------------------
    GraphCache(const GraphCache&) = delete;
    GraphCache& operator=(const GraphCache&) = delete;
    GraphCache(GraphCache&&) = delete;
    GraphCache& operator=(GraphCache&&) = delete;

    // -- Lookup --------------------------------------------------------------

#ifdef __OBJC__
    /// Look up a cached executable by key.
    /// Returns the cached MPSGraphExecutable if found in L1 or L2, or nil if not found.
    /// On L2 hit, the entry is promoted to L1.
    MPSGraphExecutable* lookup(const GraphCacheKey& key);
#else
    /// Opaque handle to the executable (cast in .mm translation units).
    void* lookup(const GraphCacheKey& key);
#endif

    // -- Insertion -----------------------------------------------------------

#ifdef __OBJC__
    /// Insert a new executable into the cache.
    /// The entry is added to L1 and asynchronously serialized to L2.
    void insert(const GraphCacheKey& key, MPSGraphExecutable* executable);
#else
    /// Opaque handle variant.
    void insert(const GraphCacheKey& key, void* executable);
#endif

    // -- Eviction timer ------------------------------------------------------

    /// Start a background timer that periodically evicts L1 entries unused
    /// for more than 60 seconds.
    /// @param interval_seconds  How often the timer fires (default: 30 seconds).
    void start_eviction_timer(unsigned int interval_seconds = 30);

    /// Stop the eviction timer.
    void stop_eviction_timer();

    // -- Cache management ----------------------------------------------------

    /// Clear all entries from L1 and L2.
    void clear();

    /// Clear only the L1 (in-memory) cache.
    void clear_l1();

    /// Clear only the L2 (disk) cache.
    void clear_l2();

    // -- Statistics ----------------------------------------------------------

    /// Return current cache statistics.
    GraphCacheStats stats() const;

    /// Number of entries currently in L1.
    size_t l1_size() const;

    /// Maximum L1 capacity.
    size_t l1_capacity() const;

    /// Dynamically resize the L1 cache capacity.
    /// If new capacity < current size, LRU entries are evicted.
    void set_l1_capacity(size_t new_max_entries);

    /// Estimate the memory footprint of all cached entries (bytes).
    size_t memory_footprint() const;

    // -- Shape bucketing -----------------------------------------------------

    /// Enable or disable shape bucketing for cache key normalization.
    /// When enabled, shape tuples are rounded up to predefined bucket sizes
    /// before lookup/insertion, reducing cache misses for similar shapes.
    /// @param enabled  True to enable bucketing, false to disable.
    void set_bucketing_enabled(bool enabled);

    /// Check whether shape bucketing is currently enabled.
    /// @return  True if bucketing is enabled, false otherwise.
    bool bucketing_enabled() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
