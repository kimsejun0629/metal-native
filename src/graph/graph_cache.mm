/// @file graph_cache.mm
/// @brief Objective-C++ implementation of GraphCache.

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>

#include "metal_native/graph/graph_cache.h"
#include "metal_native/graph/graph_serializer.h"
#include "metal_native/graph/shape_bucketing.h"
#include "metal_native/core/error.h"

#include <chrono>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace metal_native {

// ---------------------------------------------------------------------------
// GraphCacheKey implementation
// ---------------------------------------------------------------------------

bool GraphCacheKey::operator==(const GraphCacheKey& other) const {
    return topology_hash == other.topology_hash &&
           shape_tuple == other.shape_tuple &&
           dtype == other.dtype;
}

// ---------------------------------------------------------------------------
// GraphCacheStats implementation
// ---------------------------------------------------------------------------

double GraphCacheStats::l1_hit_rate() const {
    uint64_t total = l1_hits + l1_misses;
    if (total == 0) return 0.0;
    return static_cast<double>(l1_hits) / static_cast<double>(total);
}

double GraphCacheStats::overall_hit_rate() const {
    uint64_t total_hits = l1_hits + l2_hits;
    uint64_t total_accesses = l1_hits + l1_misses;
    if (total_accesses == 0) return 0.0;
    return static_cast<double>(total_hits) / static_cast<double>(total_accesses);
}

} // namespace metal_native

// ---------------------------------------------------------------------------
// Hash function for GraphCacheKey
// ---------------------------------------------------------------------------

namespace std {
size_t hash<metal_native::GraphCacheKey>::operator()(
    const metal_native::GraphCacheKey& key) const noexcept {
    size_t h = std::hash<uint64_t>{}(key.topology_hash);

    // Hash the shape tuple.
    for (size_t val : key.shape_tuple) {
        h ^= std::hash<size_t>{}(val) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }

    // Mix in the dtype.
    h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(key.dtype)) + 0x9e3779b9 + (h << 6) + (h >> 2);

    return h;
}
} // namespace std

namespace metal_native {

// ---------------------------------------------------------------------------
// L1 Cache Entry
// ---------------------------------------------------------------------------

struct CacheEntry {
    GraphCacheKey key;
    MPSGraphExecutable* executable;
    std::chrono::steady_clock::time_point last_access_time;

    CacheEntry(const GraphCacheKey& k, MPSGraphExecutable* exec)
        : key(k), executable(exec), last_access_time(std::chrono::steady_clock::now()) {}
};

// ---------------------------------------------------------------------------
// Impl -- LRU cache + disk persistence
// ---------------------------------------------------------------------------

struct GraphCache::Impl {
    // L1: in-memory LRU cache
    using EntryList = std::list<CacheEntry>;
    EntryList order;
    std::unordered_map<GraphCacheKey, EntryList::iterator> map;
    size_t max_l1_entries = 128;

    // L2: disk cache directory
    std::string cache_dir;

    // Statistics
    uint64_t l1_hits   = 0;
    uint64_t l1_misses = 0;
    uint64_t l2_hits   = 0;
    uint64_t l2_misses = 0;

    // Thread safety
    mutable std::mutex mu;

    // Eviction timer
    dispatch_source_t eviction_timer = nil;
    dispatch_queue_t timer_queue = nil;

    // Shape bucketing
    ShapeBucketer bucketer_;
    bool bucketing_enabled_ = false;

    Impl() {
        // Set up cache directory: ~/.cache/metal_native/graph_cache/
        @autoreleasepool {
            NSArray<NSString*>* paths = NSSearchPathForDirectoriesInDomains(
                NSCachesDirectory, NSUserDomainMask, YES);
            if (paths.count > 0) {
                NSString* base_cache = paths[0];
                NSString* mn_cache = [base_cache stringByAppendingPathComponent:@"metal_native/graph_cache"];
                cache_dir = [mn_cache UTF8String];

                // Create directory if it doesn't exist
                NSError* error = nil;
                [[NSFileManager defaultManager] createDirectoryAtPath:mn_cache
                                          withIntermediateDirectories:YES
                                                           attributes:nil
                                                                error:&error];
                if (error) {
                    MN_THROW(MetalNativeError::InternalError,
                             "GraphCache: failed to create cache directory");
                }
            } else {
                MN_THROW(MetalNativeError::InternalError,
                         "GraphCache: could not determine cache directory");
            }
        }
    }

    ~Impl() {
        if (eviction_timer != nil) {
            dispatch_source_cancel(eviction_timer);
            // No dispatch_release under ARC -- GCD objects are ARC-managed.
        }
    }

    // Generate disk cache path for a key
    std::string disk_path(const GraphCacheKey& key) const {
        size_t full_hash = std::hash<GraphCacheKey>{}(key);
        char hex[17];
        snprintf(hex, sizeof(hex), "%016zx", full_hash);
        return cache_dir + "/" + hex + ".mpsgraphpackage";
    }

    // Apply shape bucketing to a key (returns a copy with bucketed shapes)
    GraphCacheKey apply_bucketing(const GraphCacheKey& key) const {
        if (!bucketing_enabled_) {
            return key;
        }

        GraphCacheKey bucketed_key = key;

        // Convert shape_tuple to MNShape
        std::vector<int64_t> shape_dims;
        shape_dims.reserve(key.shape_tuple.size());
        for (size_t dim : key.shape_tuple) {
            shape_dims.push_back(static_cast<int64_t>(dim));
        }
        MNShape original(shape_dims);

        // Bucket the shape
        MNShape bucketed = bucketer_.bucket_shape(original);

        // Convert back to shape_tuple
        bucketed_key.shape_tuple.clear();
        bucketed_key.shape_tuple.reserve(bucketed.ndim());
        for (size_t i = 0; i < bucketed.ndim(); ++i) {
            bucketed_key.shape_tuple.push_back(static_cast<size_t>(bucketed[static_cast<int64_t>(i)]));
        }

        return bucketed_key;
    }
};

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

GraphCache::GraphCache(size_t max_l1_entries)
    : impl_(std::make_unique<Impl>()) {
    impl_->max_l1_entries = (max_l1_entries > 0) ? max_l1_entries : 1;
}

GraphCache::~GraphCache() = default;

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

MPSGraphExecutable* GraphCache::lookup(const GraphCacheKey& key) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    // Apply bucketing if enabled
    GraphCacheKey effective_key = impl_->apply_bucketing(key);

    // Check L1 cache first
    auto it = impl_->map.find(effective_key);
    if (it != impl_->map.end()) {
        // L1 hit: move to front and update access time
        auto& entry = *it->second;
        entry.last_access_time = std::chrono::steady_clock::now();
        impl_->order.splice(impl_->order.begin(), impl_->order, it->second);
        ++impl_->l1_hits;
        return entry.executable;
    }

    ++impl_->l1_misses;

    // Check L2 (disk) cache
    std::string path = impl_->disk_path(effective_key);

    @autoreleasepool {
        NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
        if ([[NSFileManager defaultManager] fileExistsAtPath:ns_path]) {
            // Validate cache entry
            if (!GraphSerializer::validate_cache_entry(path)) {
                ++impl_->l2_misses;
                return nil;
            }

            // Load from disk
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            MPSGraphExecutable* executable = GraphSerializer::deserialize(path, device);

            if (executable != nil) {
                // L2 hit: promote to L1
                ++impl_->l2_hits;

                // Insert into L1 (without disk write)
                if (impl_->map.size() >= impl_->max_l1_entries) {
                    auto& back = impl_->order.back();
                    impl_->map.erase(back.key);
                    impl_->order.pop_back();
                }

                impl_->order.emplace_front(effective_key, executable);
                impl_->map[effective_key] = impl_->order.begin();

                return executable;
            }
        }
    }

    ++impl_->l2_misses;
    return nil;
}

// ---------------------------------------------------------------------------
// Insertion
// ---------------------------------------------------------------------------

void GraphCache::insert(const GraphCacheKey& key, MPSGraphExecutable* executable) {
    MN_CHECK(executable != nil,
             MetalNativeError::InvalidArgument,
             "GraphCache::insert: executable must not be nil");

    std::lock_guard<std::mutex> lock(impl_->mu);

    // Apply bucketing if enabled
    GraphCacheKey effective_key = impl_->apply_bucketing(key);

    // Check if already in L1
    auto it = impl_->map.find(effective_key);
    if (it != impl_->map.end()) {
        // Update existing entry and promote to front
        auto& entry = *it->second;
        entry.executable = executable;
        entry.last_access_time = std::chrono::steady_clock::now();
        impl_->order.splice(impl_->order.begin(), impl_->order, it->second);
        return;
    }

    // Evict LRU entry if at capacity
    if (impl_->map.size() >= impl_->max_l1_entries) {
        auto& back = impl_->order.back();
        impl_->map.erase(back.key);
        impl_->order.pop_back();
    }

    // Insert new entry at front
    impl_->order.emplace_front(effective_key, executable);
    impl_->map[effective_key] = impl_->order.begin();

    // Asynchronously serialize to L2
    std::string path = impl_->disk_path(effective_key);
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_LOW, 0), ^{
        @autoreleasepool {
            try {
                GraphSerializer::serialize(executable, path);
            } catch (...) {
                // Silently ignore serialization errors
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Eviction timer
// ---------------------------------------------------------------------------

void GraphCache::start_eviction_timer(unsigned int interval_seconds) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    if (impl_->eviction_timer != nil) {
        return; // Timer already running
    }

    impl_->timer_queue = dispatch_queue_create(
        "com.metal_native.graph_cache.eviction",
        DISPATCH_QUEUE_SERIAL);

    impl_->eviction_timer = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_TIMER, 0, 0, impl_->timer_queue);

    dispatch_source_set_timer(
        impl_->eviction_timer,
        dispatch_time(DISPATCH_TIME_NOW, interval_seconds * NSEC_PER_SEC),
        interval_seconds * NSEC_PER_SEC,
        (interval_seconds * NSEC_PER_SEC) / 10);

    // Capture impl pointer for timer callback
    auto impl_ptr = impl_.get();

    dispatch_source_set_event_handler(impl_->eviction_timer, ^{
        std::lock_guard<std::mutex> timer_lock(impl_ptr->mu);

        auto now = std::chrono::steady_clock::now();
        auto threshold = std::chrono::seconds(60);

        // Evict entries unused for > 60 seconds
        auto it = impl_ptr->order.begin();
        while (it != impl_ptr->order.end()) {
            auto age = std::chrono::duration_cast<std::chrono::seconds>(
                now - it->last_access_time);

            if (age > threshold) {
                impl_ptr->map.erase(it->key);
                it = impl_ptr->order.erase(it);
            } else {
                ++it;
            }
        }
    });

    dispatch_resume(impl_->eviction_timer);
}

void GraphCache::stop_eviction_timer() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    if (impl_->eviction_timer != nil) {
        dispatch_source_cancel(impl_->eviction_timer);
        impl_->eviction_timer = nil;
    }

    // timer_queue is ARC-managed; just nil it out.
    impl_->timer_queue = nil;
}

// ---------------------------------------------------------------------------
// Cache management
// ---------------------------------------------------------------------------

void GraphCache::clear() {
    clear_l1();
    clear_l2();
}

void GraphCache::clear_l1() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->order.clear();
    impl_->map.clear();
    impl_->l1_hits = 0;
    impl_->l1_misses = 0;
    impl_->l2_hits = 0;
    impl_->l2_misses = 0;
}

void GraphCache::clear_l2() {
    @autoreleasepool {
        NSString* cache_dir_ns = [NSString stringWithUTF8String:impl_->cache_dir.c_str()];
        NSError* error = nil;
        NSArray<NSString*>* files = [[NSFileManager defaultManager]
            contentsOfDirectoryAtPath:cache_dir_ns error:&error];

        if (!error) {
            for (NSString* file in files) {
                NSString* full_path = [cache_dir_ns stringByAppendingPathComponent:file];
                [[NSFileManager defaultManager] removeItemAtPath:full_path error:nil];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

GraphCacheStats GraphCache::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    GraphCacheStats s;
    s.l1_hits = impl_->l1_hits;
    s.l1_misses = impl_->l1_misses;
    s.l2_hits = impl_->l2_hits;
    s.l2_misses = impl_->l2_misses;
    return s;
}

size_t GraphCache::l1_size() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->map.size();
}

size_t GraphCache::l1_capacity() const {
    return impl_->max_l1_entries;
}

void GraphCache::set_l1_capacity(size_t new_max_entries) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->max_l1_entries = (new_max_entries > 0) ? new_max_entries : 1;
    // Evict excess LRU entries
    while (impl_->map.size() > impl_->max_l1_entries) {
        auto& back = impl_->order.back();
        impl_->map.erase(back.key);
        impl_->order.pop_back();
    }
}

size_t GraphCache::memory_footprint() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    // Rough estimate: each MPSGraphExecutable is ~2-50KB depending on graph complexity.
    // Use a conservative 32KB average plus the CacheEntry overhead.
    constexpr size_t kAvgExecutableSize = 32 * 1024;
    constexpr size_t kEntryOverhead = sizeof(GraphCacheKey) + 64; // pointers, timestamps
    return impl_->map.size() * (kAvgExecutableSize + kEntryOverhead);
}

// ---------------------------------------------------------------------------
// Shape bucketing
// ---------------------------------------------------------------------------

void GraphCache::set_bucketing_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->bucketing_enabled_ = enabled;
}

bool GraphCache::bucketing_enabled() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->bucketing_enabled_;
}

} // namespace metal_native
