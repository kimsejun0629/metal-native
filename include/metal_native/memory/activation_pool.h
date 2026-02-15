#pragma once

/// @file activation_pool.h
/// @brief Transformer-specific activation memory pool for buffer reuse.
///
/// ActivationPool pre-allocates and reuses GPU buffers for intermediate
/// activations during transformer inference.  By caching buffers of known
/// sizes (QKV projections, attention scores, FFN intermediates) instead of
/// allocating/freeing on every layer, peak memory usage and allocation
/// latency are both reduced.
///
/// Thread-safe: all public methods are guarded by a mutex.

#include <cstddef>
#include <cstdint>
#include <memory>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace metal_native {

class MNDevice;

/// Activation memory pool for transformer inference.
/// Pre-allocates and reuses GPU buffers for intermediate activations.
///
/// Usage:
///   auto pool = ActivationPool(device, max_budget_bytes);
///   pool.configure_for_model(hidden_dim, num_heads, max_seq_len, batch_size);
///
///   // During inference:
///   auto buf = pool.acquire(size_needed, "qkv_proj");
///   // ... use buf ...
///   pool.release(buf);  // Returns to pool, NOT freed
///
/// Thread-safe: all public methods are guarded by a mutex.
class ActivationPool {
public:
    struct BufferHandle {
        void* metal_buffer;  // id<MTLBuffer> cast to void*
        size_t size;
        uint32_t pool_index;
    };

    explicit ActivationPool(MNDevice& device, size_t max_budget_bytes = 0);
    ~ActivationPool();

    // Non-copyable
    ActivationPool(const ActivationPool&) = delete;
    ActivationPool& operator=(const ActivationPool&) = delete;

    /// Pre-allocate buffers for a known model configuration
    void configure_for_model(size_t hidden_dim, size_t num_heads,
                            size_t max_seq_len, size_t batch_size,
                            size_t num_layers = 1);

    /// Acquire a buffer of at least `size` bytes
    /// Returns a handle that must be released back to the pool
    /// tag is optional, for debugging/profiling
    BufferHandle acquire(size_t size, const char* tag = nullptr);

    /// Release a buffer back to the pool
    void release(const BufferHandle& handle);

    /// Release all acquired buffers (e.g., between inference passes)
    void release_all();

    /// Purge unused buffers to reduce memory footprint
    /// Called automatically when memory pressure increases
    void purge_unused();

    // Stats
    size_t total_allocated() const noexcept;
    size_t total_in_use() const noexcept;
    size_t pool_size() const noexcept;
    size_t hit_count() const noexcept;
    size_t miss_count() const noexcept;
    float hit_rate() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
