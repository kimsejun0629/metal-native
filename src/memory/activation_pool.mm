/// @file activation_pool.mm
/// @brief Objective-C++ implementation of ActivationPool.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/memory/activation_pool.h"
#include "metal_native/memory/budget_controller.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

#include <algorithm>
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

/// Standard size classes (power-of-2).
static constexpr size_t kSizeClasses[] = {
    256  * 1024,  //   256 KB
    512  * 1024,  //   512 KB
    1024 * 1024,  //     1 MB
    2048 * 1024,  //     2 MB
    4096 * 1024,  //     4 MB
    8192 * 1024,  //     8 MB
    16384 * 1024, //    16 MB
    32768 * 1024, //    32 MB
};
static constexpr size_t kNumSizeClasses = sizeof(kSizeClasses) / sizeof(kSizeClasses[0]);

/// Maximum waste tolerance (25%).
/// If best-fit size class wastes more than 25%, allocate exact size instead.
static constexpr float kMaxWasteFraction = 0.25f;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the smallest size class >= requested size.
/// Returns SIZE_MAX if no size class fits.
static size_t find_size_class(size_t requested, const std::vector<size_t>& classes) {
    for (size_t sc : classes) {
        if (sc >= requested) {
            return sc;
        }
    }
    return SIZE_MAX;
}

/// Check if using a size class would waste too much memory.
static bool excessive_waste(size_t requested, size_t size_class) {
    if (size_class == requested) return false;
    float waste_fraction = static_cast<float>(size_class - requested) / static_cast<float>(size_class);
    return waste_fraction > kMaxWasteFraction;
}

// ---------------------------------------------------------------------------
// PooledBuffer -- a free-list entry
// ---------------------------------------------------------------------------

struct PooledBuffer {
    id<MTLBuffer> buffer;
    size_t size;
    uint32_t index;
    bool in_use;

    PooledBuffer(id<MTLBuffer> buf, size_t sz, uint32_t idx)
        : buffer(buf), size(sz), index(idx), in_use(false) {}
};

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct ActivationPool::Impl {
    MNDevice& device;
    size_t max_budget;
    mutable std::mutex mu;

    /// All size classes (standard + model-specific).
    std::vector<size_t> size_classes;

    /// Free list per size class.
    std::unordered_map<size_t, std::vector<uint32_t>> free_lists;

    /// All buffers (both free and in-use).
    std::vector<PooledBuffer> buffers;

    /// Next buffer index.
    uint32_t next_index = 0;

    // Statistics
    size_t total_allocated_bytes = 0;
    size_t total_in_use_bytes = 0;
    size_t hit_count = 0;
    size_t miss_count = 0;

    explicit Impl(MNDevice& dev, size_t budget)
        : device(dev), max_budget(budget) {
        // Initialize with standard size classes
        size_classes.assign(kSizeClasses, kSizeClasses + kNumSizeClasses);
        std::sort(size_classes.begin(), size_classes.end());
    }

    ~Impl() {
        // Release all buffers
        for (auto& pb : buffers) {
            if (pb.buffer) {
                [pb.buffer setPurgeableState:MTLPurgeableStateEmpty];
                pb.buffer = nil;
            }
        }
    }

    /// Add model-specific size classes.
    void add_model_size_classes(const std::vector<size_t>& model_sizes) {
        std::lock_guard<std::mutex> lock(mu);

        for (size_t sz : model_sizes) {
            // Only add if not already present
            if (std::find(size_classes.begin(), size_classes.end(), sz) == size_classes.end()) {
                size_classes.push_back(sz);
            }
        }

        // Keep sorted
        std::sort(size_classes.begin(), size_classes.end());
    }

    /// Allocate a new Metal buffer.
    id<MTLBuffer> allocate_buffer(size_t size) {
        id<MTLDevice> mtl_device = device.metal_device();

        MTLResourceOptions options = MTLResourceStorageModePrivate;
        id<MTLBuffer> buffer = [mtl_device newBufferWithLength:size options:options];

        if (!buffer) {
            throw MNException(MetalNativeError::AllocationFailed,
                            "Failed to allocate Metal buffer of size " + std::to_string(size));
        }

        return buffer;
    }

    /// Find a free buffer from the pool.
    PooledBuffer* find_free_buffer(size_t size_class) {
        auto it = free_lists.find(size_class);
        if (it == free_lists.end() || it->second.empty()) {
            return nullptr;
        }

        uint32_t idx = it->second.back();
        it->second.pop_back();

        PooledBuffer& pb = buffers[idx];
        pb.in_use = true;

        return &pb;
    }

    /// Create a new buffer and add to pool.
    PooledBuffer* create_buffer(size_t size) {
        // Check budget
        if (max_budget > 0 && total_allocated_bytes + size > max_budget) {
            // Try to purge unused first
            purge_unused_internal();

            // Still over budget?
            if (total_allocated_bytes + size > max_budget) {
                throw MNException(MetalNativeError::OutOfMemory,
                                "ActivationPool budget exceeded: " +
                                std::to_string(total_allocated_bytes + size) +
                                " > " + std::to_string(max_budget));
            }
        }

        id<MTLBuffer> buffer = allocate_buffer(size);

        uint32_t idx = next_index++;
        buffers.emplace_back(buffer, size, idx);

        PooledBuffer& pb = buffers.back();
        pb.in_use = true;

        total_allocated_bytes += size;

        return &pb;
    }

    /// Purge unused buffers (internal, assumes lock held).
    void purge_unused_internal() {
        size_t freed = 0;

        for (auto& pb : buffers) {
            if (!pb.in_use && pb.buffer) {
                [pb.buffer setPurgeableState:MTLPurgeableStateEmpty];
                freed += pb.size;
                pb.buffer = nil;
            }
        }

        // Remove from free lists
        for (auto& [sc, indices] : free_lists) {
            indices.erase(
                std::remove_if(indices.begin(), indices.end(),
                    [this](uint32_t idx) { return buffers[idx].buffer == nil; }),
                indices.end()
            );
        }

        total_allocated_bytes -= freed;
    }
};

// ---------------------------------------------------------------------------
// ActivationPool
// ---------------------------------------------------------------------------

ActivationPool::ActivationPool(MNDevice& device, size_t max_budget_bytes)
    : impl_(std::make_unique<Impl>(device, max_budget_bytes)) {

    // If budget not specified, use recommended working set size
    if (max_budget_bytes == 0) {
        impl_->max_budget = device.recommended_max_working_set_size();
    }
}

ActivationPool::~ActivationPool() = default;

void ActivationPool::configure_for_model(size_t hidden_dim, size_t num_heads,
                                         size_t max_seq_len, size_t batch_size,
                                         size_t num_layers) {
    // Calculate transformer activation sizes
    std::vector<size_t> model_sizes;

    // QKV projection: [batch, seq_len, 3 * hidden_dim]
    size_t qkv_size = batch_size * max_seq_len * 3 * hidden_dim * sizeof(float);
    model_sizes.push_back(qkv_size);

    // Attention scores: [batch, num_heads, seq_len, seq_len]
    size_t attn_scores_size = batch_size * num_heads * max_seq_len * max_seq_len * sizeof(float);
    model_sizes.push_back(attn_scores_size);

    // Attention output: [batch, seq_len, hidden_dim]
    size_t attn_out_size = batch_size * max_seq_len * hidden_dim * sizeof(float);
    model_sizes.push_back(attn_out_size);

    // FFN intermediate: [batch, seq_len, 4 * hidden_dim]
    size_t ffn_size = batch_size * max_seq_len * 4 * hidden_dim * sizeof(float);
    model_sizes.push_back(ffn_size);

    // Add model-specific size classes
    impl_->add_model_size_classes(model_sizes);

    // Pre-allocate buffers for one layer
    // This ensures first inference has warm pool
    std::lock_guard<std::mutex> lock(impl_->mu);

    for (size_t size : model_sizes) {
        // Allocate 2 buffers per size (for double buffering)
        for (int i = 0; i < 2; ++i) {
            try {
                auto* pb = impl_->create_buffer(size);
                pb->in_use = false;
                impl_->free_lists[size].push_back(pb->index);
            } catch (const MNException& e) {
                // Pre-allocation failure is not fatal
                // Will allocate on-demand during inference
                break;
            }
        }
    }
}

ActivationPool::BufferHandle ActivationPool::acquire(size_t size, const char* tag) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    if (size < kMinAllocSize) {
        size = kMinAllocSize;
    }

    // Find best-fit size class
    size_t size_class = find_size_class(size, impl_->size_classes);

    PooledBuffer* pb = nullptr;

    // Try to reuse from pool
    if (size_class != SIZE_MAX && !excessive_waste(size, size_class)) {
        pb = impl_->find_free_buffer(size_class);
        if (pb) {
            impl_->hit_count++;
            impl_->total_in_use_bytes += pb->size;

            // Make buffer non-purgeable
            [pb->buffer setPurgeableState:MTLPurgeableStateNonVolatile];

            return BufferHandle{
                .metal_buffer = (__bridge void*)pb->buffer,
                .size = pb->size,
                .pool_index = pb->index
            };
        }
    }

    // Pool miss - allocate new buffer
    impl_->miss_count++;

    // Use exact size if no good size class or excessive waste
    size_t alloc_size = (size_class != SIZE_MAX && !excessive_waste(size, size_class))
                        ? size_class : size;

    pb = impl_->create_buffer(alloc_size);
    impl_->total_in_use_bytes += pb->size;

    return BufferHandle{
        .metal_buffer = (__bridge void*)pb->buffer,
        .size = pb->size,
        .pool_index = pb->index
    };
}

void ActivationPool::release(const BufferHandle& handle) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    if (handle.pool_index >= impl_->buffers.size()) {
        throw MNException(MetalNativeError::InvalidArgument,
                        "Invalid BufferHandle: index out of range");
    }

    PooledBuffer& pb = impl_->buffers[handle.pool_index];

    if (!pb.in_use) {
        throw MNException(MetalNativeError::InvalidArgument,
                        "Double release of BufferHandle");
    }

    pb.in_use = false;
    impl_->total_in_use_bytes -= pb.size;

    // Return to free list
    impl_->free_lists[pb.size].push_back(pb.index);

    // Mark as purgeable (system can reclaim if needed)
    [pb.buffer setPurgeableState:MTLPurgeableStateVolatile];
}

void ActivationPool::release_all() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    for (auto& pb : impl_->buffers) {
        if (pb.in_use) {
            pb.in_use = false;
            impl_->free_lists[pb.size].push_back(pb.index);

            // Mark as purgeable
            if (pb.buffer) {
                [pb.buffer setPurgeableState:MTLPurgeableStateVolatile];
            }
        }
    }

    impl_->total_in_use_bytes = 0;
}

void ActivationPool::purge_unused() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->purge_unused_internal();
}

size_t ActivationPool::total_allocated() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->total_allocated_bytes;
}

size_t ActivationPool::total_in_use() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->total_in_use_bytes;
}

size_t ActivationPool::pool_size() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->buffers.size();
}

size_t ActivationPool::hit_count() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->hit_count;
}

size_t ActivationPool::miss_count() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->miss_count;
}

float ActivationPool::hit_rate() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mu);
    size_t total = impl_->hit_count + impl_->miss_count;
    if (total == 0) return 0.0f;
    return static_cast<float>(impl_->hit_count) / static_cast<float>(total);
}

} // namespace metal_native
