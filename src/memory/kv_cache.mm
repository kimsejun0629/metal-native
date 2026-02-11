/// @file kv_cache.mm
/// @brief Implementation of KV cache for autoregressive generation.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/memory/kv_cache.h"
#include "metal_native/core/device.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/error.h"
#include "metal_native/memory/budget_controller.h"
#include "metal_native/dispatch/command_pipeline.h"

#include <mutex>
#include <vector>
#include <algorithm>

namespace metal_native {

// ---------------------------------------------------------------------------
// KVCache::Impl
// ---------------------------------------------------------------------------

struct KVCache::Impl {
    KVCacheConfig config;
    MNDevice& device;

    std::vector<id<MTLBuffer>> key_buffers;
    std::vector<id<MTLBuffer>> value_buffers;

    size_t current_seq_len_ = 0;
    size_t actual_max_seq_len;

    std::mutex mu;

    Impl(const KVCacheConfig& cfg, MNDevice& dev)
        : config(cfg), device(dev), actual_max_seq_len(cfg.max_seq_len) {}

    size_t element_size() const {
        switch (config.dtype) {
            case MNDType::Float32: return 4;
            case MNDType::Float16: return 2;
            case MNDType::BFloat16: return 2;
            default:
                MN_THROW(MetalNativeError::InvalidArgument,
                         "Unsupported dtype for KV cache");
        }
    }

    size_t buffer_size_per_layer() const {
        return actual_max_seq_len * config.num_heads * config.head_dim * element_size();
    }

    size_t total_memory() const {
        return 2 * config.num_layers * buffer_size_per_layer();
    }
};

// ---------------------------------------------------------------------------
// KVCache public API
// ---------------------------------------------------------------------------

KVCache::KVCache(const KVCacheConfig& config, MNDevice& device)
    : impl_(std::make_unique<Impl>(config, device)) {

    MN_CHECK(config.num_layers > 0 && config.num_heads > 0 &&
             config.head_dim > 0 && config.max_seq_len > 0,
             MetalNativeError::InvalidArgument,
             "KVCache: all config dimensions must be > 0");

    // Request budget from controller
    size_t desired_bytes = impl_->total_memory();
    auto grant = MemoryBudgetController::instance().request_budget(
        BudgetStrategy::KVCache, desired_bytes);

    if (!grant.approved || grant.bytes < desired_bytes) {
        // Reduce max_seq_len proportionally
        if (grant.bytes > 0) {
            size_t bytes_per_token = desired_bytes / config.max_seq_len;
            impl_->actual_max_seq_len = grant.bytes / bytes_per_token;
            impl_->actual_max_seq_len = std::max<size_t>(impl_->actual_max_seq_len, 1);
        } else {
            // Fallback: try to allocate at least 128 tokens
            impl_->actual_max_seq_len = std::min<size_t>(config.max_seq_len, 128);
        }
    }

    // Pre-allocate buffers
    size_t buffer_size = impl_->buffer_size_per_layer();
    id<MTLDevice> mtl_device = device.metal_device();

    impl_->key_buffers.reserve(config.num_layers);
    impl_->value_buffers.reserve(config.num_layers);

    for (size_t layer = 0; layer < config.num_layers; ++layer) {
        id<MTLBuffer> key_buf = [mtl_device newBufferWithLength:buffer_size
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> val_buf = [mtl_device newBufferWithLength:buffer_size
                                                        options:MTLResourceStorageModeShared];

        MN_CHECK(key_buf != nil && val_buf != nil,
                 MetalNativeError::AllocationFailed,
                 "KVCache: failed to allocate buffer for layer " +
                 std::to_string(layer));

        impl_->key_buffers.push_back(key_buf);
        impl_->value_buffers.push_back(val_buf);
    }
}

KVCache::~KVCache() {
    // Release all MTLBuffers
    if (impl_) {
        for (auto& buf : impl_->key_buffers) {
            buf = nil;
        }
        for (auto& buf : impl_->value_buffers) {
            buf = nil;
        }
    }
}

void KVCache::append(size_t layer, const MNTensor& new_key,
                     const MNTensor& new_value, size_t position) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    MN_CHECK(layer < impl_->config.num_layers,
             MetalNativeError::InvalidArgument,
             "KVCache::append: layer index out of bounds");

    MN_CHECK(position < impl_->actual_max_seq_len,
             MetalNativeError::InvalidArgument,
             "KVCache::append: position exceeds max_seq_len");

    // Get source buffers
    id<MTLBuffer> src_key_buf = new_key.buffer()->metal_buffer();
    id<MTLBuffer> src_val_buf = new_value.buffer()->metal_buffer();

    // Get destination buffers
    id<MTLBuffer> dst_key_buf = impl_->key_buffers[layer];
    id<MTLBuffer> dst_val_buf = impl_->value_buffers[layer];

    // Calculate offsets and size
    size_t elem_size = impl_->element_size();
    size_t slice_size = impl_->config.num_heads * impl_->config.head_dim * elem_size;
    size_t dst_offset = position * slice_size;
    size_t src_offset = new_key.offset();

    // Use blit encoder to copy data
    CommandPipeline& pipeline = impl_->device.command_pipeline();
    id<MTLCommandBuffer> cmd = pipeline.current_buffer();
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];

    [blit copyFromBuffer:src_key_buf
            sourceOffset:src_offset
                toBuffer:dst_key_buf
       destinationOffset:dst_offset
                    size:slice_size];

    size_t src_val_offset = new_value.offset();
    [blit copyFromBuffer:src_val_buf
            sourceOffset:src_val_offset
                toBuffer:dst_val_buf
       destinationOffset:dst_offset
                    size:slice_size];

    [blit endEncoding];
    pipeline.commit();

    // Update current_seq_len
    impl_->current_seq_len_ = std::max(impl_->current_seq_len_, position + 1);
}

MNTensor KVCache::get_key(size_t layer) const {
    std::lock_guard<std::mutex> lock(impl_->mu);

    MN_CHECK(layer < impl_->config.num_layers,
             MetalNativeError::InvalidArgument,
             "KVCache::get_key: layer index out of bounds");

    // Create a tensor by copying from the cache buffer
    // Shape: [1, num_heads, current_seq_len, head_dim]
    MNShape shape({1,
                   static_cast<int64_t>(impl_->config.num_heads),
                   static_cast<int64_t>(impl_->current_seq_len_),
                   static_cast<int64_t>(impl_->config.head_dim)});

    MNTensor result = MNTensor::empty(shape, impl_->config.dtype, impl_->device);

    // Copy data from cache to result
    id<MTLBuffer> src_buf = impl_->key_buffers[layer];
    id<MTLBuffer> dst_buf = result.buffer()->metal_buffer();

    size_t copy_size = impl_->current_seq_len_ * impl_->config.num_heads *
                       impl_->config.head_dim * impl_->element_size();

    CommandPipeline& pipeline = impl_->device.command_pipeline();
    id<MTLCommandBuffer> cmd = pipeline.current_buffer();
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];

    [blit copyFromBuffer:src_buf
            sourceOffset:0
                toBuffer:dst_buf
       destinationOffset:0
                    size:copy_size];

    [blit endEncoding];
    pipeline.commit();

    return result;
}

MNTensor KVCache::get_value(size_t layer) const {
    std::lock_guard<std::mutex> lock(impl_->mu);

    MN_CHECK(layer < impl_->config.num_layers,
             MetalNativeError::InvalidArgument,
             "KVCache::get_value: layer index out of bounds");

    // Create a tensor by copying from the cache buffer
    // Shape: [1, num_heads, current_seq_len, head_dim]
    MNShape shape({1,
                   static_cast<int64_t>(impl_->config.num_heads),
                   static_cast<int64_t>(impl_->current_seq_len_),
                   static_cast<int64_t>(impl_->config.head_dim)});

    MNTensor result = MNTensor::empty(shape, impl_->config.dtype, impl_->device);

    // Copy data from cache to result
    id<MTLBuffer> src_buf = impl_->value_buffers[layer];
    id<MTLBuffer> dst_buf = result.buffer()->metal_buffer();

    size_t copy_size = impl_->current_seq_len_ * impl_->config.num_heads *
                       impl_->config.head_dim * impl_->element_size();

    CommandPipeline& pipeline = impl_->device.command_pipeline();
    id<MTLCommandBuffer> cmd = pipeline.current_buffer();
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];

    [blit copyFromBuffer:src_buf
            sourceOffset:0
                toBuffer:dst_buf
       destinationOffset:0
                    size:copy_size];

    [blit endEncoding];
    pipeline.commit();

    return result;
}

size_t KVCache::current_seq_len() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->current_seq_len_;
}

void KVCache::resize(size_t new_max_seq_len) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    if (new_max_seq_len >= impl_->actual_max_seq_len) {
        return; // No-op if growing or same size
    }

    // Ensure we don't shrink below current data
    new_max_seq_len = std::max(new_max_seq_len, impl_->current_seq_len_);

    size_t elem_size = impl_->element_size();
    size_t new_buffer_size = new_max_seq_len * impl_->config.num_heads *
                             impl_->config.head_dim * elem_size;

    id<MTLDevice> mtl_device = impl_->device.metal_device();

    // Allocate new buffers and copy existing data
    for (size_t layer = 0; layer < impl_->config.num_layers; ++layer) {
        id<MTLBuffer> old_key = impl_->key_buffers[layer];
        id<MTLBuffer> old_val = impl_->value_buffers[layer];

        id<MTLBuffer> new_key = [mtl_device newBufferWithLength:new_buffer_size
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> new_val = [mtl_device newBufferWithLength:new_buffer_size
                                                        options:MTLResourceStorageModeShared];

        MN_CHECK(new_key != nil && new_val != nil,
                 MetalNativeError::AllocationFailed,
                 "KVCache::resize: failed to allocate new buffer");

        // Copy existing data
        size_t copy_size = impl_->current_seq_len_ * impl_->config.num_heads *
                          impl_->config.head_dim * elem_size;

        if (copy_size > 0) {
            CommandPipeline& pipeline = impl_->device.command_pipeline();
            id<MTLCommandBuffer> cmd = pipeline.current_buffer();
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];

            [blit copyFromBuffer:old_key sourceOffset:0
                        toBuffer:new_key destinationOffset:0
                            size:copy_size];

            [blit copyFromBuffer:old_val sourceOffset:0
                        toBuffer:new_val destinationOffset:0
                            size:copy_size];

            [blit endEncoding];
            pipeline.commit();
        }

        // Replace buffers
        impl_->key_buffers[layer] = new_key;
        impl_->value_buffers[layer] = new_val;
    }

    impl_->actual_max_seq_len = new_max_seq_len;
}

void KVCache::reset() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->current_seq_len_ = 0;
}

size_t KVCache::memory_footprint() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->total_memory();
}

} // namespace metal_native
