/// @file icb_recorder.mm
/// @brief Implementation of ICBRecorder for indirect command buffer recording/replay.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/dispatch/icb_recorder.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

#include <mutex>
#include <unordered_map>
#include <vector>
#include <list>

namespace metal_native {

// ---------------------------------------------------------------------------
// Recorded dispatch info (for building ICB)
// ---------------------------------------------------------------------------

struct RecordedDispatch {
    id<MTLComputePipelineState> pipeline;
    std::vector<id<MTLBuffer>> buffers;
    std::vector<uint32_t> buffer_offsets;
    std::vector<uint8_t> bytes_data;
    uint32_t bytes_index;
    MTLSize grid_size;
    MTLSize threadgroup_size;
    std::vector<uint32_t> threadgroup_memory_lengths;
};

// ---------------------------------------------------------------------------
// Cached ICB entry
// ---------------------------------------------------------------------------

struct CachedICB {
    id<MTLIndirectCommandBuffer> icb;
    uint32_t command_count;
    ICBShapeKey key;

    // Store pipeline states and buffer references to keep them alive
    std::vector<id<MTLComputePipelineState>> pipelines;
    std::vector<id<MTLBuffer>> buffers;
};

// ---------------------------------------------------------------------------
// ICBRecorder::Impl
// ---------------------------------------------------------------------------

struct ICBRecorder::Impl {
    mutable std::mutex mu;
    bool enabled = false;

    // LRU cache: key -> cached ICB
    std::unordered_map<ICBShapeKey, std::shared_ptr<CachedICB>, ICBShapeKeyHash> cache;
    std::list<ICBShapeKey> lru_order;  // front = most recently used

    // Recording state
    bool is_recording = false;
    ICBShapeKey recording_key;
    std::vector<RecordedDispatch> recorded_dispatches;

    void evict_if_needed() {
        while (cache.size() >= ICBRecorder::MAX_CACHE_SIZE && !lru_order.empty()) {
            auto oldest = lru_order.back();
            cache.erase(oldest);
            lru_order.pop_back();
        }
    }

    void touch(const ICBShapeKey& key) {
        lru_order.remove(key);
        lru_order.push_front(key);
    }

    // NOTE: MTLIndirectComputeCommand does NOT support setBytes.
    // Only setKernelBuffer, setThreadgroupMemoryLength, and
    // setComputePipelineState are available. Kernels relying on
    // setBytes parameters should pre-allocate constant buffers.
    std::shared_ptr<CachedICB> build_icb(const std::vector<RecordedDispatch>& dispatches) {
        if (dispatches.empty()) return nullptr;

        @autoreleasepool {
            id<MTLDevice> device = MNDevice::instance().metal_device();

            // Create ICB descriptor
            MTLIndirectCommandBufferDescriptor* desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
            desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
            desc.inheritPipelineState = NO;
            desc.inheritBuffers = NO;
            desc.maxKernelBufferBindCount = 16;  // Support up to 16 buffer bindings

            uint32_t count = static_cast<uint32_t>(dispatches.size());
            if (count > ICBRecorder::MAX_COMMANDS) {
                count = ICBRecorder::MAX_COMMANDS;
            }

            id<MTLIndirectCommandBuffer> icb = [device newIndirectCommandBufferWithDescriptor:desc
                                                                              maxCommandCount:count
                                                                                      options:MTLResourceStorageModeShared];

            if (icb == nil) {
                NSLog(@"ICBRecorder: failed to create indirect command buffer");
                return nullptr;
            }

            auto cached = std::make_shared<CachedICB>();
            cached->icb = icb;
            cached->command_count = count;

            // Encode each dispatch into the ICB
            for (uint32_t i = 0; i < count; i++) {
                const auto& d = dispatches[i];

                id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:i];

                [cmd setComputePipelineState:d.pipeline];

                // Set buffers
                for (uint32_t j = 0; j < d.buffers.size(); j++) {
                    if (d.buffers[j] != nil) {
                        [cmd setKernelBuffer:d.buffers[j]
                                      offset:j < d.buffer_offsets.size() ? d.buffer_offsets[j] : 0
                                     atIndex:j];
                    }
                }

                // Set threadgroup memory
                for (uint32_t j = 0; j < d.threadgroup_memory_lengths.size(); j++) {
                    [cmd setThreadgroupMemoryLength:d.threadgroup_memory_lengths[j] atIndex:j];
                }

                // Set dispatch grid
                [cmd concurrentDispatchThreadgroups:d.grid_size
                             threadsPerThreadgroup:d.threadgroup_size];

                // Keep references alive
                cached->pipelines.push_back(d.pipeline);
                for (auto& buf : d.buffers) {
                    if (buf != nil) cached->buffers.push_back(buf);
                }
            }

            return cached;
        }
    }
};

// ---------------------------------------------------------------------------
// ICBRecorder public API
// ---------------------------------------------------------------------------

ICBRecorder& ICBRecorder::instance() {
    static ICBRecorder recorder;
    return recorder;
}

ICBRecorder::ICBRecorder() : impl_(std::make_unique<Impl>()) {}
ICBRecorder::~ICBRecorder() = default;

bool ICBRecorder::enabled() const noexcept {
    return impl_->enabled;
}

void ICBRecorder::set_enabled(bool enable) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->enabled = enable;
    if (!enable) {
        impl_->cache.clear();
        impl_->lru_order.clear();
        impl_->is_recording = false;
        impl_->recorded_dispatches.clear();
    }
}

bool ICBRecorder::has_cached(const ICBShapeKey& key) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->cache.find(key) != impl_->cache.end();
}

bool ICBRecorder::begin_recording(const ICBShapeKey& key) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    if (!impl_->enabled) return false;

    // Check cache first
    if (impl_->cache.find(key) != impl_->cache.end()) {
        return false;  // Already cached, use replay instead
    }

    impl_->is_recording = true;
    impl_->recording_key = key;
    impl_->recorded_dispatches.clear();
    return true;
}

void ICBRecorder::record_dispatch(id<MTLComputePipelineState> pipeline,
                                   NSArray<id<MTLBuffer>>* buffers,
                                   const uint32_t* buffer_offsets,
                                   uint32_t buffer_count,
                                   const void* bytes_data,
                                   uint32_t bytes_length,
                                   uint32_t bytes_index,
                                   MTLSize grid_size,
                                   MTLSize threadgroup_size,
                                   const uint32_t* threadgroup_memory_lengths,
                                   uint32_t threadgroup_memory_count) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    if (!impl_->is_recording) return;
    if (impl_->recorded_dispatches.size() >= MAX_COMMANDS) return;

    RecordedDispatch dispatch;
    dispatch.pipeline = pipeline;
    dispatch.grid_size = grid_size;
    dispatch.threadgroup_size = threadgroup_size;

    for (uint32_t i = 0; i < buffer_count && i < buffers.count; i++) {
        dispatch.buffers.push_back(buffers[i]);
        dispatch.buffer_offsets.push_back(buffer_offsets ? buffer_offsets[i] : 0);
    }

    if (bytes_data && bytes_length > 0) {
        dispatch.bytes_data.assign(static_cast<const uint8_t*>(bytes_data),
                                    static_cast<const uint8_t*>(bytes_data) + bytes_length);
        dispatch.bytes_index = bytes_index;
    }

    if (threadgroup_memory_lengths && threadgroup_memory_count > 0) {
        dispatch.threadgroup_memory_lengths.assign(threadgroup_memory_lengths,
                                                     threadgroup_memory_lengths + threadgroup_memory_count);
    }

    impl_->recorded_dispatches.push_back(std::move(dispatch));
}

void ICBRecorder::end_recording() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    if (!impl_->is_recording) return;
    impl_->is_recording = false;

    if (impl_->recorded_dispatches.empty()) return;

    auto cached = impl_->build_icb(impl_->recorded_dispatches);
    if (cached) {
        cached->key = impl_->recording_key;
        impl_->evict_if_needed();
        impl_->cache[impl_->recording_key] = cached;
        impl_->lru_order.push_front(impl_->recording_key);
    }

    impl_->recorded_dispatches.clear();
}

bool ICBRecorder::replay(const ICBShapeKey& key, id<MTLComputeCommandEncoder> encoder) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    if (!impl_->enabled) return false;

    auto it = impl_->cache.find(key);
    if (it == impl_->cache.end()) return false;

    auto& cached = it->second;
    impl_->touch(key);

    // Use resources from cached ICB
    for (auto& buf : cached->buffers) {
        [encoder useResource:buf usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    }

    // Execute the ICB
    [encoder executeCommandsInBuffer:cached->icb withRange:NSMakeRange(0, cached->command_count)];

    return true;
}

void ICBRecorder::invalidate_all() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->cache.clear();
    impl_->lru_order.clear();
}

void ICBRecorder::invalidate(const ICBShapeKey& key) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->cache.erase(key);
    impl_->lru_order.remove(key);
}

size_t ICBRecorder::cache_size() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->cache.size();
}

} // namespace metal_native
