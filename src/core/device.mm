/// @file device.mm
/// @brief Objective-C++ implementation of MNDevice.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/dispatch/command_pipeline.h"
#include "metal_native/graph/graph_cache.h"
#include "metal_native/memory/allocator.h"

#include <mutex>
#include <string>

namespace metal_native {

// ---------------------------------------------------------------------------
// Impl (pimpl -- hides Objective-C types from the header)
// ---------------------------------------------------------------------------

struct MNDevice::Impl {
    id<MTLDevice>       device       = nil;
    id<MTLCommandQueue> queue        = nil;
    id<MTLCommandQueue> blit_queue_  = nil;
    std::string         device_name;
    bool                unified      = false;
    size_t              max_buf_len  = 0;
    size_t              max_working  = 0;
    bool                bf16_support = false;
    std::unique_ptr<CommandPipeline> cmd_pipeline;
    std::unique_ptr<GraphCache>     graph_cache_inst;
    std::unique_ptr<MetalSmartAllocator> allocator_inst;
};

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

MNDevice& MNDevice::instance() {
    static std::once_flag flag;
    static MNDevice* singleton = nullptr;

    std::call_once(flag, [] {
        singleton = new MNDevice();
    });

    return *singleton;
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

MNDevice::MNDevice() : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        MN_CHECK(impl_->device != nil,
                 MetalNativeError::DeviceNotFound,
                 "MTLCreateSystemDefaultDevice() returned nil -- "
                 "no Metal-capable GPU found");

        impl_->queue = [impl_->device newCommandQueue];
        MN_CHECK(impl_->queue != nil,
                 MetalNativeError::InternalError,
                 "failed to create default MTLCommandQueue");

        impl_->blit_queue_ = [impl_->device newCommandQueue];
        impl_->blit_queue_.label = @"metal_native.blit";

        impl_->device_name  = std::string([[impl_->device name] UTF8String]);
        impl_->unified      = [impl_->device hasUnifiedMemory];
        impl_->max_buf_len  = static_cast<size_t>([impl_->device maxBufferLength]);
        impl_->max_working  = static_cast<size_t>(
            [impl_->device recommendedMaxWorkingSetSize]);

        // BFloat16 requires Apple GPU family 9 (M3 and later).
        // MTLGPUFamilyApple9 == 1009 in the Metal enum.
        impl_->bf16_support = [impl_->device supportsFamily:MTLGPUFamilyApple9];

        // Shared subsystems (must be after device + queue init)
        impl_->cmd_pipeline = std::make_unique<CommandPipeline>(*this, 3);
        impl_->graph_cache_inst = std::make_unique<GraphCache>(128);
        impl_->graph_cache_inst->start_eviction_timer(30);
        impl_->allocator_inst = std::make_unique<MetalSmartAllocator>(*this);
    }
}

MNDevice::~MNDevice() = default;

// ---------------------------------------------------------------------------
// Property accessors
// ---------------------------------------------------------------------------

const std::string& MNDevice::name() const noexcept {
    return impl_->device_name;
}

bool MNDevice::has_unified_memory() const noexcept {
    return impl_->unified;
}

size_t MNDevice::max_buffer_length() const noexcept {
    return impl_->max_buf_len;
}

size_t MNDevice::recommended_max_working_set_size() const noexcept {
    return impl_->max_working;
}

bool MNDevice::supports_family(int family) const noexcept {
    return [impl_->device supportsFamily:static_cast<MTLGPUFamily>(family)];
}

bool MNDevice::supports_bfloat16() const noexcept {
    return impl_->bf16_support;
}

// ---------------------------------------------------------------------------
// Raw Metal accessors
// ---------------------------------------------------------------------------

id<MTLDevice> MNDevice::metal_device() const noexcept {
    return impl_->device;
}

id<MTLCommandQueue> MNDevice::command_queue() const noexcept {
    return impl_->queue;
}

id<MTLCommandQueue> MNDevice::blit_queue() const noexcept {
    return impl_->blit_queue_;
}

// ---------------------------------------------------------------------------
// Synchronization
// ---------------------------------------------------------------------------

void MNDevice::synchronize() {
    @autoreleasepool {
        id<MTLCommandBuffer> barrier = [impl_->queue commandBuffer];
        [barrier commit];
        [barrier waitUntilCompleted];
    }
}

// ---------------------------------------------------------------------------
// Shared subsystem accessors
// ---------------------------------------------------------------------------

CommandPipeline& MNDevice::command_pipeline() {
    return *impl_->cmd_pipeline;
}

GraphCache& MNDevice::graph_cache() {
    return *impl_->graph_cache_inst;
}

MetalSmartAllocator& MNDevice::allocator() {
    return *impl_->allocator_inst;
}

void MNDevice::set_prefer_private_storage(bool enable) {
    prefer_private_storage_.store(enable, std::memory_order_relaxed);
}

bool MNDevice::prefer_private_storage() const noexcept {
    return prefer_private_storage_.load(std::memory_order_relaxed);
}

} // namespace metal_native
