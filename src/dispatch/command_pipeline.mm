/// @file command_pipeline.mm
/// @brief Objective-C++ implementation of CommandPipeline.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/dispatch/command_pipeline.h"
#include "metal_native/dispatch/backpressure.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

#include <mutex>
#include <vector>

namespace metal_native {

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct CommandPipeline::Impl {
    MNDevice&               device;
    size_t                  buffer_count;
    BackpressureController  backpressure;

    mutable std::mutex      mu;
    id<MTLCommandBuffer>    active_buffer = nil;
    bool                    has_active    = false;
    bool                    lazy_commit   = false;

    explicit Impl(MNDevice& dev, size_t count)
        : device(dev)
        , buffer_count(count)
        , backpressure(count) {}

    /// Create a fresh command buffer from the device's command queue.
    id<MTLCommandBuffer> create_buffer() {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [device.command_queue() commandBuffer];
            MN_CHECK(cb != nil,
                     MetalNativeError::InternalError,
                     "CommandPipeline: failed to create MTLCommandBuffer");
            return cb;
        }
    }
};

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

CommandPipeline::CommandPipeline(MNDevice& device, size_t buffer_count)
    : impl_(std::make_unique<Impl>(device, buffer_count)) {
    MN_CHECK(buffer_count > 0,
             MetalNativeError::InvalidArgument,
             "CommandPipeline: buffer_count must be > 0");
}

CommandPipeline::~CommandPipeline() {
    // Ensure all in-flight work completes before destroying.
    synchronize();
}

// ---------------------------------------------------------------------------
// Buffer lifecycle
// ---------------------------------------------------------------------------

id<MTLCommandBuffer> CommandPipeline::current_buffer() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    if (!impl_->has_active) {
        impl_->active_buffer = impl_->create_buffer();
        impl_->has_active    = true;
    }

    return impl_->active_buffer;
}

void CommandPipeline::commit_and_continue() {
    // In lazy mode, skip the commit — operations accumulate in the
    // current buffer until synchronize() or flush() is called.
    // This matches PyTorch MPS behavior of batching many ops per submit.
    if (impl_->lazy_commit) return;

    // Acquire a backpressure slot BEFORE locking the mutex to prevent starvation.
    // If backpressure blocks (all slots full), we don't want to hold impl_->mu.
    impl_->backpressure.acquire();

    std::lock_guard<std::mutex> lock(impl_->mu);

    MN_CHECK(impl_->has_active,
             MetalNativeError::InternalError,
             "CommandPipeline::commit_and_continue: no active buffer to commit");

    id<MTLCommandBuffer> buf = impl_->active_buffer;
    BackpressureController* bp = &impl_->backpressure;

    [buf addCompletedHandler:^(id<MTLCommandBuffer> /*cb*/) {
        bp->release();
    }];

    [buf commit];

    // Create next buffer immediately for continued encoding.
    impl_->active_buffer = impl_->create_buffer();
}

void CommandPipeline::commit() {
    // Acquire a backpressure slot BEFORE locking the mutex to prevent starvation.
    // If backpressure blocks (all slots full), we don't want to hold impl_->mu.
    impl_->backpressure.acquire();

    std::lock_guard<std::mutex> lock(impl_->mu);

    MN_CHECK(impl_->has_active,
             MetalNativeError::InternalError,
             "CommandPipeline::commit: no active buffer to commit");

    id<MTLCommandBuffer> buf = impl_->active_buffer;
    BackpressureController* bp = &impl_->backpressure;

    [buf addCompletedHandler:^(id<MTLCommandBuffer> /*cb*/) {
        bp->release();
    }];

    [buf commit];

    impl_->active_buffer = nil;
    impl_->has_active    = false;
}

void CommandPipeline::synchronize() {
    // Commit any active buffer first.
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        if (impl_->has_active) {
            impl_->backpressure.acquire();

            id<MTLCommandBuffer> buf = impl_->active_buffer;
            BackpressureController* bp = &impl_->backpressure;

            [buf addCompletedHandler:^(id<MTLCommandBuffer> /*cb*/) {
                bp->release();
            }];

            [buf commit];
            impl_->active_buffer = nil;
            impl_->has_active    = false;
        }
    }

    // Wait until all backpressure slots are freed (all buffers completed).
    // We do this by acquiring all slots and then releasing them.
    for (size_t i = 0; i < impl_->buffer_count; ++i) {
        impl_->backpressure.acquire();
    }
    for (size_t i = 0; i < impl_->buffer_count; ++i) {
        impl_->backpressure.release();
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

size_t CommandPipeline::in_flight_count() const {
    return impl_->backpressure.current_depth();
}

size_t CommandPipeline::buffer_count() const noexcept {
    return impl_->buffer_count;
}

// ---------------------------------------------------------------------------
// Lazy commit mode
// ---------------------------------------------------------------------------

void CommandPipeline::set_lazy_commit(bool enable) {
    impl_->lazy_commit = enable;
}

bool CommandPipeline::lazy_commit() const noexcept {
    return impl_->lazy_commit;
}

void CommandPipeline::flush() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (!impl_->has_active) return;

    impl_->backpressure.acquire();

    id<MTLCommandBuffer> buf = impl_->active_buffer;
    BackpressureController* bp = &impl_->backpressure;

    [buf addCompletedHandler:^(id<MTLCommandBuffer> /*cb*/) {
        bp->release();
    }];

    [buf commit];

    // Create next buffer for continued encoding.
    impl_->active_buffer = impl_->create_buffer();
}

} // namespace metal_native
