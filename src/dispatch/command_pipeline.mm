/// @file command_pipeline.mm
/// @brief Objective-C++ implementation of CommandPipeline.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/dispatch/command_pipeline.h"
#include "metal_native/dispatch/backpressure.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

#include <atomic>
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
    std::atomic<bool>       lazy_commit{false};

    // Operation counting for auto-flush
    std::atomic<size_t>     op_count{0};
    std::atomic<size_t>     auto_flush_threshold{64};

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
    // Count this operation
    size_t count = ++impl_->op_count;

    // In lazy mode, check auto-flush threshold
    if (impl_->lazy_commit) {
        if (count >= impl_->auto_flush_threshold) {
            flush();
            impl_->op_count = 0;
        }
        return;
    }

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

    // Reset operation counter after commit
    impl_->op_count = 0;
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

    // Reset operation counter after commit
    impl_->op_count = 0;
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

    // Reset operation counter after flush
    impl_->op_count = 0;
}

// ---------------------------------------------------------------------------
// Operation counting
// ---------------------------------------------------------------------------

void CommandPipeline::record_op() {
    size_t count = ++impl_->op_count;
    // Auto-flush when threshold reached (only in lazy mode)
    if (impl_->lazy_commit && count >= impl_->auto_flush_threshold) {
        flush();
        impl_->op_count = 0;
    }
}

void CommandPipeline::set_auto_flush_threshold(size_t threshold) {
    impl_->auto_flush_threshold = threshold;
}

size_t CommandPipeline::auto_flush_threshold() const noexcept {
    return impl_->auto_flush_threshold;
}

// ---------------------------------------------------------------------------
// BatchScope RAII guard
// ---------------------------------------------------------------------------

CommandPipeline::BatchScope::BatchScope(CommandPipeline& pipeline)
    : pipeline_(pipeline)
    , previous_lazy_state_(pipeline.lazy_commit())
{
    pipeline_.set_lazy_commit(true);
}

CommandPipeline::BatchScope::~BatchScope() {
    pipeline_.flush();
    pipeline_.set_lazy_commit(previous_lazy_state_);
}

CommandPipeline::BatchScope CommandPipeline::batch_scope() {
    return BatchScope(*this);
}

} // namespace metal_native
