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
#include <mach/mach_time.h>
#include <algorithm>

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

    // Encoder scope support
    id<MTLComputeCommandEncoder> active_encoder = nil;
    bool                    encoder_scope_active = false;
    size_t                  encoder_op_count = 0;
    size_t                  encoder_max_ops = 32;

    // Adaptive encoder timing
    uint64_t                encoder_start_time = 0;
    double                  encoder_duration_ema = 1.0;  // ms
    size_t                  adaptive_encoder_max_ops = 32;
    static constexpr double EMA_ALPHA = 0.3;

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

    /// Calculate encoder duration and update adaptive tuning.
    void update_encoder_timing() {
        if (encoder_start_time > 0) {
            uint64_t end_time = mach_absolute_time();
            mach_timebase_info_data_t info;
            mach_timebase_info(&info);
            double duration_ms = (double)(end_time - encoder_start_time) * info.numer / info.denom / 1e6;

            // Update EMA
            encoder_duration_ema = EMA_ALPHA * duration_ms + (1.0 - EMA_ALPHA) * encoder_duration_ema;

            // Adaptive tuning
            if (encoder_duration_ema > 2.0) {
                // Encoders taking too long, reduce max_ops
                adaptive_encoder_max_ops = std::max(size_t(8),
                    adaptive_encoder_max_ops * 3 / 4);  // 25% reduction
            } else if (encoder_duration_ema < 0.5) {
                // Encoders finishing quickly, increase max_ops
                adaptive_encoder_max_ops = std::min(size_t(128),
                    adaptive_encoder_max_ops * 5 / 4);  // 25% increase
            }

            encoder_start_time = 0;
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
        // Skip auto-flush if encoder scope is active to avoid invalidating the active encoder
        if (impl_->encoder_scope_active) return;
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
    // Skip auto-flush if encoder scope is active to avoid invalidating the active encoder
    if (impl_->lazy_commit && !impl_->encoder_scope_active
        && count >= impl_->auto_flush_threshold) {
        // Lock-free: only the thread that successfully resets the counter calls flush
        size_t expected = count;
        if (impl_->op_count.compare_exchange_strong(expected, 0)) {
            flush();
        }
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

// ---------------------------------------------------------------------------
// Encoder scope
// ---------------------------------------------------------------------------

id<MTLComputeCommandEncoder> CommandPipeline::current_encoder() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    // If encoder scope is active and we have an encoder, return it
    if (impl_->encoder_scope_active && impl_->active_encoder != nil) {
        // Check if we've encoded too many operations in this encoder (use adaptive value)
        if (impl_->encoder_op_count >= impl_->adaptive_encoder_max_ops) {
            // End current encoder and measure timing
            [impl_->active_encoder endEncoding];
            impl_->update_encoder_timing();

            // Commit the current buffer and get a fresh one
            impl_->backpressure.acquire();
            id<MTLCommandBuffer> buf = impl_->active_buffer;
            BackpressureController* bp = &impl_->backpressure;
            [buf addCompletedHandler:^(id<MTLCommandBuffer> /*cb*/) {
                bp->release();
            }];
            [buf commit];

            impl_->active_buffer = impl_->create_buffer();
            impl_->has_active = true;

            // Create new encoder and start timing
            impl_->active_encoder = [impl_->active_buffer computeCommandEncoder];
            impl_->encoder_op_count = 0;
            impl_->encoder_start_time = mach_absolute_time();
        }

        impl_->encoder_op_count++;
        return impl_->active_encoder;
    }

    // No encoder scope active, return nil
    return nil;
}

bool CommandPipeline::has_active_encoder() const noexcept {
    return impl_->encoder_scope_active && impl_->active_encoder != nil;
}

void CommandPipeline::set_encoder_max_ops(size_t max_ops) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->encoder_max_ops = std::clamp(max_ops, size_t(1), size_t(512));
    impl_->adaptive_encoder_max_ops = impl_->encoder_max_ops;  // Reset adaptive to manual value
}

size_t CommandPipeline::encoder_max_ops() const noexcept {
    return impl_->encoder_max_ops;
}

size_t CommandPipeline::adaptive_encoder_max_ops() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->adaptive_encoder_max_ops;
}

double CommandPipeline::encoder_duration_ema() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->encoder_duration_ema;
}

// ---------------------------------------------------------------------------
// EncoderScope RAII guard
// ---------------------------------------------------------------------------

CommandPipeline::EncoderScope::EncoderScope(CommandPipeline& pipeline)
    : pipeline_(pipeline)
    , previous_encoder_state_(pipeline.impl_->encoder_scope_active)
    , previous_lazy_state_(pipeline.lazy_commit())
{
    std::lock_guard<std::mutex> lock(pipeline.impl_->mu);

    // Enable lazy commit for this scope (batching)
    pipeline.set_lazy_commit(true);

    // Mark encoder scope as active
    pipeline.impl_->encoder_scope_active = true;

    // Create the encoder if we don't have one yet
    if (pipeline.impl_->active_encoder == nil) {
        // Ensure we have an active buffer
        if (!pipeline.impl_->has_active) {
            pipeline.impl_->active_buffer = pipeline.impl_->create_buffer();
            pipeline.impl_->has_active = true;
        }

        pipeline.impl_->active_encoder = [pipeline.impl_->active_buffer computeCommandEncoder];
        pipeline.impl_->encoder_op_count = 0;
        pipeline.impl_->encoder_start_time = mach_absolute_time();
    }
}

CommandPipeline::EncoderScope::~EncoderScope() {
    // End the active encoder and restore state
    {
        std::lock_guard<std::mutex> lock(pipeline_.impl_->mu);

        // End the active encoder if we have one
        if (pipeline_.impl_->active_encoder != nil) {
            [pipeline_.impl_->active_encoder endEncoding];
            pipeline_.impl_->update_encoder_timing();
            pipeline_.impl_->active_encoder = nil;
            pipeline_.impl_->encoder_op_count = 0;
        }

        // Restore encoder scope state
        pipeline_.impl_->encoder_scope_active = previous_encoder_state_;

        // If restoring to an active outer scope, recreate the encoder
        // so the outer scope can continue encoding operations.
        if (previous_encoder_state_ && pipeline_.impl_->has_active) {
            pipeline_.impl_->active_encoder =
                [pipeline_.impl_->active_buffer computeCommandEncoder];
            pipeline_.impl_->encoder_op_count = 0;
            pipeline_.impl_->encoder_start_time = mach_absolute_time();
        }
    }
    // Lock released here

    // Only flush if we're the outermost scope (no outer encoder scope active)
    if (!previous_encoder_state_) {
        pipeline_.flush();
    }

    // Restore lazy commit state
    pipeline_.set_lazy_commit(previous_lazy_state_);
}

CommandPipeline::EncoderScope CommandPipeline::encoder_scope() {
    return EncoderScope(*this);
}

CommandPipeline::EncoderScope CommandPipeline::transformer_layer_scope() {
    // This combines batch_scope + encoder_scope
    // Since EncoderScope already enables lazy_commit internally,
    // we just return an EncoderScope
    return EncoderScope(*this);
}

} // namespace metal_native
