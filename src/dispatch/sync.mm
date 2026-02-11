/// @file sync.mm
/// @brief Objective-C++ implementation of EventManager.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/dispatch/sync.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace metal_native {

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct EventManager::Impl {
    MNDevice&               device;
    id<MTLSharedEvent>      event          = nil;
    MTLSharedEventListener* listener       = nil;
    dispatch_queue_t        listener_queue = nil;

    std::atomic<uint64_t>   counter{0};
    mutable std::mutex      mu;

    explicit Impl(MNDevice& dev) : device(dev) {}
};

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

EventManager::EventManager(MNDevice& device)
    : impl_(std::make_unique<Impl>(device)) {
    @autoreleasepool {
        impl_->event = [device.metal_device() newSharedEvent];
        MN_CHECK(impl_->event != nil,
                 MetalNativeError::InternalError,
                 "EventManager: failed to create MTLSharedEvent");

        impl_->listener_queue = dispatch_queue_create(
            "com.metal_native.event_listener",
            DISPATCH_QUEUE_SERIAL);
        MN_CHECK(impl_->listener_queue != nil,
                 MetalNativeError::InternalError,
                 "EventManager: failed to create dispatch queue");

        impl_->listener = [[MTLSharedEventListener alloc]
            initWithDispatchQueue:impl_->listener_queue];
        MN_CHECK(impl_->listener != nil,
                 MetalNativeError::InternalError,
                 "EventManager: failed to create MTLSharedEventListener");
    }
}

EventManager::~EventManager() = default;

// ---------------------------------------------------------------------------
// Counter
// ---------------------------------------------------------------------------

uint64_t EventManager::next_value() {
    return impl_->counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

// ---------------------------------------------------------------------------
// GPU-side operations
// ---------------------------------------------------------------------------

void EventManager::encode_signal(id<MTLCommandBuffer> cb, uint64_t value) {
    MN_CHECK(cb != nil,
             MetalNativeError::InvalidArgument,
             "EventManager::encode_signal: command buffer must not be nil");

    [cb encodeSignalEvent:impl_->event value:value];
}

void EventManager::encode_wait(id<MTLCommandBuffer> cb, uint64_t value) {
    MN_CHECK(cb != nil,
             MetalNativeError::InvalidArgument,
             "EventManager::encode_wait: command buffer must not be nil");

    [cb encodeWaitForEvent:impl_->event value:value];
}

// ---------------------------------------------------------------------------
// CPU-GPU synchronisation
// ---------------------------------------------------------------------------

void EventManager::cpu_wait(uint64_t value, uint64_t timeout_ms) {
    // Fast path: check if the GPU has already completed past this value.
    if (impl_->event.signaledValue >= value) {
        return;
    }

    // Use MTLSharedEvent's listener-based notification to avoid busy-waiting.
    // Heap-allocate the sync state so it can be shared between the ObjC block
    // (which captures by copy) and the waiting thread.
    struct WaitState {
        std::mutex mu;
        std::condition_variable cv;
        bool notified = false;
    };
    auto state = std::make_shared<WaitState>();

    [impl_->event notifyListener:impl_->listener
                         atValue:value
                           block:^(id<MTLSharedEvent> /*event*/, uint64_t /*val*/) {
        std::lock_guard<std::mutex> lock(state->mu);
        state->notified = true;
        state->cv.notify_one();
    }];

    std::unique_lock<std::mutex> lock(state->mu);
    bool ok = state->cv.wait_for(
        lock,
        std::chrono::milliseconds(timeout_ms),
        [&state] { return state->notified; });

    MN_CHECK(ok,
             MetalNativeError::TimeoutError,
             "EventManager::cpu_wait: timed out waiting for GPU event "
             "(value=" + std::to_string(value) +
             ", timeout=" + std::to_string(timeout_ms) + "ms)");
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

uint64_t EventManager::completed_value() const {
    return impl_->event.signaledValue;
}

uint64_t EventManager::current_value() const {
    return impl_->counter.load(std::memory_order_relaxed);
}

} // namespace metal_native
