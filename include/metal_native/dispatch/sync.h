#pragma once

/// @file sync.h
/// @brief GPU synchronisation primitives using MTLSharedEvent.
///
/// EventManager provides a monotonically increasing counter backed by an
/// MTLSharedEvent.  GPU command buffers can encode signal/wait operations
/// on the event, and the CPU can block until the GPU reaches a given value.
///
/// This is the primary mechanism for CPU-GPU synchronisation in
/// metal_native (e.g. waiting for a tensor to be ready before reading it
/// back on the CPU).
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

class EventManager {
public:
    /// Construct an EventManager backed by the given device.
    ///
    /// Creates an MTLSharedEvent and a listener dispatch queue.
    ///
    /// @param device  The MNDevice that owns the GPU.
    explicit EventManager(MNDevice& device);
    ~EventManager();

    // Non-copyable, non-movable.
    EventManager(const EventManager&) = delete;
    EventManager& operator=(const EventManager&) = delete;
    EventManager(EventManager&&) = delete;
    EventManager& operator=(EventManager&&) = delete;

    // -- Counter -------------------------------------------------------------

    /// Return the next monotonically increasing event value.
    ///
    /// Each call increments the internal counter and returns the new value.
    uint64_t next_value();

    // -- GPU-side operations -------------------------------------------------

#ifdef __OBJC__
    /// Encode a signal operation on a command buffer.
    ///
    /// When the GPU reaches this point in the command buffer, it will set
    /// the event to @p value.
    ///
    /// @param cb     The command buffer to encode the signal on.
    /// @param value  The event value to signal.
    void encode_signal(id<MTLCommandBuffer> cb, uint64_t value);

    /// Encode a wait operation on a command buffer.
    ///
    /// The GPU will stall at this point until the event reaches @p value.
    ///
    /// @param cb     The command buffer to encode the wait on.
    /// @param value  The event value to wait for.
    void encode_wait(id<MTLCommandBuffer> cb, uint64_t value);
#else
    /// Encode a signal operation (opaque command buffer handle for C++).
    void encode_signal(void* cb, uint64_t value);

    /// Encode a wait operation (opaque command buffer handle for C++).
    void encode_wait(void* cb, uint64_t value);
#endif

    // -- CPU-GPU synchronisation ---------------------------------------------

    /// Block the calling thread until the GPU signals the given value.
    ///
    /// Uses MTLSharedEvent's notification listener to avoid busy-waiting.
    ///
    /// @param value       The event value to wait for.
    /// @param timeout_ms  Maximum time to wait in milliseconds (default: 10 s).
    /// @throws MNException with TimeoutError if the wait exceeds timeout_ms.
    void cpu_wait(uint64_t value, uint64_t timeout_ms = 10000);

    // -- Queries -------------------------------------------------------------

    /// Return the most recent value that the GPU has signalled.
    uint64_t completed_value() const;

    /// Return the most recently issued counter value (from next_value()).
    uint64_t current_value() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
