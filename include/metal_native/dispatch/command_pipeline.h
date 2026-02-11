#pragma once

/// @file command_pipeline.h
/// @brief Triple-buffered MTLCommandBuffer management for GPU command submission.
///
/// CommandPipeline maintains a ring buffer of MTLCommandBuffer instances,
/// allowing overlap between CPU encoding and GPU execution.  The default
/// configuration uses 3 buffers (triple-buffering), which is the sweet spot
/// for Apple Silicon GPUs.
///
/// The pipeline integrates with BackpressureController to throttle submission
/// when the GPU falls behind.
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

class CommandPipeline {
public:
    /// Construct a pipeline backed by the given device.
    ///
    /// @param device        The MNDevice that owns the command queue.
    /// @param buffer_count  Number of command buffer slots (default: 3).
    explicit CommandPipeline(MNDevice& device, size_t buffer_count = 3);
    ~CommandPipeline();

    // Non-copyable, non-movable.
    CommandPipeline(const CommandPipeline&) = delete;
    CommandPipeline& operator=(const CommandPipeline&) = delete;
    CommandPipeline(CommandPipeline&&) = delete;
    CommandPipeline& operator=(CommandPipeline&&) = delete;

    // -- Buffer lifecycle ----------------------------------------------------

#ifdef __OBJC__
    /// Return the current command buffer for encoding compute/blit commands.
    ///
    /// If no buffer exists yet, one is created from the device's command queue.
    id<MTLCommandBuffer> current_buffer();
#else
    /// Opaque handle to the current MTLCommandBuffer (cast in .mm files).
    void* current_buffer();
#endif

    /// Commit the current command buffer and create a new one for the
    /// next encoding pass.
    ///
    /// This adds a completion handler, commits the buffer, and advances
    /// the ring index.  Blocks if all buffer slots are in-flight
    /// (backpressure).
    void commit_and_continue();

    /// Commit the current command buffer (final submission, no rotation).
    void commit();

    /// Block until all in-flight command buffers have completed on the GPU.
    void synchronize();

    // -- Queries -------------------------------------------------------------

    /// Number of committed-but-not-completed buffers currently in flight.
    size_t in_flight_count() const;

    /// Total number of buffer slots in the ring.
    size_t buffer_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
