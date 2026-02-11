#pragma once

/// @file device.h
/// @brief Singleton wrapper around an Apple GPU (MTLDevice).
///
/// MNDevice owns the default Metal device and its command queue.  All other
/// subsystems obtain GPU access through this class.  The singleton is
/// initialised lazily and is thread-safe (std::call_once).
///
/// Because this header is included from both .cpp and .mm translation units,
/// Objective-C types are hidden behind `#ifdef __OBJC__` guards; the pure-C++
/// surface exposes only opaque `void*` accessors that .mm callers can cast.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace metal_native {

class CommandPipeline;
class GraphCache;

class MNDevice {
public:
    // -- Singleton access ----------------------------------------------------

    /// Return the process-wide MNDevice instance. The underlying MTLDevice is
    /// created on first call via MTLCreateSystemDefaultDevice().
    static MNDevice& instance();

    // -- Device properties ---------------------------------------------------

    /// Human-readable GPU name (e.g. "Apple M2 Max").
    const std::string& name() const noexcept;

    /// True when the device has unified memory (always true on Apple Silicon).
    bool has_unified_memory() const noexcept;

    /// Maximum size in bytes for a single MTLBuffer allocation.
    size_t max_buffer_length() const noexcept;

    /// Recommended upper bound of the working set (VRAM-equivalent budget).
    size_t recommended_max_working_set_size() const noexcept;

    /// Check whether the GPU supports a given MTLGPUFamily.
    /// The @p family value is cast to the Objective-C enum internally.
    bool supports_family(int family) const noexcept;

    /// True if the GPU supports BFloat16 (Apple GPU family 9 -- M3 and later).
    bool supports_bfloat16() const noexcept;

    // -- Raw Metal accessors -------------------------------------------------
    // Objective-C callers get typed id<>; C++ callers get void*.

#ifdef __OBJC__
    /// The underlying MTLDevice.
    id<MTLDevice> metal_device() const noexcept;

    /// The default command queue created at init time.
    id<MTLCommandQueue> command_queue() const noexcept;
#else
    /// Opaque handle to the underlying MTLDevice (cast to id<MTLDevice> in .mm).
    void* metal_device() const noexcept;

    /// Opaque handle to the default MTLCommandQueue.
    void* command_queue() const noexcept;
#endif

    // -- Synchronization -----------------------------------------------------

    /// Block until all pending GPU work on the command queue completes.
    void synchronize();

    /// Shared CommandPipeline for triple-buffered GPU submission.
    CommandPipeline& command_pipeline();

    /// Shared GraphCache for compiled MPSGraphExecutable objects.
    GraphCache& graph_cache();

    // -- Non-copyable / non-movable ------------------------------------------
    MNDevice(const MNDevice&) = delete;
    MNDevice& operator=(const MNDevice&) = delete;
    MNDevice(MNDevice&&) = delete;
    MNDevice& operator=(MNDevice&&) = delete;

private:
    MNDevice();
    ~MNDevice();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
