#pragma once

/// @file trace.h
/// @brief Metal System Trace integration for GPU profiling.
///
/// TraceManager provides programmatic GPU capture using MTLCaptureManager
/// and signpost markers for operation dispatch tracking.  Captured traces
/// can be viewed in Xcode's Instruments or Metal Debugger.
///
/// Thread-safe: all public methods are guarded by a mutex.

#include <cstdint>
#include <memory>
#include <string>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace metal_native {

class MNDevice;

/// RAII helper for automatic signpost scope management.
///
/// Usage:
/// @code
///   {
///       TraceScope scope("matmul_forward");
///       // ... GPU operations ...
///   } // signpost automatically ended
/// @endcode
class TraceScope {
public:
    /// Begin a signpost with the given name.
    ///
    /// @param name  Signpost label (will appear in Instruments timeline).
    explicit TraceScope(const std::string& name);

    /// End the signpost automatically.
    ~TraceScope();

    // Non-copyable, non-movable.
    TraceScope(const TraceScope&) = delete;
    TraceScope& operator=(const TraceScope&) = delete;
    TraceScope(TraceScope&&) = delete;
    TraceScope& operator=(TraceScope&&) = delete;

private:
    std::string name_;
};

/// Singleton manager for Metal System Trace capture and signpost markers.
///
/// Provides programmatic control over GPU frame capture (for viewing in
/// Xcode Instruments) and os_signpost integration for fine-grained
/// operation tracking.
class TraceManager {
public:
    /// Return the process-wide TraceManager instance.
    static TraceManager& instance();

    // -- GPU Capture ---------------------------------------------------------

    /// Begin programmatic GPU capture.
    ///
    /// Starts recording all GPU commands issued after this call.  The capture
    /// continues until end_capture() is called or the destination file limit
    /// is reached.
    ///
    /// @throws MNException with InternalError if capture fails to start.
    void begin_capture();

    /// End GPU capture and write the .gputrace file.
    ///
    /// If set_capture_destination() was called, the trace is saved to that
    /// path; otherwise it goes to a default location in /tmp.
    ///
    /// @throws MNException with InternalError if capture fails to stop.
    void end_capture();

    /// Check if GPU capture is currently active.
    ///
    /// @return true if a capture session is in progress.
    bool is_capturing() const;

    /// Set the destination path for GPU trace files.
    ///
    /// Must be called before begin_capture() to take effect.
    /// The file extension should be .gputrace.
    ///
    /// @param path  Absolute path where the .gputrace file will be written.
    void set_capture_destination(const std::string& path);

    // -- Signpost Markers ----------------------------------------------------

    /// Begin a named signpost interval.
    ///
    /// Emits an os_signpost_interval_begin event that will appear in
    /// Instruments timelines.  Must be paired with end_signpost().
    ///
    /// @param name  Signpost label (e.g., "conv2d_forward").
    void begin_signpost(const std::string& name);

    /// End a named signpost interval.
    ///
    /// Emits an os_signpost_interval_end event matching the most recent
    /// begin_signpost() call with the same name.
    ///
    /// @param name  Signpost label (must match a previous begin_signpost).
    void end_signpost(const std::string& name);

    // -- Non-copyable / non-movable ------------------------------------------
    TraceManager(const TraceManager&) = delete;
    TraceManager& operator=(const TraceManager&) = delete;
    TraceManager(TraceManager&&) = delete;
    TraceManager& operator=(TraceManager&&) = delete;

private:
    TraceManager();
    ~TraceManager();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
