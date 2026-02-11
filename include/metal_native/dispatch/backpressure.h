#pragma once

/// @file backpressure.h
/// @brief Throttle GPU command submission to prevent out-of-memory conditions.
///
/// BackpressureController limits the number of in-flight (committed but not
/// yet completed) command buffers.  When the limit is reached, acquire()
/// blocks the calling thread until a completion handler calls release().
///
/// Uses std::condition_variable for C++17 compatibility (no std::counting_semaphore).
///
/// Thread-safe: all public methods are guarded by a mutex.

#include <cstddef>
#include <condition_variable>
#include <mutex>

namespace metal_native {

class BackpressureController {
public:
    /// Construct a controller with a maximum number of in-flight buffers.
    ///
    /// @param max_in_flight  Maximum concurrent in-flight command buffers
    ///                       (default: 3, matching triple-buffering).
    explicit BackpressureController(size_t max_in_flight = 3);

    ~BackpressureController() = default;

    // Non-copyable, non-movable.
    BackpressureController(const BackpressureController&) = delete;
    BackpressureController& operator=(const BackpressureController&) = delete;
    BackpressureController(BackpressureController&&) = delete;
    BackpressureController& operator=(BackpressureController&&) = delete;

    // -- Slot management -----------------------------------------------------

    /// Block the calling thread until a submission slot is available.
    ///
    /// If the number of in-flight buffers is already at max_in_flight,
    /// this will wait on a condition variable until release() is called.
    void acquire();

    /// Signal that an in-flight buffer has completed.
    ///
    /// Typically called from a command-buffer completion handler.
    /// Wakes one thread blocked in acquire().
    void release();

    // -- Queries -------------------------------------------------------------

    /// Number of in-flight (acquired but not released) slots.
    size_t current_depth() const;

    /// Maximum allowed in-flight count.
    size_t max_depth() const noexcept;

private:
    const size_t max_in_flight_;
    size_t       in_flight_ = 0;

    mutable std::mutex      mu_;
    std::condition_variable  cv_;
};

} // namespace metal_native
