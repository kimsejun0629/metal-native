/// @file backpressure.cpp
/// @brief Implementation of BackpressureController.

#include "metal_native/dispatch/backpressure.h"
#include "metal_native/core/error.h"

namespace metal_native {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

BackpressureController::BackpressureController(size_t max_in_flight)
    : max_in_flight_(max_in_flight) {
    MN_CHECK(max_in_flight_ > 0,
             MetalNativeError::InvalidArgument,
             "BackpressureController: max_in_flight must be > 0");
}

// ---------------------------------------------------------------------------
// Slot management
// ---------------------------------------------------------------------------

void BackpressureController::acquire() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this] { return in_flight_ < max_in_flight_; });
    ++in_flight_;
}

void BackpressureController::release() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        MN_CHECK(in_flight_ > 0,
                 MetalNativeError::InternalError,
                 "BackpressureController::release: no in-flight slots to release");
        --in_flight_;
    }
    cv_.notify_one();
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

size_t BackpressureController::current_depth() const {
    std::lock_guard<std::mutex> lock(mu_);
    return in_flight_;
}

size_t BackpressureController::max_depth() const noexcept {
    return max_in_flight_;
}

} // namespace metal_native
