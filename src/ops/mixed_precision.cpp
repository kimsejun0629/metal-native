/// @file mixed_precision.cpp
/// @brief Pure C++ implementation of AMPController.

#include "metal_native/ops/mixed_precision.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/error.h"

#include <mutex>
#include <unordered_set>

namespace metal_native {

// ---------------------------------------------------------------------------
// AMPController::Impl
// ---------------------------------------------------------------------------

struct AMPController::Impl {
    mutable std::mutex mu;
    bool enabled = false;
    float scale = 65536.0f;  // Default gradient scale for FP16

    // Operations that benefit from FP16 (compute-bound)
    std::unordered_set<std::string> fp16_ops = {
        "matmul",
        "conv2d",
        "conv1d",
        "conv3d",
        "linear",
        "bmm",  // batch matrix multiply
    };

    // Operations that require FP32 (precision-sensitive)
    std::unordered_set<std::string> fp32_ops = {
        "layer_norm",
        "batch_norm",
        "instance_norm",
        "group_norm",
        "softmax",
        "log_softmax",
        "cross_entropy",
        "nll_loss",
        "mse_loss",
        "l1_loss",
        "embedding",  // lookup ops should preserve precision
    };

    MNDType get_compute_dtype_impl(const std::string& op_name) const {
        if (!enabled) {
            return MNDType::Float32;  // Default to FP32 when AMP disabled
        }

        // Check if explicitly marked as FP16-friendly
        if (fp16_ops.find(op_name) != fp16_ops.end()) {
            return MNDType::Float16;
        }

        // Check if explicitly marked as requiring FP32
        if (fp32_ops.find(op_name) != fp32_ops.end()) {
            return MNDType::Float32;
        }

        // Default conservative choice: FP32
        return MNDType::Float32;
    }
};

// ---------------------------------------------------------------------------
// AMPController public API
// ---------------------------------------------------------------------------

AMPController::AMPController() : impl_(std::make_unique<Impl>()) {}

AMPController::~AMPController() = default;

AMPController& AMPController::instance() {
    static AMPController instance;
    return instance;
}

bool AMPController::enabled() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->enabled;
}

void AMPController::set_enabled(bool enable) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->enabled = enable;
}

MNDType AMPController::get_compute_dtype(const std::string& op_name) const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->get_compute_dtype_impl(op_name);
}

MNTensor AMPController::cast_if_needed(const MNTensor& tensor,
                                        MNDType target_dtype) const {
    std::lock_guard<std::mutex> lock(impl_->mu);

    // If AMP is disabled, return original tensor
    if (!impl_->enabled) {
        return tensor;
    }

    // If tensor already has target dtype, no cast needed
    if (tensor.dtype() == target_dtype) {
        return tensor;
    }

    // Cast operation not implemented in v0.1.0
    MN_THROW(MetalNativeError::NotImplemented,
             "cast_if_needed: dtype casting not implemented in v0.1.0");
}

MNTensor AMPController::scale(const MNTensor& loss) const {
    // Stub for gradient scaling
    // In a full training framework, this would multiply the loss by the scale factor
    // For now, return the loss unchanged (forward-only inference)
    return loss;
}

void AMPController::unscale_gradients() const {
    // Stub for gradient unscaling
    // In a full training framework, this would divide accumulated gradients by scale
    // No-op for forward-only inference
}

void AMPController::update_scale() const {
    // Stub for dynamic scale adjustment
    // In a full training framework, this would:
    // - Check for gradient overflow/NaN
    // - Increase scale if gradients are healthy (growth_factor=2.0)
    // - Decrease scale on overflow (backoff_factor=0.5)
    // No-op for forward-only inference
}

float AMPController::get_scale() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->scale;
}

void AMPController::set_scale(float scale) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    MN_CHECK(scale > 0.0f,
             MetalNativeError::InvalidArgument,
             "AMPController: scale must be positive");
    impl_->scale = scale;
}

} // namespace metal_native
