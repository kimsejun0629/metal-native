#pragma once

/// @file mixed_precision.h
/// @brief Automatic Mixed Precision (AMP) controller for training optimization.
///
/// Provides automatic dtype selection and gradient scaling for mixed-precision
/// training. Compute-bound operations (matmul, conv) use FP16 for speed, while
/// precision-sensitive operations (normalization, loss) use FP32 for stability.

#include <memory>
#include <string>

#include "metal_native/core/dtype.h"

namespace metal_native {

class MNTensor;

/// Automatic Mixed Precision controller (singleton).
///
/// Controls per-operation dtype policy and provides gradient scaling utilities
/// for stable mixed-precision training. The controller follows these policies:
///
/// - Compute-bound ops (matmul, conv2d): FP16 for performance
/// - Precision-sensitive ops (layer_norm, softmax, loss): FP32 for stability
/// - Element-wise ops: match input dtype
///
/// Example usage:
/// @code
///   AMPController& amp = AMPController::instance();
///   amp.set_enabled(true);
///
///   // Automatically selects FP16 for matmul
///   MNDType compute_dtype = amp.get_compute_dtype("matmul");
///
///   // Cast tensor if needed
///   MNTensor x_compute = amp.cast_if_needed(x, compute_dtype);
/// @endcode
class AMPController {
public:
    /// Get the singleton instance.
    static AMPController& instance();

    // Non-copyable, non-movable.
    AMPController(const AMPController&) = delete;
    AMPController& operator=(const AMPController&) = delete;
    AMPController(AMPController&&) = delete;
    AMPController& operator=(AMPController&&) = delete;

    // -- Enable/disable AMP --------------------------------------------------

    /// Check if AMP is enabled.
    bool enabled() const noexcept;

    /// Enable or disable AMP globally.
    ///
    /// When disabled, all operations use their native input dtypes.
    void set_enabled(bool enable) noexcept;

    // -- Dtype policy --------------------------------------------------------

    /// Get the recommended compute dtype for an operation.
    ///
    /// Returns the dtype to use for computation based on the operation type:
    /// - "matmul", "conv2d", "linear": FP16 (compute-bound)
    /// - "layer_norm", "softmax", "cross_entropy", "mse_loss": FP32 (precision-sensitive)
    /// - Others: FP32 (default conservative choice)
    ///
    /// @param op_name  Name of the operation.
    /// @return         Recommended dtype (FP16 or FP32).
    MNDType get_compute_dtype(const std::string& op_name) const noexcept;

    /// Cast a tensor to the target dtype if AMP is enabled and dtypes differ.
    ///
    /// If AMP is disabled or the tensor already has the target dtype, returns
    /// the original tensor. Otherwise, performs a dtype cast.
    ///
    /// @param tensor        Input tensor.
    /// @param target_dtype  Desired dtype.
    /// @return              Original or cast tensor.
    MNTensor cast_if_needed(const MNTensor& tensor, MNDType target_dtype) const;

    // -- Gradient scaling (stubs for future training support) ---------------

    /// Scale loss for backward pass to prevent FP16 underflow.
    ///
    /// In mixed-precision training, gradients can underflow in FP16.
    /// This scales the loss by a large factor before backward() to keep
    /// gradients in FP16 range.
    ///
    /// @param loss  Loss tensor.
    /// @return      Scaled loss tensor.
    /// @note        Currently a no-op stub; scaling is applied in future backward pass.
    MNTensor scale(const MNTensor& loss) const;

    /// Unscale gradients before optimizer step.
    ///
    /// After backward(), gradients are scaled. This method would unscale them
    /// before the optimizer updates weights.
    ///
    /// @note  Currently a no-op stub for forward-only inference framework.
    void unscale_gradients() const;

    /// Update the gradient scaler based on observed gradient overflow.
    ///
    /// Dynamically adjusts the scaling factor: increases it when gradients
    /// are healthy, decreases it on overflow/NaN detection.
    ///
    /// @note  Currently a no-op stub for forward-only inference framework.
    void update_scale() const;

    // -- Configuration -------------------------------------------------------

    /// Get the current gradient scaling factor.
    float get_scale() const noexcept;

    /// Set the gradient scaling factor.
    ///
    /// @param scale  Scaling factor (default: 65536.0f for FP16).
    void set_scale(float scale);

private:
    AMPController();
    ~AMPController();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
