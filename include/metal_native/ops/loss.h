#pragma once

/// @file loss.h
/// @brief Loss function operators.
///
/// Provides common loss functions for training neural networks, including
/// cross-entropy loss, mean squared error, and L1 loss.

#include <cstddef>

namespace metal_native {

// Forward declaration
class MNTensor;

/// Reduction mode for loss functions.
enum class ReductionMode {
    None,   ///< No reduction, return per-element losses.
    Mean,   ///< Return the mean of all losses.
    Sum     ///< Return the sum of all losses.
};

/// Cross-entropy loss with integrated log-softmax.
///
/// Computes the cross-entropy loss between input logits and target labels.
/// This function fuses log_softmax and negative log-likelihood loss for
/// numerical stability and efficiency.
///
/// @param input      Input logits tensor [N, C] or [N, C, ...].
/// @param target     Target labels tensor [N] (integer class indices).
/// @param reduction  Reduction mode (default: Mean).
/// @return           Loss tensor (scalar if reduction != None, otherwise [N]).
/// @throws MNException(InvalidArgument) if shapes are incompatible.
MNTensor cross_entropy_loss(const MNTensor& input,
                            const MNTensor& target,
                            ReductionMode reduction = ReductionMode::Mean);

/// Mean squared error loss.
///
/// Computes the mean squared error between input and target:
/// loss = (input - target)^2
///
/// @param input      Input tensor.
/// @param target     Target tensor (must have same shape as input).
/// @param reduction  Reduction mode (default: Mean).
/// @return           Loss tensor (scalar if reduction != None, otherwise same shape as input).
/// @throws MNException(InvalidArgument) if shapes don't match.
MNTensor mse_loss(const MNTensor& input,
                  const MNTensor& target,
                  ReductionMode reduction = ReductionMode::Mean);

/// L1 loss (mean absolute error).
///
/// Computes the L1 loss between input and target:
/// loss = |input - target|
///
/// @param input      Input tensor.
/// @param target     Target tensor (must have same shape as input).
/// @param reduction  Reduction mode (default: Mean).
/// @return           Loss tensor (scalar if reduction != None, otherwise same shape as input).
/// @throws MNException(InvalidArgument) if shapes don't match.
MNTensor l1_loss(const MNTensor& input,
                 const MNTensor& target,
                 ReductionMode reduction = ReductionMode::Mean);

} // namespace metal_native
