#pragma once

/// @file softmax.h
/// @brief Numerically stable softmax and log-softmax operations.
///
/// Implements softmax using a three-pass algorithm with SIMD-group reductions:
///   1. Find maximum value along the dimension
///   2. Compute exp(x - max) and sum
///   3. Normalize by dividing by the sum
///
/// Uses FP32 accumulation for FP16 inputs to maintain numerical precision.

#include <cstdint>

namespace metal_native {

class MNTensor;

/// Numerically stable softmax along a specified dimension.
///
/// Computes: softmax(x)[i] = exp(x[i]) / sum_j(exp(x[j]))
///
/// Uses the numerically stable formulation:
///   softmax(x)[i] = exp(x[i] - max(x)) / sum_j(exp(x[j] - max(x)))
///
/// @param input   Input tensor of any shape.
/// @param dim     Dimension along which to apply softmax (negative indexing
///                supported, e.g., -1 for last dimension).
/// @return        Output tensor with the same shape as input.
///
/// @throws MNException(InvalidArgument) if dim is out of range.
MNTensor softmax(const MNTensor& input, int64_t dim);

/// Numerically stable log-softmax along a specified dimension.
///
/// Computes: log_softmax(x)[i] = log(exp(x[i]) / sum_j(exp(x[j])))
///                              = x[i] - log(sum_j(exp(x[j])))
///
/// More numerically stable than log(softmax(x)).
///
/// @param input   Input tensor of any shape.
/// @param dim     Dimension along which to apply log-softmax.
/// @return        Output tensor with the same shape as input.
///
/// @throws MNException(InvalidArgument) if dim is out of range.
MNTensor log_softmax(const MNTensor& input, int64_t dim);

} // namespace metal_native
