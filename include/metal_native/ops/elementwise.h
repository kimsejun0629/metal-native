#pragma once

/// @file elementwise.h
/// @brief Element-wise arithmetic and math operations.
///
/// Provides element-wise binary operations (add, sub, mul, div), unary math
/// functions (exp, log, sqrt, abs, neg), and conditional operations (clamp, where).
/// All operations support broadcasting following NumPy semantics and dispatch
/// to optimized Metal kernels.

#include <cstddef>

#include "metal_native/core/dtype.h"
#include "metal_native/core/shape.h"

namespace metal_native {

class MNTensor;
class MNDevice;

// ---------------------------------------------------------------------------
// Binary arithmetic operations
// ---------------------------------------------------------------------------

/// Element-wise addition with broadcasting: output = a + b.
///
/// @param a       First input tensor.
/// @param b       Second input tensor.
/// @param device  Device to perform the operation on.
/// @return        Result tensor with broadcast shape.
/// @throws        MNException if shapes are not broadcastable.
MNTensor add(const MNTensor& a, const MNTensor& b, MNDevice& device);

/// Element-wise subtraction with broadcasting: output = a - b.
MNTensor sub(const MNTensor& a, const MNTensor& b, MNDevice& device);

/// Element-wise multiplication with broadcasting: output = a * b.
MNTensor mul(const MNTensor& a, const MNTensor& b, MNDevice& device);

/// Element-wise division with broadcasting: output = a / b.
MNTensor div(const MNTensor& a, const MNTensor& b, MNDevice& device);

// ---------------------------------------------------------------------------
// Unary math operations
// ---------------------------------------------------------------------------

/// Element-wise exponential: output = exp(x).
///
/// @param x       Input tensor.
/// @param device  Device to perform the operation on.
/// @return        Result tensor with same shape as input.
MNTensor exp(const MNTensor& x, MNDevice& device);

/// Element-wise natural logarithm: output = log(x).
MNTensor log(const MNTensor& x, MNDevice& device);

/// Element-wise square root: output = sqrt(x).
///
/// @param x       Input tensor.
/// @param device  Device to perform the operation on.
/// @return        Result tensor with same shape as input.
MNTensor sqrt(const MNTensor& x, MNDevice& device);

/// Element-wise absolute value: output = |x|.
MNTensor abs(const MNTensor& x, MNDevice& device);

/// Element-wise negation: output = -x.
MNTensor neg(const MNTensor& x, MNDevice& device);

// ---------------------------------------------------------------------------
// Conditional operations
// ---------------------------------------------------------------------------

/// Clamp values to a specified range: output = min(max(x, min_val), max_val).
///
/// @param x        Input tensor.
/// @param min_val  Minimum value.
/// @param max_val  Maximum value.
/// @param device   Device to perform the operation on.
/// @return         Result tensor with same shape as input.
MNTensor clamp(const MNTensor& x, float min_val, float max_val, MNDevice& device);

/// Element-wise conditional selection with broadcasting.
///
/// Returns elements from x where condition is true, otherwise from y.
/// All three tensors must be broadcastable to a common shape.
///
/// @param condition  Boolean tensor (dtype=Bool).
/// @param x          Tensor with values to select when condition is true.
/// @param y          Tensor with values to select when condition is false.
/// @param device     Device to perform the operation on.
/// @return           Result tensor with broadcast shape.
/// @throws           MNException if condition is not Bool dtype or shapes incompatible.
MNTensor where(const MNTensor& condition,
               const MNTensor& x,
               const MNTensor& y,
               MNDevice& device);

// ---------------------------------------------------------------------------
// Dtype conversion operations
// ---------------------------------------------------------------------------

/// Cast tensor to a different dtype.
///
/// Converts tensor elements to the target dtype using GPU kernels.
/// Currently supports FP32 ↔ FP16 conversions. If input already has
/// target dtype, returns the input unchanged.
///
/// @param input         Input tensor.
/// @param target_dtype  Desired output dtype.
/// @param device        Device to perform the operation on.
/// @return              Tensor with target dtype.
/// @throws              MNException if conversion is not supported.
MNTensor cast_dtype(const MNTensor& input, MNDType target_dtype, MNDevice& device);

} // namespace metal_native
