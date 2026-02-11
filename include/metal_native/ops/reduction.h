#pragma once

/// @file reduction.h
/// @brief Reduction operations along tensor dimensions.
///
/// Provides sum, mean, max, min, argmax, and argmin reductions with support
/// for keepdim. All operations use multi-stage parallel reduction with
/// threadgroup shared memory and SIMD-group shuffles for optimal performance.

#include <cstddef>
#include <cstdint>

#include "metal_native/core/dtype.h"
#include "metal_native/core/shape.h"

namespace metal_native {

class MNTensor;
class MNDevice;

/// Reduce sum along a specified dimension.
///
/// @param input    Input tensor.
/// @param dim      Dimension to reduce over (negative indices count from end).
/// @param keepdim  If true, retain the reduced dimension with size 1.
/// @param device   Device to perform the operation on.
/// @return         Reduced tensor with FP32 accumulation for FP16 inputs.
/// @throws         MNException on invalid dimension.
MNTensor reduce_sum(const MNTensor& input,
                    int64_t dim,
                    bool keepdim,
                    MNDevice& device);

/// Reduce mean along a specified dimension.
///
/// @param input    Input tensor.
/// @param dim      Dimension to reduce over.
/// @param keepdim  If true, retain the reduced dimension with size 1.
/// @param device   Device to perform the operation on.
/// @return         Reduced tensor with mean values.
/// @throws         MNException on invalid dimension.
MNTensor reduce_mean(const MNTensor& input,
                     int64_t dim,
                     bool keepdim,
                     MNDevice& device);

/// Reduce max along a specified dimension.
///
/// @param input    Input tensor.
/// @param dim      Dimension to reduce over.
/// @param keepdim  If true, retain the reduced dimension with size 1.
/// @param device   Device to perform the operation on.
/// @return         Reduced tensor with maximum values.
/// @throws         MNException on invalid dimension.
MNTensor reduce_max(const MNTensor& input,
                    int64_t dim,
                    bool keepdim,
                    MNDevice& device);

/// Reduce min along a specified dimension.
///
/// @param input    Input tensor.
/// @param dim      Dimension to reduce over.
/// @param keepdim  If true, retain the reduced dimension with size 1.
/// @param device   Device to perform the operation on.
/// @return         Reduced tensor with minimum values.
/// @throws         MNException on invalid dimension.
MNTensor reduce_min(const MNTensor& input,
                    int64_t dim,
                    bool keepdim,
                    MNDevice& device);

/// Find indices of maximum values along a dimension.
///
/// @param input    Input tensor.
/// @param dim      Dimension to reduce over.
/// @param device   Device to perform the operation on.
/// @return         Int64 tensor with indices of max values (keepdim=false).
/// @throws         MNException on invalid dimension.
MNTensor argmax(const MNTensor& input,
                int64_t dim,
                MNDevice& device);

/// Find indices of minimum values along a dimension.
///
/// @param input    Input tensor.
/// @param dim      Dimension to reduce over.
/// @param device   Device to perform the operation on.
/// @return         Int64 tensor with indices of min values (keepdim=false).
/// @throws         MNException on invalid dimension.
MNTensor argmin(const MNTensor& input,
                int64_t dim,
                MNDevice& device);

} // namespace metal_native
