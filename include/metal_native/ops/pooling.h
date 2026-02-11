#pragma once

/// @file pooling.h
/// @brief Pooling operators.
///
/// Provides max pooling, average pooling, and adaptive pooling operations
/// using MPSGraph.

#include <cstddef>
#include <vector>

namespace metal_native {

// Forward declaration
class MNTensor;

/// 2D max pooling operation.
///
/// Performs max pooling over 4D input tensors (NCHW format).
///
/// @param input         Input tensor [N, C, H, W].
/// @param kernel_size   Pooling window size as [kH, kW].
/// @param stride        Stride as [stride_h, stride_w]. If empty, defaults to kernel_size.
/// @param padding       Padding as [pad_h, pad_w] (default: {0, 0}).
/// @return              Output tensor [N, C, H_out, W_out].
/// @throws MNException(InvalidArgument) if shapes or parameters are invalid.
MNTensor max_pool2d(const MNTensor& input,
                    const std::vector<size_t>& kernel_size,
                    const std::vector<size_t>& stride = {},
                    const std::vector<size_t>& padding = {0, 0});

/// 2D average pooling operation.
///
/// Performs average pooling over 4D input tensors (NCHW format).
///
/// @param input         Input tensor [N, C, H, W].
/// @param kernel_size   Pooling window size as [kH, kW].
/// @param stride        Stride as [stride_h, stride_w]. If empty, defaults to kernel_size.
/// @param padding       Padding as [pad_h, pad_w] (default: {0, 0}).
/// @return              Output tensor [N, C, H_out, W_out].
/// @throws MNException(InvalidArgument) if shapes or parameters are invalid.
MNTensor avg_pool2d(const MNTensor& input,
                    const std::vector<size_t>& kernel_size,
                    const std::vector<size_t>& stride = {},
                    const std::vector<size_t>& padding = {0, 0});

/// 2D adaptive average pooling operation.
///
/// Performs adaptive average pooling that produces a fixed-size output
/// regardless of input size.
///
/// @param input         Input tensor [N, C, H, W].
/// @param output_size   Target output size as [H_out, W_out].
/// @return              Output tensor [N, C, H_out, W_out].
/// @throws MNException(InvalidArgument) if shapes or parameters are invalid.
MNTensor adaptive_avg_pool2d(const MNTensor& input,
                             const std::vector<size_t>& output_size);

} // namespace metal_native
