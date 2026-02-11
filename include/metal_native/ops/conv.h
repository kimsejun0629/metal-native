#pragma once

/// @file conv.h
/// @brief Convolution operators.
///
/// Provides 1D, 2D, and 3D convolution operations using MPSGraph. Supports
/// grouped convolution, depthwise separable convolution, and various padding
/// modes.

#include <cstddef>
#include <vector>

namespace metal_native {

// Forward declaration
class MNTensor;

/// 2D convolution operation.
///
/// Performs 2D convolution on 4D input tensors (NCHW format).
///
/// @param input      Input tensor [N, C_in, H, W].
/// @param weight     Weight tensor [C_out, C_in/groups, kH, kW].
/// @param bias       Optional bias tensor [C_out]. Pass empty tensor for no bias.
/// @param stride     Stride as [stride_h, stride_w] (default: {1, 1}).
/// @param padding    Padding as [pad_h, pad_w] (default: {0, 0}).
/// @param dilation   Dilation as [dilation_h, dilation_w] (default: {1, 1}).
/// @param groups     Number of groups for grouped convolution (default: 1).
/// @return           Output tensor [N, C_out, H_out, W_out].
/// @throws MNException(InvalidArgument) if shapes or parameters are invalid.
MNTensor conv2d(const MNTensor& input,
                const MNTensor& weight,
                const MNTensor& bias,
                const std::vector<size_t>& stride = {1, 1},
                const std::vector<size_t>& padding = {0, 0},
                const std::vector<size_t>& dilation = {1, 1},
                size_t groups = 1);

/// 1D convolution operation (stub).
///
/// @note This function is not yet implemented and will throw an exception.
///
/// @param input      Input tensor [N, C_in, L].
/// @param weight     Weight tensor [C_out, C_in/groups, kL].
/// @param bias       Optional bias tensor [C_out].
/// @param stride     Stride value (default: 1).
/// @param padding    Padding value (default: 0).
/// @param dilation   Dilation value (default: 1).
/// @param groups     Number of groups (default: 1).
/// @return           Output tensor [N, C_out, L_out].
/// @throws MNException(InternalError) - not implemented.
MNTensor conv1d(const MNTensor& input,
                const MNTensor& weight,
                const MNTensor& bias,
                size_t stride = 1,
                size_t padding = 0,
                size_t dilation = 1,
                size_t groups = 1);

/// 3D convolution operation (stub).
///
/// @note This function is not yet implemented and will throw an exception.
///
/// @param input      Input tensor [N, C_in, D, H, W].
/// @param weight     Weight tensor [C_out, C_in/groups, kD, kH, kW].
/// @param bias       Optional bias tensor [C_out].
/// @param stride     Stride as [stride_d, stride_h, stride_w] (default: {1, 1, 1}).
/// @param padding    Padding as [pad_d, pad_h, pad_w] (default: {0, 0, 0}).
/// @param dilation   Dilation as [dilation_d, dilation_h, dilation_w] (default: {1, 1, 1}).
/// @param groups     Number of groups (default: 1).
/// @return           Output tensor [N, C_out, D_out, H_out, W_out].
/// @throws MNException(InternalError) - not implemented.
MNTensor conv3d(const MNTensor& input,
                const MNTensor& weight,
                const MNTensor& bias,
                const std::vector<size_t>& stride = {1, 1, 1},
                const std::vector<size_t>& padding = {0, 0, 0},
                const std::vector<size_t>& dilation = {1, 1, 1},
                size_t groups = 1);

} // namespace metal_native
