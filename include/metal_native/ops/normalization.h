#pragma once

/// @file normalization.h
/// @brief Normalization operators: LayerNorm, RMSNorm, BatchNorm, GroupNorm.
///
/// All normalization operations use Welford's online algorithm for numerically
/// stable mean and variance computation, with SIMD-group reductions for
/// parallelism. Weight and bias are fused into the same kernel for efficiency.

#include <cstddef>

namespace metal_native {

class MNTensor;

/// Layer normalization: normalize over the last dimension(s).
///
/// Computes: y = (x - mean) / sqrt(variance + eps) * weight + bias
///
/// Normalizes each sample independently over the feature dimensions.
/// Uses Welford's algorithm for stable mean/variance calculation.
///
/// @param input   Input tensor: [..., normalized_dims]
/// @param weight  Weight (gamma) tensor with shape matching normalized dims.
/// @param bias    Bias (beta) tensor with shape matching normalized dims.
/// @param eps     Small constant for numerical stability (default: 1e-5).
/// @return        Output tensor with the same shape as input.
///
/// @throws MNException(InvalidArgument) if weight/bias shapes don't match
///         the normalized dimensions of input.
MNTensor layer_norm(const MNTensor& input,
                    const MNTensor& weight,
                    const MNTensor& bias,
                    float eps = 1e-5f);

/// Root Mean Square normalization.
///
/// Computes: y = x / sqrt(mean(x^2) + eps) * weight
///
/// RMSNorm is similar to LayerNorm but omits mean centering and bias,
/// which can be more efficient and equally effective in many architectures.
///
/// @param input   Input tensor: [..., normalized_dims]
/// @param weight  Weight (gamma) tensor with shape matching normalized dims.
/// @param eps     Small constant for numerical stability (default: 1e-5).
/// @return        Output tensor with the same shape as input.
///
/// @throws MNException(InvalidArgument) if weight shape doesn't match
///         the normalized dimensions of input.
MNTensor rms_norm(const MNTensor& input,
                  const MNTensor& weight,
                  float eps = 1e-5f);

/// Batch normalization: normalize over the batch dimension.
///
/// Training mode:
///   Computes batch statistics and updates running mean/variance.
///   y = (x - batch_mean) / sqrt(batch_var + eps) * weight + bias
///
/// Inference mode:
///   Uses pre-computed running statistics.
///   y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
///
/// @param input          Input tensor: [batch, channels, ...spatial_dims]
/// @param running_mean   Running mean: [channels] (updated in training mode)
/// @param running_var    Running variance: [channels] (updated in training mode)
/// @param weight         Weight (gamma): [channels]
/// @param bias           Bias (beta): [channels]
/// @param training       If true, compute batch statistics and update running stats.
///                       If false, use running_mean and running_var.
/// @param momentum       Momentum for running statistics update (default: 0.1).
/// @param eps            Small constant for numerical stability (default: 1e-5).
/// @return               Output tensor with the same shape as input.
///
/// @throws MNException(InvalidArgument) if tensor shapes are incompatible.
MNTensor batch_norm(const MNTensor& input,
                    MNTensor& running_mean,
                    MNTensor& running_var,
                    const MNTensor& weight,
                    const MNTensor& bias,
                    bool training,
                    float momentum = 0.1f,
                    float eps = 1e-5f);

/// Group normalization: divide channels into groups and normalize each group.
///
/// Computes: y = (x - group_mean) / sqrt(group_var + eps) * weight + bias
///
/// Splits the channel dimension into num_groups and normalizes each group
/// independently. Useful for small batch sizes where BatchNorm is unstable.
///
/// @param input       Input tensor: [batch, channels, ...spatial_dims]
/// @param num_groups  Number of groups to divide channels into.
///                    Channels must be divisible by num_groups.
/// @param weight      Weight (gamma): [channels]
/// @param bias        Bias (beta): [channels]
/// @param eps         Small constant for numerical stability (default: 1e-5).
/// @return            Output tensor with the same shape as input.
///
/// @throws MNException(InvalidArgument) if channels % num_groups != 0 or
///         if weight/bias shapes don't match channels.
MNTensor group_norm(const MNTensor& input,
                    size_t num_groups,
                    const MNTensor& weight,
                    const MNTensor& bias,
                    float eps = 1e-5f);

} // namespace metal_native
