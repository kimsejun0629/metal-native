#pragma once

/// @file matmul.h
/// @brief Matrix multiplication operators.
///
/// Provides standard and batched matrix multiplication using MPSGraph for
/// large matrices (M, N, K >= 128) and direct Metal Shading Language kernels
/// for smaller matrices. Supports transposition flags and multiple data types.

#include <cstddef>
#include <cstdint>

namespace metal_native {

// Forward declaration
class MNTensor;

/// Matrix multiplication: C = A @ B.
///
/// Performs standard 2D matrix multiplication. The input tensors must satisfy:
/// - A: shape [..., M, K]
/// - B: shape [..., K, N]
/// - Result: shape [..., M, N]
///
/// For large matrices (M, N, K >= 128), routes to MPSGraph for optimal
/// performance. For small matrices, uses direct MSL kernels.
///
/// @param a             Left operand tensor.
/// @param b             Right operand tensor.
/// @param transpose_a   Transpose A before multiplication (default: false).
/// @param transpose_b   Transpose B before multiplication (default: false).
/// @return              Result tensor C = A @ B.
/// @throws MNException(InvalidArgument) if shapes are incompatible.
MNTensor matmul(const MNTensor& a,
                const MNTensor& b,
                bool transpose_a = false,
                bool transpose_b = false);

/// Batched matrix multiplication: C[i] = A[i] @ B[i].
///
/// Performs batch-wise matrix multiplication on 3D or higher-dimensional
/// tensors. The batch dimensions must be broadcastable.
///
/// - A: shape [batch..., M, K]
/// - B: shape [batch..., K, N]
/// - Result: shape [batch..., M, N]
///
/// @param a             Left operand tensor.
/// @param b             Right operand tensor.
/// @param transpose_a   Transpose A before multiplication (default: false).
/// @param transpose_b   Transpose B before multiplication (default: false).
/// @return              Batched result tensor.
/// @throws MNException(InvalidArgument) if shapes are incompatible.
MNTensor batched_matmul(const MNTensor& a,
                        const MNTensor& b,
                        bool transpose_a = false,
                        bool transpose_b = false);

/// Quantization type for dequant_matmul
enum class QuantType {
    INT4,  ///< 4-bit integer quantization
    INT8   ///< 8-bit integer quantization
};

/// Fused dequantization + matrix multiplication.
///
/// Reads quantized weights (INT4 or INT8), dequantizes on-the-fly in shared
/// memory during matmul, eliminating the intermediate FP16 buffer.
///
/// @param activations  FP16 activation tensor [M, K]
/// @param weights_packed  Packed weight tensor [N, K/2] for INT4 or [N, K] for INT8
/// @param scales  Scale factors [N, K/group_size]
/// @param zeros   Zero points [N, K/group_size]
/// @param group_size  Quantization group size (32, 64, or 128)
/// @param quant_type  INT4 or INT8
/// @return Result tensor [M, N] in FP16
MNTensor dequant_matmul(const MNTensor& activations,
                         const MNTensor& weights_packed,
                         const MNTensor& scales,
                         const MNTensor& zeros,
                         uint32_t group_size,
                         QuantType quant_type);

} // namespace metal_native
