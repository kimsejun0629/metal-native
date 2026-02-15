#pragma once

/// @file fused_matmul.h
/// @brief Matrix multiplication with fused epilogue operations.
///
/// Provides matmul with epilogue fusion (bias, activation, residual) to eliminate
/// memory round-trips. The epilogue operation is applied during the matmul output
/// write step, avoiding a separate kernel dispatch.

#include <cstddef>

namespace metal_native {

// Forward declaration
class MNTensor;

/// Epilogue operation type for fused matmul.
enum class MatmulEpilogue {
    None = 0,        ///< No epilogue: C = A @ B
    Bias = 1,        ///< Bias addition: C = A @ B + bias
    BiasReLU = 2,    ///< Bias + ReLU: C = ReLU(A @ B + bias)
    BiasSiLU = 3,    ///< Bias + SiLU: C = SiLU(A @ B + bias)
    ResidualAdd = 4  ///< Residual connection: C = A @ B + residual
};

/// Fused matrix multiplication with epilogue operation.
///
/// Performs C = epilogue(A @ B, bias/residual) in a single kernel dispatch.
/// The epilogue operation is applied during the output write step, eliminating
/// the memory round-trip required by separate kernel dispatches.
///
/// Performance benefit: 1.1-1.2x speedup over separate ops for common transformer
/// patterns (bias + activation, residual connections).
///
/// Epilogue types:
/// - None: C = A @ B (equivalent to regular matmul)
/// - Bias: C = A @ B + bias (bias is [N] shaped, broadcast across M)
/// - BiasReLU: C = ReLU(A @ B + bias)
/// - BiasSiLU: C = SiLU(A @ B + bias) (for gated FFN)
/// - ResidualAdd: C = A @ B + residual (residual is [M, N] shaped)
///
/// @param a             Left operand tensor [M, K].
/// @param b             Right operand tensor [K, N].
/// @param epilogue      Type of epilogue operation to apply.
/// @param bias          Bias tensor [N] (required for Bias/BiasReLU/BiasSiLU).
/// @param residual      Residual tensor [M, N] (required for ResidualAdd).
/// @param transpose_a   Transpose A before multiplication (default: false).
/// @param transpose_b   Transpose B before multiplication (default: false).
/// @return              Result tensor C with epilogue applied.
/// @throws MNException(InvalidArgument) if shapes are incompatible or required
///                                       epilogue tensors are nullptr.
///
/// Example usage:
/// @code
///   // Bias + SiLU (common in gated FFN):
///   MNTensor bias = MNTensor::zeros({N}, MNDType::Float16, device);
///   MNTensor result = fused_matmul(a, b, MatmulEpilogue::BiasSiLU, &bias);
///
///   // Residual connection (common in transformer attention):
///   MNTensor residual = /* ... */;
///   MNTensor result = fused_matmul(a, b, MatmulEpilogue::ResidualAdd,
///                                  nullptr, &residual);
/// @endcode
MNTensor fused_matmul(const MNTensor& a,
                      const MNTensor& b,
                      MatmulEpilogue epilogue,
                      const MNTensor* bias = nullptr,
                      const MNTensor* residual = nullptr,
                      bool transpose_a = false,
                      bool transpose_b = false);

} // namespace metal_native
