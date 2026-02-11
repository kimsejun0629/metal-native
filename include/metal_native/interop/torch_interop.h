#pragma once

/// @file torch_interop.h
/// @brief PyTorch C++ integration layer for MetalNative.
///
/// Provides utilities for bridging MNTensor with PyTorch's C++ API.
/// This layer prepares metadata for Python-side tensor conversion
/// and registers custom ops with PyTorch's dispatcher.
///
/// Example usage:
/// @code
///   // Check if PyTorch is available
///   if (TorchInterop::is_torch_available()) {
///       TorchInterop::register_custom_ops();
///   }
///
///   // Export tensor metadata for Python conversion
///   MNTensor tensor = ...;
///   auto meta = TorchInterop::tensor_to_torch_metadata(tensor);
///   // Pass meta to Python side for torch.from_numpy() or similar
/// @endcode

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace metal_native {

// Forward declaration
class MNTensor;

// ---------------------------------------------------------------------------
// Metadata structures
// ---------------------------------------------------------------------------

/// Metadata extracted from an MNTensor for PyTorch conversion.
///
/// This struct carries the essential information needed by the Python side
/// to construct a torch.Tensor without copying the underlying data.
struct TorchTensorMeta {
    std::vector<int64_t> shape;      ///< Tensor shape
    std::vector<int64_t> strides;    ///< Element strides (not byte strides)
    std::string dtype_str;           ///< PyTorch dtype name (e.g., "float32")
    void* data_ptr;                  ///< Raw pointer to tensor data
    size_t nbytes;                   ///< Total size in bytes
};

// ---------------------------------------------------------------------------
// PyTorch Interop
// ---------------------------------------------------------------------------

/// Static utilities for PyTorch integration.
class TorchInterop {
public:
    /// Check if PyTorch is available at runtime.
    ///
    /// This stub always returns false until pybind11 bindings are wired up.
    /// In production, this would attempt to load PyTorch's shared library
    /// or check for the presence of torch C++ symbols.
    ///
    /// @return true if PyTorch is loadable, false otherwise.
    static bool is_torch_available();

    /// Register MetalNative custom operators with PyTorch's dispatcher.
    ///
    /// This allows PyTorch to call MetalNative ops via torch.ops.metal_native.*.
    /// Currently a stub; implementation depends on PyTorch's C++ dispatcher API.
    ///
    /// @throws MNException(InternalError) if registration fails.
    static void register_custom_ops();

    /// Extract PyTorch-compatible metadata from an MNTensor.
    ///
    /// The returned metadata can be used on the Python side to construct
    /// a torch.Tensor via torch.as_tensor() or similar, enabling zero-copy
    /// sharing when the MNBuffer is accessible from Python.
    ///
    /// @param tensor  The tensor to extract metadata from.
    /// @return        Metadata struct with shape, strides, dtype, and pointer.
    static TorchTensorMeta tensor_to_torch_metadata(const MNTensor& tensor);

    // Non-instantiable utility class
    TorchInterop() = delete;
};

} // namespace metal_native
