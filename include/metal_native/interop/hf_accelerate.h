#pragma once

/// @file hf_accelerate.h
/// @brief HuggingFace Accelerate integration stubs.
///
/// Provides a minimal C++ interface for HuggingFace Accelerate backend
/// registration. The actual integration is primarily Python-side; this
/// layer exposes hooks that Python bindings can call to interact with
/// MetalNative's device management.
///
/// Example usage (from Python side):
/// @code{.py}
///   from accelerate import Accelerator
///   accelerator = Accelerator(device_placement=True)
///   # Backend automatically detects metal_native device
/// @endcode

#include <string>

namespace metal_native {

// Forward declarations
struct TorchTensorMeta;

// ---------------------------------------------------------------------------
// Accelerate Backend Interface
// ---------------------------------------------------------------------------

/// HuggingFace Accelerate backend interface for MetalNative.
///
/// This class provides the minimal interface required for HuggingFace
/// Accelerate to recognize and use MetalNative as a compute backend.
class AccelerateBackend {
public:
    /// Return the device name for this backend.
    ///
    /// @return "metal_native"
    static std::string device_name();

    /// Check if the backend is available.
    ///
    /// @return true if MetalNative device can be initialized, false otherwise.
    static bool is_available();

    /// Stub for moving tensor data to the MetalNative device.
    ///
    /// This would be called from Python to transfer a tensor to GPU memory.
    /// The actual implementation happens in the Python bindings layer.
    ///
    /// @param tensor_meta  Metadata describing the tensor to move.
    /// @return             Updated metadata after device placement.
    static TorchTensorMeta move_to_device(const TorchTensorMeta& tensor_meta);

    /// Synchronize the GPU command queue.
    ///
    /// Blocks the calling thread until all pending GPU operations complete.
    static void synchronize();

    // Non-instantiable utility class
    AccelerateBackend() = delete;
};

} // namespace metal_native
