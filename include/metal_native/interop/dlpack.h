#pragma once

/// @file dlpack.h
/// @brief DLPack tensor exchange protocol for zero-copy interop.
///
/// Provides DLPackExporter and DLPackImporter for converting between
/// MNTensor and DLManagedTensor. This enables zero-copy tensor exchange
/// with frameworks that support the DLPack protocol (PyTorch, JAX, etc.).
///
/// Example usage:
/// @code
///   // Export to DLPack
///   MNTensor tensor = MNTensor::zeros({2, 3}, MNDType::Float32, device);
///   DLManagedTensor* dl_tensor = DLPackExporter::to_dlpack(tensor);
///
///   // Import from DLPack
///   MNTensor imported = DLPackImporter::from_dlpack(dl_tensor);
///
///   // Release after use (calls deleter)
///   dl_tensor->deleter(dl_tensor);
/// @endcode

#include <memory>

#include "metal_native/interop/dlpack_header.h"

namespace metal_native {

// Forward declaration
class MNTensor;

// ---------------------------------------------------------------------------
// DLPack Exporter
// ---------------------------------------------------------------------------

/// Export MNTensor to DLPack format.
class DLPackExporter {
public:
    /// Convert an MNTensor to a DLManagedTensor.
    ///
    /// The returned DLManagedTensor holds a shared reference to the
    /// underlying MNBuffer. The deleter callback releases this reference
    /// when called.
    ///
    /// @param tensor  The tensor to export.
    /// @return        A new DLManagedTensor that must be freed by calling
    ///                its deleter exactly once.
    /// @throws MNException(InvalidArgument) if the tensor is not contiguous.
    static DLManagedTensor* to_dlpack(const MNTensor& tensor);

    // Non-instantiable utility class
    DLPackExporter() = delete;
};

// ---------------------------------------------------------------------------
// DLPack Importer
// ---------------------------------------------------------------------------

/// Import DLPack tensors as MNTensor.
class DLPackImporter {
public:
    /// Convert a DLManagedTensor to an MNTensor.
    ///
    /// Zero-copy when the device type is kDLMetal and the data is already
    /// backed by a Metal buffer. Otherwise, the data is copied to GPU memory.
    ///
    /// @param dl_tensor  The DLPack tensor to import. Ownership is NOT
    ///                   transferred; the caller must still call the deleter.
    /// @return           A new MNTensor wrapping or copying the data.
    /// @throws MNException(InvalidArgument) if the device type is unsupported.
    /// @throws MNException(AllocationFailed) if GPU allocation fails.
    static MNTensor from_dlpack(DLManagedTensor* dl_tensor);

    // Non-instantiable utility class
    DLPackImporter() = delete;
};

} // namespace metal_native
