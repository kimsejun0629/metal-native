/// @file torch_interop.cpp
/// @brief Implementation of PyTorch C++ integration.

#include "metal_native/interop/torch_interop.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/dtype.h"
#include "metal_native/core/error.h"

namespace metal_native {

namespace {

// ---------------------------------------------------------------------------
// DType name mapping
// ---------------------------------------------------------------------------

/// Convert MNDType to PyTorch dtype string.
std::string mndtype_to_torch_name(MNDType dtype) {
    switch (dtype) {
        case MNDType::Float32:  return "float32";
        case MNDType::Float16:  return "float16";
        case MNDType::BFloat16: return "bfloat16";
        case MNDType::Int64:    return "int64";
        case MNDType::Int32:    return "int32";
        case MNDType::Int16:    return "int16";
        case MNDType::Int8:     return "int8";
        case MNDType::UInt8:    return "uint8";
        case MNDType::Bool:     return "bool";
        default:
            MN_THROW(MetalNativeError::InvalidArgument,
                     "Unknown dtype for PyTorch conversion");
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// TorchInterop implementation
// ---------------------------------------------------------------------------

bool TorchInterop::is_torch_available() {
    // Stub: Always returns false until pybind11 bindings are wired up.
    // In production, this would attempt to dlopen libtorch or check for
    // torch C++ symbols.
    return false;
}

void TorchInterop::register_custom_ops() {
    // Stub: Custom op registration with PyTorch dispatcher.
    // This would use TORCH_LIBRARY macro or the C++ dispatcher API to
    // register MetalNative ops under torch.ops.metal_native.*.
    //
    // Example (pseudo-code):
    //   TORCH_LIBRARY(metal_native, m) {
    //       m.def("matmul", &torch_matmul_wrapper);
    //       m.def("conv2d", &torch_conv2d_wrapper);
    //   }
    //
    // For now, this is a no-op stub.
}

TorchTensorMeta TorchInterop::tensor_to_torch_metadata(const MNTensor& tensor) {
    TorchTensorMeta meta;

    // Extract shape
    const auto& shape = tensor.shape();
    meta.shape.reserve(shape.ndim());
    for (size_t i = 0; i < shape.ndim(); ++i) {
        meta.shape.push_back(shape[i]);
    }

    // Extract strides
    meta.strides = tensor.strides();

    // Convert dtype to PyTorch name
    meta.dtype_str = mndtype_to_torch_name(tensor.dtype());

    // Raw pointer and byte size
    meta.data_ptr = const_cast<void*>(tensor.raw_data());
    meta.nbytes = tensor.nbytes();

    return meta;
}

} // namespace metal_native
