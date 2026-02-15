/// @file dlpack.cpp
/// @brief Implementation of DLPack tensor exchange.

#include "metal_native/interop/dlpack.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

#include <cstring>
#include <memory>

namespace metal_native {

namespace {

// ---------------------------------------------------------------------------
// DType conversion helpers
// ---------------------------------------------------------------------------

DLDataType mndtype_to_dldtype(MNDType dtype) {
    DLDataType dl_dtype;
    dl_dtype.lanes = 1;  // Scalar types only

    switch (dtype) {
        case MNDType::Float32:
            dl_dtype.code = kDLFloat;
            dl_dtype.bits = 32;
            break;
        case MNDType::Float16:
            dl_dtype.code = kDLFloat;
            dl_dtype.bits = 16;
            break;
        case MNDType::BFloat16:
            dl_dtype.code = kDLBfloat;
            dl_dtype.bits = 16;
            break;
        case MNDType::Int64:
            dl_dtype.code = kDLInt;
            dl_dtype.bits = 64;
            break;
        case MNDType::Int32:
            dl_dtype.code = kDLInt;
            dl_dtype.bits = 32;
            break;
        case MNDType::Int16:
            dl_dtype.code = kDLInt;
            dl_dtype.bits = 16;
            break;
        case MNDType::Int8:
            dl_dtype.code = kDLInt;
            dl_dtype.bits = 8;
            break;
        case MNDType::UInt8:
            dl_dtype.code = kDLUInt;
            dl_dtype.bits = 8;
            break;
        case MNDType::Bool:
            dl_dtype.code = kDLUInt;
            dl_dtype.bits = 8;
            break;
        default:
            MN_THROW(MetalNativeError::InvalidArgument,
                     "Unsupported dtype for DLPack conversion");
    }

    return dl_dtype;
}

MNDType dldtype_to_mndtype(DLDataType dl_dtype) {
    MN_CHECK(dl_dtype.lanes == 1,
             MetalNativeError::InvalidArgument,
             "DLPack vector types not supported");

    if (dl_dtype.code == kDLFloat) {
        if (dl_dtype.bits == 32) return MNDType::Float32;
        if (dl_dtype.bits == 16) return MNDType::Float16;
    } else if (dl_dtype.code == kDLBfloat) {
        if (dl_dtype.bits == 16) return MNDType::BFloat16;
    } else if (dl_dtype.code == kDLInt) {
        if (dl_dtype.bits == 64) return MNDType::Int64;
        if (dl_dtype.bits == 32) return MNDType::Int32;
        if (dl_dtype.bits == 16) return MNDType::Int16;
        if (dl_dtype.bits == 8)  return MNDType::Int8;
    } else if (dl_dtype.code == kDLUInt) {
        if (dl_dtype.bits == 8)  return MNDType::UInt8;
    }

    MN_THROW(MetalNativeError::InvalidArgument,
             "Unsupported DLDataType: code=" + std::to_string(dl_dtype.code) +
             ", bits=" + std::to_string(dl_dtype.bits));
}

// ---------------------------------------------------------------------------
// Manager context for owned tensors
// ---------------------------------------------------------------------------

/// Context stored in DLManagedTensor for deletion.
struct DLPackContext {
    std::shared_ptr<MNBuffer> buffer;  // Keep buffer alive
    int64_t* shape;                     // Owned shape array
    int64_t* strides;                   // Owned strides array
};

/// Deleter callback for DLManagedTensor.
void dlpack_deleter(DLManagedTensor* self) {
    if (!self) return;

    auto* ctx = static_cast<DLPackContext*>(self->manager_ctx);
    delete[] ctx->shape;
    delete[] ctx->strides;
    delete ctx;
    delete self;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// DLPackExporter
// ---------------------------------------------------------------------------

DLManagedTensor* DLPackExporter::to_dlpack(const MNTensor& tensor) {
    MN_CHECK(tensor.is_contiguous(),
             MetalNativeError::InvalidArgument,
             "DLPack export requires contiguous tensor");

    // Allocate the managed tensor structure
    auto* managed = new DLManagedTensor;
    std::memset(managed, 0, sizeof(DLManagedTensor));

    // Allocate context to keep the buffer alive
    auto* ctx = new DLPackContext;
    ctx->buffer = tensor.buffer();

    // Copy shape
    const auto& shape = tensor.shape();
    ctx->shape = new int64_t[shape.ndim()];
    for (size_t i = 0; i < shape.ndim(); ++i) {
        ctx->shape[i] = shape[i];
    }

    // Copy strides
    const auto& strides = tensor.strides();
    ctx->strides = new int64_t[strides.size()];
    for (size_t i = 0; i < strides.size(); ++i) {
        ctx->strides[i] = strides[i];
    }

    // Fill DLTensor
    managed->dl_tensor.data = const_cast<void*>(tensor.raw_data());
    managed->dl_tensor.device.device_type = kDLMetal;
    managed->dl_tensor.device.device_id = 0;
    managed->dl_tensor.ndim = static_cast<int32_t>(tensor.ndim());
    managed->dl_tensor.dtype = mndtype_to_dldtype(tensor.dtype());
    managed->dl_tensor.shape = ctx->shape;
    managed->dl_tensor.strides = ctx->strides;
    managed->dl_tensor.byte_offset = tensor.offset();

    // Attach context and deleter
    managed->manager_ctx = ctx;
    managed->deleter = dlpack_deleter;

    return managed;
}

// ---------------------------------------------------------------------------
// DLPackImporter
// ---------------------------------------------------------------------------

MNTensor DLPackImporter::from_dlpack(DLManagedTensor* dl_tensor) {
    MN_CHECK(dl_tensor != nullptr,
             MetalNativeError::InvalidArgument,
             "DLManagedTensor is null");

    const DLTensor& dl = dl_tensor->dl_tensor;

    // Convert dtype
    MNDType dtype = dldtype_to_mndtype(dl.dtype);

    // Build shape
    std::vector<int64_t> shape_vec;
    shape_vec.reserve(dl.ndim);
    for (int32_t i = 0; i < dl.ndim; ++i) {
        shape_vec.push_back(dl.shape[i]);
    }
    MNShape shape(std::move(shape_vec));

    // Build strides (default to row-major if null)
    std::vector<int64_t> strides;
    if (dl.strides != nullptr) {
        strides.assign(dl.strides, dl.strides + dl.ndim);
    } else {
        // Compute row-major strides
        strides.resize(dl.ndim);
        int64_t stride = 1;
        for (int32_t i = dl.ndim - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= dl.shape[i];
        }
    }

    // Handle device type
    if (dl.device.device_type == kDLMetal) {
        // Zero-copy path: wrap existing Metal buffer
        // Extract data pointer with byte offset
        void* data = static_cast<uint8_t*>(dl.data) + dl.byte_offset;

        // Compute total bytes from shape and dtype
        size_t total_bytes = dtype_size(dtype);
        for (int32_t i = 0; i < dl.ndim; ++i) {
            total_bytes *= dl.shape[i];
        }

        // Wrap external pointer as MNBuffer (zero-copy)
        size_t alignment_offset = 0;
        auto buffer = MNBuffer::wrap_external(
            MNDevice::instance(),
            data,
            total_bytes,
            nullptr,
            &alignment_offset
        );

        // Construct and return MNTensor
        return MNTensor(buffer, shape, strides, dtype, alignment_offset);
    } else if (dl.device.device_type == kDLCPU ||
               dl.device.device_type == kDLCUDAHost) {
        // Copy from CPU memory to GPU
        auto& device = MNDevice::instance();
        MNTensor result = MNTensor::empty(shape, dtype, device);

        // Copy data
        const size_t nbytes = result.nbytes();
        std::memcpy(result.raw_data(),
                    static_cast<uint8_t*>(dl.data) + dl.byte_offset,
                    nbytes);

        return result;
    } else {
        MN_THROW(MetalNativeError::InvalidArgument,
                 "Unsupported DLPack device type: " +
                 std::to_string(dl.device.device_type));
    }
}

} // namespace metal_native
