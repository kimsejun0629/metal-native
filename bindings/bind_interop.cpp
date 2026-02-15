/**
 * @file bind_interop.cpp
 * @brief Pybind11 bindings for interoperability (Phase 3 placeholder).
 *
 * This file will provide Python bindings for:
 * - DLPack import/export
 * - PyTorch tensor conversion
 * - NumPy zero-copy views (on UMA)
 * - Accelerate framework integration
 *
 * Note: Implementation planned for Phase 3.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "metal_native/core/buffer.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/dtype.h"
#include "metal_native/core/error.h"
#include <string>
#include <cstring>
#include <dispatch/dispatch.h>

namespace py = pybind11;

namespace metal_native {
namespace python {

/**
 * @brief Convert dtype string to MNDType enum.
 */
static MNDType parse_dtype(const std::string& dtype_str) {
    if (dtype_str == "float32") return MNDType::Float32;
    if (dtype_str == "float16") return MNDType::Float16;
    if (dtype_str == "bfloat16") return MNDType::BFloat16;
    if (dtype_str == "int64") return MNDType::Int64;
    if (dtype_str == "int32") return MNDType::Int32;
    if (dtype_str == "int16") return MNDType::Int16;
    if (dtype_str == "int8") return MNDType::Int8;
    if (dtype_str == "uint8") return MNDType::UInt8;
    if (dtype_str == "bool") return MNDType::Bool;

    MN_THROW(MetalNativeError::InvalidArgument,
             "Unsupported dtype string: " + dtype_str);
}

/**
 * @brief Create MNTensor from external MPS pointer (zero-copy).
 *
 * This function wraps an existing MPS tensor's data pointer as an MNTensor
 * without copying data. The caller must ensure the external memory remains
 * valid for the lifetime of the returned tensor.
 *
 * @param data_ptr Raw pointer to MPS tensor data (from PyTorch .data_ptr())
 * @param nbytes Total size of the storage in bytes
 * @param shape Tensor shape
 * @param strides Tensor strides (element strides, not byte strides)
 * @param dtype_str Data type string ("float32", "float16", "bfloat16", etc.)
 * @return MNTensor wrapping the external memory
 */
static MNTensor tensor_from_mps_ptr(
    uintptr_t data_ptr,
    size_t nbytes,
    std::vector<int64_t> shape,
    std::vector<int64_t> strides,
    const std::string& dtype_str)
{
    // Convert dtype string to enum
    MNDType dtype = parse_dtype(dtype_str);

    // Wrap external pointer as MNBuffer (zero-copy).
    // MPS GPU pointers may not be page-aligned; wrap_external aligns
    // DOWN and returns the alignment offset via out_offset.
    size_t alignment_offset = 0;
    auto buffer = MNBuffer::wrap_external(
        MNDevice::instance(),
        reinterpret_cast<void*>(data_ptr),
        nbytes,
        nullptr,           // no release callback
        &alignment_offset   // receive page-alignment offset
    );

    // Construct MNTensor from buffer + shape + strides + dtype.
    // The alignment_offset ensures Metal encoders address the correct
    // data start within the page-aligned buffer.
    MNShape mn_shape(std::move(shape));
    return MNTensor(buffer, mn_shape, std::move(strides), dtype, alignment_offset);
}

/**
 * @brief Parallel memcpy using GCD dispatch_apply for large transfers.
 *
 * Splits the copy into 4MB chunks and distributes across CPU cores
 * via the global concurrent queue. Falls back to plain memcpy for
 * transfers <= 4MB.
 */
static void parallel_memcpy(void* dst, const void* src, size_t nbytes) {
    constexpr size_t kChunkSize = 4 * 1024 * 1024; // 4MB
    if (nbytes <= kChunkSize) {
        std::memcpy(dst, src, nbytes);
        return;
    }
    size_t nchunks = (nbytes + kChunkSize - 1) / kChunkSize;
    dispatch_apply(nchunks,
        dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
        ^(size_t i) {
            size_t offset = i * kChunkSize;
            size_t len = std::min(kChunkSize, nbytes - offset);
            memcpy(static_cast<char*>(dst) + offset,
                   static_cast<const char*>(src) + offset, len);
        });
}

/**
 * @brief Create MNTensor from a CPU-accessible buffer (numpy array).
 *
 * Allocates a Metal Shared buffer and copies data from the source.
 * This is the safe path for creating MNTensors when the source data
 * lives in CPU memory (e.g. numpy arrays, CPU PyTorch tensors).
 *
 * @param data_ptr CPU pointer to source data
 * @param nbytes Total bytes to copy
 * @param shape Tensor shape
 * @param dtype_str Data type string
 * @return MNTensor with data copied into a Metal buffer
 */
static MNTensor tensor_from_cpu_data(
    uintptr_t data_ptr,
    size_t nbytes,
    std::vector<int64_t> shape,
    const std::string& dtype_str)
{
    MNDType dtype = parse_dtype(dtype_str);
    MNDevice& device = MNDevice::instance();

    // Allocate Metal buffer and copy from CPU source (parallel for large buffers)
    auto buf = std::make_shared<MNBuffer>(device, nbytes, StorageMode::Shared);
    parallel_memcpy(buf->data(), reinterpret_cast<void*>(data_ptr), nbytes);

    MNShape mn_shape(std::move(shape));
    auto strides = mn_shape.contiguous_strides();
    return MNTensor(buf, mn_shape, std::move(strides), dtype, 0);
}

/**
 * @brief Allocate a Metal Shared buffer and return its CPU pointer + MNTensor.
 *
 * This enables the "direct buffer target" optimization: Python can create
 * a torch CPU tensor view over the returned pointer, then copy MPS data
 * directly into our Metal buffer via PyTorch's GPU blit — eliminating
 * the second CPU→Metal memcpy that the tensor_from_cpu_data path requires.
 *
 * @param shape Tensor shape
 * @param dtype_str Data type string
 * @return Tuple of (cpu_data_ptr as uintptr_t, MNTensor)
 */
static py::tuple allocate_tensor_for_copy(
    std::vector<int64_t> shape,
    const std::string& dtype_str)
{
    MNDType dtype = parse_dtype(dtype_str);
    MNDevice& device = MNDevice::instance();
    MNShape mn_shape(shape);
    MNTensor tensor = MNTensor::empty(mn_shape, dtype, device);
    uintptr_t data_ptr = reinterpret_cast<uintptr_t>(tensor.raw_data());
    return py::make_tuple(data_ptr, tensor);
}

void bind_interop(py::module_& m) {
    // Zero-copy MPS tensor bridge (works for small tensors; large MPS
    // tensors may SIGBUS — use tensor_from_cpu_data for those).
    m.def("tensor_from_mps_ptr", &tensor_from_mps_ptr,
          py::arg("data_ptr"),
          py::arg("nbytes"),
          py::arg("shape"),
          py::arg("strides"),
          py::arg("dtype"),
          "Create MNTensor from external MPS pointer (zero-copy)");

    // Safe CPU-based tensor creation (copies data into Metal buffer)
    m.def("tensor_from_cpu_data", &tensor_from_cpu_data,
          py::arg("data_ptr"),
          py::arg("nbytes"),
          py::arg("shape"),
          py::arg("dtype"),
          "Create MNTensor by copying from CPU memory into Metal buffer");

    // Direct buffer target: allocate Metal buffer, return CPU ptr + MNTensor
    m.def("allocate_tensor_for_copy", &allocate_tensor_for_copy,
          py::arg("shape"),
          py::arg("dtype"),
          "Allocate Metal Shared buffer and return (cpu_ptr, MNTensor) for direct copy");
}

} // namespace python
} // namespace metal_native
