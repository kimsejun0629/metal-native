/**
 * @file bind_core.cpp
 * @brief Pybind11 bindings for metal_native core functionality.
 *
 * This file provides Python bindings for:
 * - Device queries and management
 * - Tensor construction and operations
 * - Buffer management
 * - Data type enums
 * - Memory utilities
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <metal_native/core/dtype.h>
#include <metal_native/core/error.h>
#include <metal_native/core/device.h>
#include <metal_native/core/tensor.h>
#include <metal_native/core/buffer.h>
#include <metal_native/core/shape.h>
#include <metal_native/future/fast_ops.h>
#include <metal_native/kernels/kernel_registry.h>
#include <metal_native/dispatch/command_pipeline.h>
#include <metal_native/memory/allocator.h>
#include <metal_native/ops/elementwise.h>
#include <metal_native/ops/matmul.h>

#include <cstring>

namespace py = pybind11;
using namespace metal_native;

// Forward declarations for functions to be implemented
namespace metal_native {
namespace python {

// Forward declarations for sub-module bindings
void bind_ops(py::module_& m);
void bind_profiling(py::module_& m);
void bind_interop(py::module_& m);

// Device queries
std::string device_name();
bool is_available();
bool supports_bfloat16();
py::dict device_properties();
void set_prefer_private_storage(bool enable);
bool prefer_private_storage();

// Initialization
void initialize();
void load_library(const std::string& path);

// Synchronization and memory management
void synchronize();
void set_lazy_commit(bool enable);
bool lazy_commit();
void empty_cache();
size_t memory_allocated();
size_t max_memory_allocated();
void reset_peak_stats();

// Random number generation
void set_seed(int64_t seed);

// Tensor operations (placeholders - will be implemented with actual Tensor class)
py::object tensor_from_numpy(py::array array, const std::string& dtype_name, bool requires_grad);
py::object tensor_empty(py::tuple shape, const std::string& dtype_name, bool requires_grad);
py::object tensor_zeros(py::tuple shape, const std::string& dtype_name, bool requires_grad);
py::object tensor_ones(py::tuple shape, const std::string& dtype_name, bool requires_grad);
py::tuple tensor_shape(py::object handle);
std::string tensor_dtype(py::object handle);
py::array tensor_to_numpy(py::object handle);
py::object tensor_item(py::object handle);

// Tensor arithmetic (placeholders)
py::object tensor_add(py::object lhs, py::object rhs);
py::object tensor_add_scalar(py::object tensor, double scalar);
py::object tensor_sub(py::object lhs, py::object rhs);
py::object tensor_sub_scalar(py::object tensor, double scalar);
py::object tensor_rsub_scalar(py::object tensor, double scalar);
py::object tensor_mul(py::object lhs, py::object rhs);
py::object tensor_mul_scalar(py::object tensor, double scalar);
py::object tensor_div(py::object lhs, py::object rhs);
py::object tensor_div_scalar(py::object tensor, double scalar);
py::object tensor_matmul(py::object lhs, py::object rhs);

// DLPack interop (placeholders)
py::object tensor_to_dlpack(py::object handle);
py::object tensor_from_dlpack(py::capsule capsule, bool requires_grad);

} // namespace python
} // namespace metal_native

/**
 * @brief Main pybind11 module definition.
 *
 * This creates the _C extension module that is imported by the Python package.
 */
PYBIND11_MODULE(_C, m) {
    m.doc() = "MetalNative C++ extension module";

    // Module version
    m.attr("__version__") = "0.1.0";

    // ---------------------------------------------------------------------------
    // MNTensor Class
    // ---------------------------------------------------------------------------
    py::class_<MNTensor, std::shared_ptr<MNTensor>>(m, "MNTensor")
        .def("shape", [](const MNTensor& t) {
            auto s = t.shape();
            py::list result;
            for (size_t i = 0; i < s.ndim(); ++i) {
                result.append(s[i]);
            }
            return result;
        })
        .def("dtype", [](const MNTensor& t) {
            return dtype_name(t.dtype());
        })
        .def("numel", &MNTensor::numel)
        .def("ndim", &MNTensor::ndim)
        .def("is_contiguous", &MNTensor::is_contiguous)
        .def("__repr__", [](const MNTensor& t) {
            return t.to_string();
        })
        .def("reshape", [](const MNTensor& t, py::list new_shape_list) {
            std::vector<int64_t> dims;
            dims.reserve(new_shape_list.size());
            for (auto item : new_shape_list) {
                dims.push_back(item.cast<int64_t>());
            }
            MNShape new_shape(dims);
            return std::make_shared<MNTensor>(t.reshape(new_shape));
        }, py::arg("new_shape"), "Reshape tensor (returns a view)")
        .def("slice", [](const MNTensor& t, int64_t dim, int64_t start, int64_t end) {
            return std::make_shared<MNTensor>(t.slice(dim, start, end));
        }, py::arg("dim"), py::arg("start"), py::arg("end") = -1,
           "Slice tensor along a dimension (returns a view)")
        .def("clone", [](const MNTensor& t) {
            return std::make_shared<MNTensor>(t.clone());
        }, "Deep-copy tensor into new contiguous allocation");

    // ---------------------------------------------------------------------------
    // QKVResult Struct
    // ---------------------------------------------------------------------------
    py::class_<fast::QKVResult>(m, "QKVResult")
        .def_readonly("q", &fast::QKVResult::q)
        .def_readonly("k", &fast::QKVResult::k)
        .def_readonly("v", &fast::QKVResult::v);

    // ---------------------------------------------------------------------------
    // Data Types
    // ---------------------------------------------------------------------------
    py::enum_<MNDType>(m, "DType", "Metal Native data type enumeration")
        .value("Float32", MNDType::Float32, "32-bit floating point")
        .value("Float16", MNDType::Float16, "16-bit floating point")
        .value("BFloat16", MNDType::BFloat16, "16-bit brain floating point")
        .value("Int64", MNDType::Int64, "64-bit signed integer")
        .value("Int32", MNDType::Int32, "32-bit signed integer")
        .value("Int16", MNDType::Int16, "16-bit signed integer")
        .value("Int8", MNDType::Int8, "8-bit signed integer")
        .value("UInt8", MNDType::UInt8, "8-bit unsigned integer")
        .value("Bool", MNDType::Bool, "Boolean")
        .export_values();

    // DType utility functions
    m.def("dtype_size", &dtype_size, "Get size in bytes of a dtype",
          py::arg("dtype"));
    m.def("dtype_name", &dtype_name, "Get human-readable name of a dtype",
          py::arg("dtype"));
    m.def("dtype_is_floating_point", &dtype_is_floating_point,
          "Check if dtype is floating point",
          py::arg("dtype"));
    m.def("dtype_is_integer", &dtype_is_integer,
          "Check if dtype is integer",
          py::arg("dtype"));
    m.def("dtype_is_signed", &dtype_is_signed,
          "Check if dtype is signed",
          py::arg("dtype"));

    // ---------------------------------------------------------------------------
    // Device Management
    // ---------------------------------------------------------------------------
    py::module_ device_module = m.def_submodule("device", "Device management");

    device_module.def("name", &python::device_name,
                     "Get the name of the Metal device");
    device_module.def("is_available", &python::is_available,
                     "Check if Metal is available");
    device_module.def("supports_bfloat16", &python::supports_bfloat16,
                     "Check if bfloat16 is supported");
    device_module.def("properties", &python::device_properties,
                     "Get device properties dictionary");
    device_module.def("set_prefer_private_storage", &python::set_prefer_private_storage,
                     "Enable/disable GPU-only (Private) storage for intermediate tensors",
                     py::arg("enable"));
    device_module.def("prefer_private_storage", &python::prefer_private_storage,
                     "Check if GPU-only (Private) storage is preferred for intermediate tensors");

    // Top-level aliases (used by Python wrapper device.py)
    m.def("device_name", &python::device_name,
          "Get the name of the Metal device");
    m.def("is_available", &python::is_available,
          "Check if Metal is available");
    m.def("supports_bfloat16", &python::supports_bfloat16,
          "Check if bfloat16 is supported");
    m.def("device_properties", &python::device_properties,
          "Get device properties dictionary");

    // ---------------------------------------------------------------------------
    // Initialization
    // ---------------------------------------------------------------------------
    m.def("initialize", &python::initialize,
          "Initialize metal_native runtime");
    m.def("load_library", &python::load_library,
          "Load Metal shader library from .metallib file",
          py::arg("path"));

    // ---------------------------------------------------------------------------
    // Synchronization and Memory
    // ---------------------------------------------------------------------------
    m.def("synchronize", &python::synchronize,
          "Block until all GPU operations complete");
    m.def("set_lazy_commit", &python::set_lazy_commit,
          "Enable/disable lazy commit mode (batch ops like PyTorch MPS)",
          py::arg("enable"));
    m.def("lazy_commit", &python::lazy_commit,
          "Check if lazy commit mode is enabled");
    m.def("empty_cache", &python::empty_cache,
          "Release cached GPU memory");
    m.def("memory_allocated", &python::memory_allocated,
          "Get current GPU memory allocated in bytes");
    m.def("max_memory_allocated", &python::max_memory_allocated,
          "Get peak GPU memory allocated in bytes");
    m.def("reset_peak_stats", &python::reset_peak_stats,
          "Reset peak memory statistics");

    // ---------------------------------------------------------------------------
    // Random Number Generation
    // ---------------------------------------------------------------------------
    m.def("set_seed", &python::set_seed,
          "Set random number generator seed",
          py::arg("seed"));

    // ---------------------------------------------------------------------------
    // Tensor Operations
    // ---------------------------------------------------------------------------
    // Note: These are placeholder bindings. Actual implementation will be
    // added when Tensor class is implemented in C++.

    m.def("tensor_from_numpy", &python::tensor_from_numpy,
          "Create tensor from NumPy array",
          py::arg("array"), py::arg("dtype"), py::arg("requires_grad") = false);

    m.def("tensor_empty", &python::tensor_empty,
          "Create empty tensor",
          py::arg("shape"), py::arg("dtype"), py::arg("requires_grad") = false);

    m.def("tensor_zeros", &python::tensor_zeros,
          "Create tensor filled with zeros",
          py::arg("shape"), py::arg("dtype"), py::arg("requires_grad") = false);

    m.def("tensor_ones", &python::tensor_ones,
          "Create tensor filled with ones",
          py::arg("shape"), py::arg("dtype"), py::arg("requires_grad") = false);

    m.def("tensor_shape", &python::tensor_shape,
          "Get tensor shape",
          py::arg("handle"));

    m.def("tensor_dtype", &python::tensor_dtype,
          "Get tensor dtype name",
          py::arg("handle"));

    m.def("tensor_to_numpy", &python::tensor_to_numpy,
          "Convert tensor to NumPy array",
          py::arg("handle"));

    m.def("tensor_item", &python::tensor_item,
          "Extract scalar value from single-element tensor",
          py::arg("handle"));

    // Arithmetic operations
    m.def("tensor_add", &python::tensor_add,
          "Element-wise addition",
          py::arg("lhs"), py::arg("rhs"));

    m.def("tensor_add_scalar", &python::tensor_add_scalar,
          "Add scalar to tensor",
          py::arg("tensor"), py::arg("scalar"));

    m.def("tensor_sub", &python::tensor_sub,
          "Element-wise subtraction",
          py::arg("lhs"), py::arg("rhs"));

    m.def("tensor_sub_scalar", &python::tensor_sub_scalar,
          "Subtract scalar from tensor",
          py::arg("tensor"), py::arg("scalar"));

    m.def("tensor_rsub_scalar", &python::tensor_rsub_scalar,
          "Subtract tensor from scalar",
          py::arg("tensor"), py::arg("scalar"));

    m.def("tensor_mul", &python::tensor_mul,
          "Element-wise multiplication",
          py::arg("lhs"), py::arg("rhs"));

    m.def("tensor_mul_scalar", &python::tensor_mul_scalar,
          "Multiply tensor by scalar",
          py::arg("tensor"), py::arg("scalar"));

    m.def("tensor_div", &python::tensor_div,
          "Element-wise division",
          py::arg("lhs"), py::arg("rhs"));

    m.def("tensor_div_scalar", &python::tensor_div_scalar,
          "Divide tensor by scalar",
          py::arg("tensor"), py::arg("scalar"));

    m.def("tensor_matmul", &python::tensor_matmul,
          "Matrix multiplication",
          py::arg("lhs"), py::arg("rhs"));

    // DLPack interop
    m.def("tensor_to_dlpack", &python::tensor_to_dlpack,
          "Export tensor to DLPack capsule",
          py::arg("handle"));

    m.def("tensor_from_dlpack", &python::tensor_from_dlpack,
          "Import tensor from DLPack capsule",
          py::arg("capsule"), py::arg("requires_grad") = false);

    // ---------------------------------------------------------------------------
    // Sub-module bindings
    // ---------------------------------------------------------------------------
    metal_native::python::bind_ops(m);
    metal_native::python::bind_profiling(m);
    metal_native::python::bind_interop(m);

    // ---------------------------------------------------------------------------
    // Error Handling
    // ---------------------------------------------------------------------------
    // Register exception translation for metal_native errors
    py::register_exception<metal_native::MNException>(m, "MetalNativeError");
}

// ---------------------------------------------------------------------------
// Implementation Stubs
// ---------------------------------------------------------------------------
// These are placeholder implementations that will be replaced with actual
// functionality once the C++ Tensor and Device classes are implemented.

namespace metal_native {
namespace python {

std::string device_name() {
    try {
        return MNDevice::instance().name();
    } catch (...) {
        return "Unknown Metal Device";
    }
}

bool is_available() {
    try {
        MNDevice::instance();
        return true;
    } catch (...) {
        return false;
    }
}

bool supports_bfloat16() {
    try {
        return MNDevice::instance().supports_bfloat16();
    } catch (...) {
        return false;
    }
}

py::dict device_properties() {
    auto& dev = MNDevice::instance();
    py::dict props;
    props["name"] = dev.name();
    props["cores"] = 0;  // Not directly queryable via MTLDevice
    props["memory"] = dev.recommended_max_working_set_size();
    props["bandwidth"] = 0.0;  // Not directly queryable via MTLDevice
    props["max_buffer_length"] = dev.max_buffer_length();
    props["supports_bfloat16"] = dev.supports_bfloat16();
    props["unified_memory"] = dev.has_unified_memory();
    props["recommended_working_set"] = dev.recommended_max_working_set_size();
    return props;
}

void initialize() {
    // Initialize device and load Metal shader library
    MNDevice::instance(); // Ensure device is created
}

void load_library(const std::string& path) {
    KernelRegistry::instance().load_library(path);
}

void synchronize() {
    MNDevice::instance().synchronize();
}

void set_lazy_commit(bool enable) {
    MNDevice::instance().command_pipeline().set_lazy_commit(enable);
}

bool lazy_commit() {
    return MNDevice::instance().command_pipeline().lazy_commit();
}

// ---------------------------------------------------------------------------
// Helper: parse dtype string to MNDType enum
// ---------------------------------------------------------------------------
static MNDType parse_dtype_str(const std::string& dtype_str) {
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

// ---------------------------------------------------------------------------
// Helper: extract int64_t dims from py::tuple
// ---------------------------------------------------------------------------
static std::vector<int64_t> tuple_to_dims(py::tuple shape) {
    std::vector<int64_t> dims;
    dims.reserve(shape.size());
    for (auto item : shape) {
        dims.push_back(item.cast<int64_t>());
    }
    return dims;
}

// ---------------------------------------------------------------------------
// Helper: map MNDType to numpy format string
// ---------------------------------------------------------------------------
static std::string dtype_to_numpy_format(MNDType dtype) {
    switch (dtype) {
        case MNDType::Float32:  return "f";
        case MNDType::Float16:  return "e";
        case MNDType::BFloat16: return "e";  // numpy has no bfloat16; approximate as float16
        case MNDType::Int64:    return "q";
        case MNDType::Int32:    return "i";
        case MNDType::Int16:    return "h";
        case MNDType::Int8:     return "b";
        case MNDType::UInt8:    return "B";
        case MNDType::Bool:     return "?";
        default: MN_THROW(MetalNativeError::InvalidArgument, "dtype_to_numpy_format: unsupported dtype");
    }
}

// ---------------------------------------------------------------------------
// Helper: create a scalar tensor filled with a value, matching a given dtype
// ---------------------------------------------------------------------------
static std::shared_ptr<MNTensor> make_scalar_tensor(double value, MNDType dtype, MNDevice& device) {
    auto t = std::make_shared<MNTensor>(MNTensor::empty(MNShape({1}), dtype, device));
    t->fill_(value);
    return t;
}

// Static RNG seed storage
static int64_t g_rng_seed = 0;

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

void empty_cache() {
    MNDevice::instance().allocator().empty_cache();
}

size_t memory_allocated() {
    return MNDevice::instance().allocator().stats().allocated_bytes;
}

size_t max_memory_allocated() {
    return MNDevice::instance().allocator().stats().peak_bytes;
}

void reset_peak_stats() {
    // AllocatorStats has no reset method; empty cache is the closest operation
    MNDevice::instance().allocator().empty_cache();
}

void set_prefer_private_storage(bool enable) {
    MNDevice::instance().set_prefer_private_storage(enable);
}

bool prefer_private_storage() {
    return MNDevice::instance().prefer_private_storage();
}

// ---------------------------------------------------------------------------
// Random number generation
// ---------------------------------------------------------------------------

void set_seed(int64_t seed) {
    g_rng_seed = seed;
}

// ---------------------------------------------------------------------------
// Tensor construction
// ---------------------------------------------------------------------------

py::object tensor_from_numpy(py::array array, const std::string& dtype_name, bool requires_grad) {
    MNDType dtype = parse_dtype_str(dtype_name);
    MNDevice& device = MNDevice::instance();

    py::buffer_info info = array.request();
    size_t nbytes = static_cast<size_t>(info.size) * dtype_size(dtype);

    std::vector<int64_t> shape_vec;
    shape_vec.reserve(info.shape.size());
    for (auto dim : info.shape) {
        shape_vec.push_back(static_cast<int64_t>(dim));
    }

    auto buf = std::make_shared<MNBuffer>(device, nbytes, StorageMode::Shared);
    std::memcpy(buf->data(), info.ptr, nbytes);

    MNShape mn_shape(std::move(shape_vec));
    auto strides = mn_shape.contiguous_strides();
    auto tensor = std::make_shared<MNTensor>(buf, mn_shape, std::move(strides), dtype, 0);
    return py::cast(tensor);
}

py::object tensor_empty(py::tuple shape, const std::string& dtype_name, bool requires_grad) {
    MNDType dtype = parse_dtype_str(dtype_name);
    MNDevice& device = MNDevice::instance();
    MNShape mn_shape(tuple_to_dims(shape));
    auto tensor = std::make_shared<MNTensor>(MNTensor::empty(mn_shape, dtype, device));
    return py::cast(tensor);
}

py::object tensor_zeros(py::tuple shape, const std::string& dtype_name, bool requires_grad) {
    MNDType dtype = parse_dtype_str(dtype_name);
    MNDevice& device = MNDevice::instance();
    MNShape mn_shape(tuple_to_dims(shape));
    auto tensor = std::make_shared<MNTensor>(MNTensor::zeros(mn_shape, dtype, device));
    return py::cast(tensor);
}

py::object tensor_ones(py::tuple shape, const std::string& dtype_name, bool requires_grad) {
    MNDType dtype = parse_dtype_str(dtype_name);
    MNDevice& device = MNDevice::instance();
    MNShape mn_shape(tuple_to_dims(shape));
    auto tensor = std::make_shared<MNTensor>(MNTensor::ones(mn_shape, dtype, device));
    return py::cast(tensor);
}

// ---------------------------------------------------------------------------
// Tensor queries
// ---------------------------------------------------------------------------

py::tuple tensor_shape(py::object handle) {
    auto t = handle.cast<std::shared_ptr<MNTensor>>();
    py::tuple result(t->ndim());
    for (size_t i = 0; i < t->ndim(); ++i) {
        result[i] = t->shape()[static_cast<int64_t>(i)];
    }
    return result;
}

std::string tensor_dtype(py::object handle) {
    auto t = handle.cast<std::shared_ptr<MNTensor>>();
    return dtype_name(t->dtype());
}

py::array tensor_to_numpy(py::object handle) {
    auto t = handle.cast<std::shared_ptr<MNTensor>>();
    auto& device = MNDevice::instance();
    device.synchronize();

    // If Private storage, convert to Shared first
    std::shared_ptr<MNTensor> readable_t = t;
    py::object base = handle;
    if (t->buffer()->storage_mode() == StorageMode::Private) {
        readable_t = std::make_shared<MNTensor>(t->to_shared());
        base = py::cast(readable_t);
    }

    size_t elem_size = dtype_size(readable_t->dtype());
    std::string fmt = dtype_to_numpy_format(readable_t->dtype());

    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides_bytes;
    shape.reserve(readable_t->ndim());
    strides_bytes.reserve(readable_t->ndim());
    for (size_t i = 0; i < readable_t->ndim(); ++i) {
        shape.push_back(static_cast<ssize_t>(readable_t->shape()[static_cast<int64_t>(i)]));
        strides_bytes.push_back(static_cast<ssize_t>(readable_t->strides()[i]) * static_cast<ssize_t>(elem_size));
    }

    return py::array(py::dtype(fmt), shape, strides_bytes, readable_t->raw_data(), base);
}

py::object tensor_item(py::object handle) {
    auto t = handle.cast<std::shared_ptr<MNTensor>>();
    MN_CHECK(t->numel() == 1, MetalNativeError::InvalidArgument,
             "tensor_item: tensor must have exactly 1 element");
    MNDevice::instance().synchronize();

    switch (t->dtype()) {
        case MNDType::Float32:
            return py::cast(*t->data_ptr<float>());
        case MNDType::Float16:
            // FP16 stored as uint16_t; use __fp16 on ARM or cast via float
            return py::cast(static_cast<float>(*reinterpret_cast<const __fp16*>(t->raw_data())));
        case MNDType::BFloat16: {
            // BFloat16: upper 16 bits of a float32
            uint16_t bits = *t->data_ptr<uint16_t>();
            uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
            float val;
            std::memcpy(&val, &f32_bits, sizeof(val));
            return py::cast(val);
        }
        case MNDType::Int64:
            return py::cast(*t->data_ptr<int64_t>());
        case MNDType::Int32:
            return py::cast(*t->data_ptr<int32_t>());
        case MNDType::Int16:
            return py::cast(*t->data_ptr<int16_t>());
        case MNDType::Int8:
            return py::cast(*t->data_ptr<int8_t>());
        case MNDType::UInt8:
            return py::cast(*t->data_ptr<uint8_t>());
        case MNDType::Bool:
            return py::cast(static_cast<bool>(*t->data_ptr<uint8_t>()));
        default:
            MN_THROW(MetalNativeError::InvalidArgument, "tensor_item: unsupported dtype");
    }
}

// ---------------------------------------------------------------------------
// Tensor-tensor arithmetic
// ---------------------------------------------------------------------------

py::object tensor_add(py::object lhs, py::object rhs) {
    auto a = lhs.cast<std::shared_ptr<MNTensor>>();
    auto b = rhs.cast<std::shared_ptr<MNTensor>>();
    MNDevice& device = MNDevice::instance();
    auto result = std::make_shared<MNTensor>(metal_native::add(*a, *b, device));
    return py::cast(result);
}

py::object tensor_sub(py::object lhs, py::object rhs) {
    auto a = lhs.cast<std::shared_ptr<MNTensor>>();
    auto b = rhs.cast<std::shared_ptr<MNTensor>>();
    MNDevice& device = MNDevice::instance();
    auto result = std::make_shared<MNTensor>(metal_native::sub(*a, *b, device));
    return py::cast(result);
}

py::object tensor_mul(py::object lhs, py::object rhs) {
    auto a = lhs.cast<std::shared_ptr<MNTensor>>();
    auto b = rhs.cast<std::shared_ptr<MNTensor>>();
    MNDevice& device = MNDevice::instance();
    auto result = std::make_shared<MNTensor>(metal_native::mul(*a, *b, device));
    return py::cast(result);
}

py::object tensor_div(py::object lhs, py::object rhs) {
    auto a = lhs.cast<std::shared_ptr<MNTensor>>();
    auto b = rhs.cast<std::shared_ptr<MNTensor>>();
    MNDevice& device = MNDevice::instance();
    auto result = std::make_shared<MNTensor>(metal_native::div(*a, *b, device));
    return py::cast(result);
}

// ---------------------------------------------------------------------------
// Tensor-scalar arithmetic
// ---------------------------------------------------------------------------

py::object tensor_add_scalar(py::object tensor, double scalar) {
    auto t = tensor.cast<std::shared_ptr<MNTensor>>();
    MNDevice& device = MNDevice::instance();
    auto scalar_t = make_scalar_tensor(scalar, t->dtype(), device);
    auto result = std::make_shared<MNTensor>(metal_native::add(*t, *scalar_t, device));
    return py::cast(result);
}

py::object tensor_sub_scalar(py::object tensor, double scalar) {
    auto t = tensor.cast<std::shared_ptr<MNTensor>>();
    MNDevice& device = MNDevice::instance();
    auto scalar_t = make_scalar_tensor(scalar, t->dtype(), device);
    auto result = std::make_shared<MNTensor>(metal_native::sub(*t, *scalar_t, device));
    return py::cast(result);
}

py::object tensor_rsub_scalar(py::object tensor, double scalar) {
    auto t = tensor.cast<std::shared_ptr<MNTensor>>();
    MNDevice& device = MNDevice::instance();
    auto scalar_t = make_scalar_tensor(scalar, t->dtype(), device);
    // rsub: scalar - tensor
    auto result = std::make_shared<MNTensor>(metal_native::sub(*scalar_t, *t, device));
    return py::cast(result);
}

py::object tensor_mul_scalar(py::object tensor, double scalar) {
    auto t = tensor.cast<std::shared_ptr<MNTensor>>();
    MNDevice& device = MNDevice::instance();
    auto scalar_t = make_scalar_tensor(scalar, t->dtype(), device);
    auto result = std::make_shared<MNTensor>(metal_native::mul(*t, *scalar_t, device));
    return py::cast(result);
}

py::object tensor_div_scalar(py::object tensor, double scalar) {
    auto t = tensor.cast<std::shared_ptr<MNTensor>>();
    MNDevice& device = MNDevice::instance();
    auto scalar_t = make_scalar_tensor(scalar, t->dtype(), device);
    auto result = std::make_shared<MNTensor>(metal_native::div(*t, *scalar_t, device));
    return py::cast(result);
}

// ---------------------------------------------------------------------------
// Matrix multiplication
// ---------------------------------------------------------------------------

py::object tensor_matmul(py::object lhs, py::object rhs) {
    auto a = lhs.cast<std::shared_ptr<MNTensor>>();
    auto b = rhs.cast<std::shared_ptr<MNTensor>>();
    auto result = std::make_shared<MNTensor>(metal_native::matmul(*a, *b));
    return py::cast(result);
}

// ---------------------------------------------------------------------------
// DLPack interop (Phase 3 - keep as stubs)
// ---------------------------------------------------------------------------

py::object tensor_to_dlpack(py::object handle) {
    throw std::runtime_error("DLPack interop not yet implemented (Phase 3)");
}

py::object tensor_from_dlpack(py::capsule capsule, bool requires_grad) {
    throw std::runtime_error("DLPack interop not yet implemented (Phase 3)");
}

} // namespace python
} // namespace metal_native
