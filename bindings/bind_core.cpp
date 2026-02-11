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

// Initialization
void initialize();

// Synchronization and memory management
void synchronize();
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

    // ---------------------------------------------------------------------------
    // Initialization
    // ---------------------------------------------------------------------------
    m.def("initialize", &python::initialize,
          "Initialize metal_native runtime");

    // ---------------------------------------------------------------------------
    // Synchronization and Memory
    // ---------------------------------------------------------------------------
    m.def("synchronize", &python::synchronize,
          "Block until all GPU operations complete");
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
    // TODO: Implement with actual Metal device query
    return "Apple M-Series GPU (Placeholder)";
}

bool is_available() {
    // TODO: Implement actual Metal availability check
    return true;
}

bool supports_bfloat16() {
    // TODO: Check actual device capabilities
    return true;
}

py::dict device_properties() {
    // TODO: Query actual device properties
    py::dict props;
    props["name"] = "Apple M-Series GPU";
    props["cores"] = 32;
    props["memory"] = 16ULL * 1024 * 1024 * 1024;  // 16 GB
    props["bandwidth"] = 400.0;  // GB/s
    props["max_buffer_length"] = 1ULL << 32;
    props["supports_bfloat16"] = true;
    props["unified_memory"] = true;
    props["recommended_working_set"] = 12ULL * 1024 * 1024 * 1024;  // 12 GB
    return props;
}

void initialize() {
    // TODO: Initialize device, allocator, etc.
}

void synchronize() {
    MNDevice::instance().synchronize();
}

void empty_cache() {
    // TODO: Free cached allocations
}

size_t memory_allocated() {
    // TODO: Return current GPU memory usage
    return 0;
}

size_t max_memory_allocated() {
    // TODO: Return peak GPU memory usage
    return 0;
}

void reset_peak_stats() {
    // TODO: Reset peak memory tracking
}

void set_seed(int64_t seed) {
    // TODO: Set RNG seed
}

// Tensor operation stubs - these will throw until Tensor is implemented
py::object tensor_from_numpy(py::array array, const std::string& dtype_name, bool requires_grad) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_empty(py::tuple shape, const std::string& dtype_name, bool requires_grad) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_zeros(py::tuple shape, const std::string& dtype_name, bool requires_grad) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_ones(py::tuple shape, const std::string& dtype_name, bool requires_grad) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::tuple tensor_shape(py::object handle) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

std::string tensor_dtype(py::object handle) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::array tensor_to_numpy(py::object handle) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_item(py::object handle) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_add(py::object lhs, py::object rhs) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_add_scalar(py::object tensor, double scalar) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_sub(py::object lhs, py::object rhs) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_sub_scalar(py::object tensor, double scalar) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_rsub_scalar(py::object tensor, double scalar) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_mul(py::object lhs, py::object rhs) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_mul_scalar(py::object tensor, double scalar) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_div(py::object lhs, py::object rhs) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_div_scalar(py::object tensor, double scalar) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_matmul(py::object lhs, py::object rhs) {
    throw std::runtime_error("Tensor operations not yet implemented (Phase 1)");
}

py::object tensor_to_dlpack(py::object handle) {
    throw std::runtime_error("DLPack interop not yet implemented (Phase 3)");
}

py::object tensor_from_dlpack(py::capsule capsule, bool requires_grad) {
    throw std::runtime_error("DLPack interop not yet implemented (Phase 3)");
}

} // namespace python
} // namespace metal_native
