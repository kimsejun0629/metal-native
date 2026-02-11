/// @file dtype.cpp
/// @brief Implementation of data-type query functions.

#include "metal_native/core/dtype.h"

namespace metal_native {

// ---------------------------------------------------------------------------
// dtype_size
// ---------------------------------------------------------------------------

size_t dtype_size(MNDType dtype) noexcept {
    switch (dtype) {
        case MNDType::Float32:  return 4;
        case MNDType::Float16:  return 2;
        case MNDType::BFloat16: return 2;
        case MNDType::Int64:    return 8;
        case MNDType::Int32:    return 4;
        case MNDType::Int16:    return 2;
        case MNDType::Int8:     return 1;
        case MNDType::UInt8:    return 1;
        case MNDType::Bool:     return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// dtype_name
// ---------------------------------------------------------------------------

const char* dtype_name(MNDType dtype) noexcept {
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
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// dtype_is_floating_point
// ---------------------------------------------------------------------------

bool dtype_is_floating_point(MNDType dtype) noexcept {
    switch (dtype) {
        case MNDType::Float32:
        case MNDType::Float16:
        case MNDType::BFloat16:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// dtype_is_integer
// ---------------------------------------------------------------------------

bool dtype_is_integer(MNDType dtype) noexcept {
    switch (dtype) {
        case MNDType::Int64:
        case MNDType::Int32:
        case MNDType::Int16:
        case MNDType::Int8:
        case MNDType::UInt8:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// dtype_is_signed
// ---------------------------------------------------------------------------

bool dtype_is_signed(MNDType dtype) noexcept {
    switch (dtype) {
        case MNDType::Float32:
        case MNDType::Float16:
        case MNDType::BFloat16:
        case MNDType::Int64:
        case MNDType::Int32:
        case MNDType::Int16:
        case MNDType::Int8:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// dtype_to_mtl_vertex_format
// ---------------------------------------------------------------------------

// MTLVertexFormat values (from <Metal/MTLVertexDescriptor.h>) so that pure-C++
// translation units do not need to include Metal headers.
//
// We only map types that have a direct 1-component vertex-format counterpart.
// Types without a mapping (BFloat16, Int64, Bool) return 0 (Invalid).

namespace {
    constexpr uint32_t kMTLVertexFormatInvalid  = 0;
    constexpr uint32_t kMTLVertexFormatFloat     = 28;  // MTLVertexFormatFloat
    constexpr uint32_t kMTLVertexFormatHalf      = 25;  // MTLVertexFormatHalf
    constexpr uint32_t kMTLVertexFormatInt       = 32;  // MTLVertexFormatInt
    constexpr uint32_t kMTLVertexFormatShort     = 29;  // MTLVertexFormatShort
    constexpr uint32_t kMTLVertexFormatChar      = 45;  // MTLVertexFormatChar
    constexpr uint32_t kMTLVertexFormatUChar     = 49;  // MTLVertexFormatUChar
} // anonymous namespace

uint32_t dtype_to_mtl_vertex_format(MNDType dtype) noexcept {
    switch (dtype) {
        case MNDType::Float32:  return kMTLVertexFormatFloat;
        case MNDType::Float16:  return kMTLVertexFormatHalf;
        case MNDType::Int32:    return kMTLVertexFormatInt;
        case MNDType::Int16:    return kMTLVertexFormatShort;
        case MNDType::Int8:     return kMTLVertexFormatChar;
        case MNDType::UInt8:    return kMTLVertexFormatUChar;
        // No direct mapping for these types.
        case MNDType::BFloat16:
        case MNDType::Int64:
        case MNDType::Bool:
            return kMTLVertexFormatInvalid;
    }
    return kMTLVertexFormatInvalid;
}

} // namespace metal_native
