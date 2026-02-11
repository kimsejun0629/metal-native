#pragma once

/// @file dtype.h
/// @brief Data-type descriptors for tensors and buffers.
///
/// Each MNDType enumerator maps one-to-one to a C++ scalar type and
/// (where applicable) to a Metal pixel/vertex format.

#include <cstddef>
#include <cstdint>

namespace metal_native {

/// Supported scalar data types.
enum class MNDType : uint8_t {
    Float32 = 0,
    Float16,
    BFloat16,
    Int64,
    Int32,
    Int16,
    Int8,
    UInt8,
    Bool,
};

/// Number of bytes occupied by a single element of the given type.
size_t dtype_size(MNDType dtype) noexcept;

/// Human-readable name (e.g. "float32", "bfloat16").
const char* dtype_name(MNDType dtype) noexcept;

/// True for Float32, Float16, BFloat16.
bool dtype_is_floating_point(MNDType dtype) noexcept;

/// True for Int64, Int32, Int16, Int8, UInt8.
bool dtype_is_integer(MNDType dtype) noexcept;

/// True for signed types: Float32, Float16, BFloat16, Int64, Int32, Int16, Int8.
bool dtype_is_signed(MNDType dtype) noexcept;

// ---------------------------------------------------------------------------
// Metal format helpers
// ---------------------------------------------------------------------------

/// Return the MTLVertexFormat integer value corresponding to the dtype, or 0
/// (MTLVertexFormatInvalid) if there is no direct mapping.
/// This avoids pulling in Metal headers from pure-C++ translation units.
uint32_t dtype_to_mtl_vertex_format(MNDType dtype) noexcept;

} // namespace metal_native
