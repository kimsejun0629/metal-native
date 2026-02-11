#pragma once

/// @file metal4.h
/// @brief Future Metal 4 API support stubs.
///
/// Metal 4 (expected with macOS 27 / iOS 21) introduces:
/// - MTLTensor: native tensor type with built-in shape/dtype
/// - MTL4ArgumentTable: efficient resource binding
/// - MTL4MachineLearningCommandEncoder: ML-specific encoder
/// - Residency sets for buffer/kernel preloading
/// - MTL4Compiler for ahead-of-time shader compilation
///
/// This header provides forward-compatible API stubs.

#include <cstdint>
#include <string>

namespace metal_native {
namespace future {

/// Check if the current device supports Metal 4 features.
/// @return true if Metal 4 is available (macOS 27+).
bool metal4_available() noexcept;

/// Metal 4 feature flags.
struct Metal4Capabilities {
    bool has_native_tensors = false;
    bool has_ml_encoder = false;
    bool has_argument_tables = false;
    bool has_residency_sets = false;
    bool has_shader_precompilation = false;
};

/// Query Metal 4 capabilities of the current device.
Metal4Capabilities query_metal4_capabilities() noexcept;

} // namespace future
} // namespace metal_native
