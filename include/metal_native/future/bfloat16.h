#pragma once

/// @file bfloat16.h
/// @brief BFloat16 support utilities.
///
/// BFloat16 (Brain Floating Point) provides FP32's dynamic range with FP16's
/// throughput, making it ideal for training. Supported on Apple GPU Family 9+
/// (M3 and later).
///
/// This module provides:
/// - Hardware capability detection
/// - Automatic mixed precision (AMP) with BF16 policies
/// - Conversion utilities

#include <cstdint>
#include "metal_native/core/dtype.h"

namespace metal_native {
namespace future {

/// Check if BFloat16 is supported by the current device.
/// Requires Apple GPU Family 9 (M3+).
bool bfloat16_supported() noexcept;

/// BFloat16 AMP (Automatic Mixed Precision) policy.
enum class BF16Policy : uint8_t {
    Disabled = 0,  ///< No BF16 usage.
    WeightsOnly,   ///< Store weights in BF16, compute in FP32.
    Full,          ///< Full BF16 compute (where safe).
    Selective,     ///< Use BF16 for specific ops only.
};

/// Configure BFloat16 AMP policy.
void set_bf16_policy(BF16Policy policy);

/// Get current BFloat16 AMP policy.
BF16Policy get_bf16_policy() noexcept;

} // namespace future
} // namespace metal_native
