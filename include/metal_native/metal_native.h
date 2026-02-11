#pragma once

/// @file metal_native.h
/// @brief Umbrella header -- includes all public metal_native headers.
///
/// Consumers can simply:
/// @code
///   #include <metal_native/metal_native.h>
/// @endcode

#include "metal_native/core/autorelease_scope.h"
#include "metal_native/core/buffer.h"
#include "metal_native/core/device.h"
#include "metal_native/core/dtype.h"
#include "metal_native/core/error.h"
#include "metal_native/core/shape.h"
#include "metal_native/core/tensor.h"

#include "metal_native/kernels/kernel_cache.h"
#include "metal_native/kernels/kernel_launch.h"
#include "metal_native/kernels/kernel_registry.h"
