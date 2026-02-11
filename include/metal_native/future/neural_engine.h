#pragma once

/// @file neural_engine.h
/// @brief Apple Neural Engine integration stubs.
///
/// Future M5 chips are expected to expose Neural Accelerator access via
/// Metal 4 TensorOps. This provides stubs for:
/// - Dedicated matmul on Neural Engine
/// - Attention acceleration
/// - Segmented matmul for mixture-of-experts

#include <cstdint>
#include "metal_native/core/dtype.h"
#include "metal_native/core/shape.h"

namespace metal_native {

class MNTensor;

namespace future {

/// Check if Neural Engine acceleration is available.
bool neural_engine_available() noexcept;

/// Neural Engine operation support flags.
struct NeuralEngineCapabilities {
    bool supports_matmul = false;
    bool supports_attention = false;
    bool supports_segmented_matmul = false;
    bool supports_conv2d = false;
    uint32_t max_batch_size = 0;
    uint32_t max_sequence_length = 0;
};

/// Query Neural Engine capabilities.
NeuralEngineCapabilities query_neural_engine() noexcept;

} // namespace future
} // namespace metal_native
