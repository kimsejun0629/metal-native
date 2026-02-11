/// @file future_stubs.mm
/// @brief Stub implementations for future hardware features.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/future/metal4.h"
#include "metal_native/future/bfloat16.h"
#include "metal_native/future/neural_engine.h"
#include "metal_native/future/fast_ops.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

namespace metal_native {
namespace future {

bool metal4_available() noexcept {
    // Metal 4 not yet available in shipping macOS
    return false;
}

Metal4Capabilities query_metal4_capabilities() noexcept {
    return Metal4Capabilities{};
}

static BF16Policy g_bf16_policy = BF16Policy::Disabled;

bool bfloat16_supported() noexcept {
    return MNDevice::instance().supports_bfloat16();
}

void set_bf16_policy(BF16Policy policy) {
    g_bf16_policy = policy;
}

BF16Policy get_bf16_policy() noexcept {
    return g_bf16_policy;
}

bool neural_engine_available() noexcept {
    // Neural Engine via Metal TensorOps not yet exposed
    return false;
}

NeuralEngineCapabilities query_neural_engine() noexcept {
    return NeuralEngineCapabilities{};
}

} // namespace future

namespace fast {

MNTensor rms_norm(const MNTensor& input, const MNTensor& weight, float eps) {
    MN_THROW(MetalNativeError::NotImplemented,
             "fast::rms_norm not yet implemented (Phase 4)");
}

MNTensor layer_norm(const MNTensor& input, const MNTensor& weight,
                    const MNTensor& bias, float eps) {
    MN_THROW(MetalNativeError::NotImplemented,
             "fast::layer_norm not yet implemented (Phase 4)");
}

void rope(MNTensor& q, MNTensor& k,
          const MNTensor& cos_cache, const MNTensor& sin_cache,
          int64_t start_pos) {
    MN_THROW(MetalNativeError::NotImplemented,
             "fast::rope not yet implemented (Phase 4)");
}

MNTensor scaled_dot_product_attention(const MNTensor& query,
                                       const MNTensor& key,
                                       const MNTensor& value,
                                       float scale,
                                       bool causal) {
    MN_THROW(MetalNativeError::NotImplemented,
             "fast::scaled_dot_product_attention not yet implemented (Phase 4)");
}

MNTensor swiglu(const MNTensor& gate, const MNTensor& up) {
    MN_THROW(MetalNativeError::NotImplemented,
             "fast::swiglu not yet implemented (Phase 4)");
}

} // namespace fast
} // namespace metal_native
