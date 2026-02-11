/// @file hf_accelerate.cpp
/// @brief Implementation of HuggingFace Accelerate integration.

#include "metal_native/interop/hf_accelerate.h"
#include "metal_native/interop/torch_interop.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

namespace metal_native {

// ---------------------------------------------------------------------------
// AccelerateBackend implementation
// ---------------------------------------------------------------------------

std::string AccelerateBackend::device_name() {
    return "metal_native";
}

bool AccelerateBackend::is_available() {
    try {
        // Attempt to initialize the device
        MNDevice& device = MNDevice::instance();
        (void)device;  // Suppress unused variable warning
        return true;
    } catch (const MNException&) {
        return false;
    }
}

TorchTensorMeta AccelerateBackend::move_to_device(const TorchTensorMeta& tensor_meta) {
    // Stub: This would be implemented on the Python side to actually copy
    // tensor data from CPU to GPU. The C++ layer just validates that the
    // device is available and returns the metadata unchanged.
    //
    // In a full implementation:
    // 1. Allocate GPU buffer via MNDevice
    // 2. Copy data from tensor_meta.data_ptr to GPU
    // 3. Update data_ptr to point to GPU memory
    // 4. Return updated metadata

    MN_CHECK(is_available(),
             MetalNativeError::DeviceNotFound,
             "MetalNative device not available for tensor placement");

    // For now, just return the input metadata unchanged
    return tensor_meta;
}

void AccelerateBackend::synchronize() {
    // Block until all GPU operations complete
    auto& device = MNDevice::instance();

#ifdef __OBJC__
    // In Objective-C++ context, we could synchronize the command queue:
    // id<MTLCommandQueue> queue = device.command_queue();
    // id<MTLCommandBuffer> buffer = [queue commandBuffer];
    // [buffer commit];
    // [buffer waitUntilCompleted];
#else
    // In pure C++ context, we can't directly call Metal APIs.
    // This would need to be implemented in a .mm file or delegated to
    // a dispatch system.
    (void)device;  // Suppress unused warning
#endif
}

} // namespace metal_native
