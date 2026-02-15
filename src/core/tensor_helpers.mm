/// @file tensor_helpers.mm
/// @brief Objective-C++ helpers for MNTensor operations requiring Metal API.

#import <Metal/Metal.h>

#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/buffer.h"
#include "metal_native/core/error.h"

namespace metal_native {

MNTensor MNTensor::to_shared() const {
    // If already Shared, return a copy (no GPU operation needed)
    if (buffer_->storage_mode() == StorageMode::Shared) {
        return *this;
    }

    // Private storage: we need to blit to a Shared buffer
    auto& device = MNDevice::instance();

    // Ensure all pending GPU work (including the kernel that wrote this tensor) is done
    device.synchronize();

    size_t byte_size = nbytes();

    // Allocate a Shared buffer (not pooled - we need a clean buffer)
    auto shared_buf = std::make_shared<MNBuffer>(device, byte_size, StorageMode::Shared);

    // Use a standalone command buffer for the blit (avoids CommandPipeline encoder conflicts)
    id<MTLCommandBuffer> cmd = [device.command_queue() commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];

    [blit copyFromBuffer:buffer_->metal_buffer()
            sourceOffset:offset_
                toBuffer:shared_buf->metal_buffer()
       destinationOffset:0
                    size:byte_size];

    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // Create contiguous strides for the new buffer (offset is 0)
    auto new_strides = shape_.contiguous_strides();

    return MNTensor(shared_buf, shape_, std::move(new_strides), dtype_, 0);
}

} // namespace metal_native
