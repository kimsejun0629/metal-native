/// @file softmax.mm
/// @brief Objective-C++ implementation of softmax operations.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/ops/softmax.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/core/buffer.h"
#include "metal_native/kernels/kernel_registry.h"
#include "metal_native/dispatch/command_pipeline.h"

namespace metal_native {

namespace {

// Normalize dimension index to positive value
int64_t normalize_dim(int64_t dim, size_t ndim) {
    if (dim < 0) {
        dim += static_cast<int64_t>(ndim);
    }
    MN_CHECK(dim >= 0 && dim < static_cast<int64_t>(ndim),
             MetalNativeError::InvalidArgument,
             "softmax: dimension out of range");
    return dim;
}

// Compute size of dimensions before, at, and after the reduction dimension
void compute_reduction_sizes(const MNShape& shape, int64_t dim,
                            uint32_t& outer_size, uint32_t& reduce_size, uint32_t& inner_size) {
    outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= static_cast<uint32_t>(shape[i]);
    }

    reduce_size = static_cast<uint32_t>(shape[dim]);

    inner_size = 1;
    for (size_t i = dim + 1; i < shape.ndim(); ++i) {
        inner_size *= static_cast<uint32_t>(shape[static_cast<int64_t>(i)]);
    }
}

} // anonymous namespace

MNTensor softmax(const MNTensor& input, int64_t dim) {
    dim = normalize_dim(dim, input.ndim());

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();

        // Allocate output buffer
        MNTensor output = MNTensor::empty(input.shape(), input.dtype(), device);

        uint32_t outer_size, reduce_size, inner_size;
        compute_reduction_sizes(input.shape(), dim, outer_size, reduce_size, inner_size);

        // Select kernel based on dtype
        const char* kernel_name = nullptr;
        if (input.dtype() == MNDType::Float32) {
            kernel_name = "softmax_online_fp32";
        } else if (input.dtype() == MNDType::Float16) {
            kernel_name = "softmax_online_fp16";
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "softmax: unsupported dtype (only Float32 and Float16)");
        }

        CommandPipeline& cmd_pipeline = device.command_pipeline();
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();

        // Single-pass online softmax
        {
            id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
            id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);
            [encoder setComputePipelineState:pipeline];

            [encoder setBuffer:input.buffer()->metal_buffer() offset:input.offset() atIndex:0];
            [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:1];
            [encoder setBytes:&outer_size length:sizeof(uint32_t) atIndex:2];
            [encoder setBytes:&reduce_size length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&inner_size length:sizeof(uint32_t) atIndex:4];

            MTLSize grid_size = MTLSizeMake(inner_size, outer_size, 1);
            MTLSize threadgroup_size = MTLSizeMake(32, 1, 1);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
            [encoder endEncoding];
        }

        cmd_pipeline.commit();

        return output;
    }
}

MNTensor log_softmax(const MNTensor& input, int64_t dim) {
    dim = normalize_dim(dim, input.ndim());

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();

        // Allocate output buffer
        MNTensor output = MNTensor::empty(input.shape(), input.dtype(), device);

        uint32_t outer_size, reduce_size, inner_size;
        compute_reduction_sizes(input.shape(), dim, outer_size, reduce_size, inner_size);

        // Select kernel based on dtype
        const char* kernel_name = nullptr;
        if (input.dtype() == MNDType::Float32) {
            kernel_name = "log_softmax_online_fp32";
        } else if (input.dtype() == MNDType::Float16) {
            kernel_name = "log_softmax_online_fp16";
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "log_softmax: unsupported dtype (only Float32 and Float16)");
        }

        CommandPipeline& cmd_pipeline = device.command_pipeline();
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();

        // Single-pass online log_softmax
        {
            id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
            id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);
            [encoder setComputePipelineState:pipeline];

            [encoder setBuffer:input.buffer()->metal_buffer() offset:input.offset() atIndex:0];
            [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:1];
            [encoder setBytes:&outer_size length:sizeof(uint32_t) atIndex:2];
            [encoder setBytes:&reduce_size length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&inner_size length:sizeof(uint32_t) atIndex:4];

            MTLSize grid_size = MTLSizeMake(inner_size, outer_size, 1);
            MTLSize threadgroup_size = MTLSizeMake(32, 1, 1);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
            [encoder endEncoding];
        }

        cmd_pipeline.commit();

        return output;
    }
}

} // namespace metal_native
