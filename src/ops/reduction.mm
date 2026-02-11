/// @file reduction.mm
/// @brief Objective-C++ implementation of reduction operations.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/ops/reduction.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/kernels/kernel_registry.h"
#include "metal_native/dispatch/command_pipeline.h"

namespace metal_native {

namespace {

/// Helper to normalize dimension index.
int64_t normalize_dim(int64_t dim, size_t ndim) {
    if (dim < 0) {
        dim += static_cast<int64_t>(ndim);
    }
    MN_CHECK(dim >= 0 && dim < static_cast<int64_t>(ndim),
             MetalNativeError::InvalidArgument,
             "reduction: dimension out of range");
    return dim;
}

/// Helper to compute output shape after reduction.
MNShape compute_reduction_shape(const MNShape& input_shape,
                                int64_t dim,
                                bool keepdim) {
    std::vector<int64_t> output_dims;
    for (size_t i = 0; i < input_shape.ndim(); ++i) {
        if (static_cast<int64_t>(i) == dim) {
            if (keepdim) {
                output_dims.push_back(1);
            }
            // Otherwise skip this dimension
        } else {
            output_dims.push_back(input_shape[static_cast<int64_t>(i)]);
        }
    }
    return MNShape(output_dims);
}

/// Generic reduction operation dispatcher.
MNTensor dispatch_reduction(const MNTensor& input,
                            int64_t dim,
                            bool keepdim,
                            MNDType output_dtype,
                            const char* kernel_prefix,
                            MNDevice& device) {
    dim = normalize_dim(dim, input.ndim());

    MN_CHECK(input.is_contiguous(),
             MetalNativeError::InvalidArgument,
             "reduction: input must be contiguous");

    // Compute output shape
    MNShape output_shape = compute_reduction_shape(input.shape(), dim, keepdim);
    MNTensor output = MNTensor::empty(output_shape, output_dtype, device);

    // Compute reduction parameters
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input.shape()[i];
    }

    const int64_t reduce_size = input.shape()[dim];

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < input.ndim(); ++i) {
        inner_size *= input.shape()[static_cast<int64_t>(i)];
    }

    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        // Select kernel based on dtype
        std::string kernel_name_str;
        if (input.dtype() == MNDType::Float32) {
            kernel_name_str = std::string(kernel_prefix) + "_fp32";
        } else if (input.dtype() == MNDType::Float16) {
            kernel_name_str = std::string(kernel_prefix) + "_fp16";
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "reduction: unsupported dtype");
        }
        const char* kernel_name = kernel_name_str.c_str();

        id<MTLCommandBuffer> command_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input.buffer()->metal_buffer() offset:input.offset() atIndex:0];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:1];

        // Pass reduction parameters
        uint32_t constants[3] = {
            static_cast<uint32_t>(outer_size),
            static_cast<uint32_t>(reduce_size),
            static_cast<uint32_t>(inner_size)
        };
        [encoder setBytes:&constants length:sizeof(constants) atIndex:2];

        // Dispatch: one threadgroup of 256 threads per output element
        const int64_t num_output = outer_size * inner_size;
        const NSUInteger THREADGROUP_SIZE = 256;
        const NSUInteger SIMD_SIZE = 32;

        MTLSize grid_size = MTLSizeMake(num_output, 1, 1);       // number of threadgroups
        MTLSize threadgroup_size = MTLSizeMake(THREADGROUP_SIZE, 1, 1);

        // Allocate threadgroup memory for SIMD group partial results
        // Each SIMD group (256/32 = 8) stores one partial result
        // For argmax/argmin we need both float + int64_t per SIMD group
        NSUInteger shared_mem_size;
        if (strncmp(kernel_prefix, "arg", 3) == 0) {
            // argmax/argmin: float values + int64_t indices
            shared_mem_size = (THREADGROUP_SIZE / SIMD_SIZE) * (sizeof(float) + sizeof(int64_t));
        } else {
            shared_mem_size = (THREADGROUP_SIZE / SIMD_SIZE) * sizeof(float);
        }

        [encoder setThreadgroupMemoryLength:shared_mem_size atIndex:0];

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        cmd_pipeline.commit();
    }

    return output;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

MNTensor reduce_sum(const MNTensor& input,
                    int64_t dim,
                    bool keepdim,
                    MNDevice& device) {
    // Sum uses FP32 accumulation even for FP16 inputs
    MNDType output_dtype = (input.dtype() == MNDType::Float16) ? MNDType::Float32 : input.dtype();
    return dispatch_reduction(input, dim, keepdim, output_dtype, "reduce_sum", device);
}

MNTensor reduce_mean(const MNTensor& input,
                     int64_t dim,
                     bool keepdim,
                     MNDevice& device) {
    MNDType output_dtype = (input.dtype() == MNDType::Float16) ? MNDType::Float32 : input.dtype();
    return dispatch_reduction(input, dim, keepdim, output_dtype, "reduce_mean", device);
}

MNTensor reduce_max(const MNTensor& input,
                    int64_t dim,
                    bool keepdim,
                    MNDevice& device) {
    // Max preserves input dtype
    return dispatch_reduction(input, dim, keepdim, input.dtype(), "reduce_max", device);
}

MNTensor reduce_min(const MNTensor& input,
                    int64_t dim,
                    bool keepdim,
                    MNDevice& device) {
    // Min preserves input dtype
    return dispatch_reduction(input, dim, keepdim, input.dtype(), "reduce_min", device);
}

MNTensor argmax(const MNTensor& input,
                int64_t dim,
                MNDevice& device) {
    // argmax always returns Int64 indices with keepdim=false
    return dispatch_reduction(input, dim, false, MNDType::Int64, "argmax", device);
}

MNTensor argmin(const MNTensor& input,
                int64_t dim,
                MNDevice& device) {
    // argmin always returns Int64 indices with keepdim=false
    return dispatch_reduction(input, dim, false, MNDType::Int64, "argmin", device);
}

} // namespace metal_native
