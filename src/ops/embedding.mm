/// @file embedding.mm
/// @brief Objective-C++ implementation of embedding operations.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/ops/embedding.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/kernels/kernel_registry.h"
#include "metal_native/dispatch/command_pipeline.h"

namespace metal_native {

// ---------------------------------------------------------------------------
// Embedding table lookup
// ---------------------------------------------------------------------------

MNTensor embedding(const MNTensor& indices,
                   const MNTensor& weight,
                   MNDevice& device) {
    // Validate inputs
    MN_CHECK(dtype_is_integer(indices.dtype()),
             MetalNativeError::InvalidArgument,
             "embedding: indices must be integer type");

    MN_CHECK(weight.ndim() == 2,
             MetalNativeError::InvalidArgument,
             "embedding: weight must be 2D [vocab_size, embed_dim]");

    MN_CHECK(weight.is_contiguous(),
             MetalNativeError::InvalidArgument,
             "embedding: weight must be contiguous");

    const int64_t vocab_size = weight.shape()[0];
    const int64_t embed_dim = weight.shape()[1];
    const int64_t num_indices = indices.numel();

    // Create output shape: [...indices.shape, embed_dim]
    std::vector<int64_t> output_dims;
    for (size_t i = 0; i < indices.ndim(); ++i) {
        output_dims.push_back(indices.shape()[i]);
    }
    output_dims.push_back(embed_dim);
    MNShape output_shape(output_dims);

    // Allocate output tensor
    MNTensor output = MNTensor::empty(output_shape, weight.dtype(), device);

    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        // Select kernel based on dtype
        const char* kernel_name = nullptr;
        if (weight.dtype() == MNDType::Float32) {
            kernel_name = "embedding_lookup_fp32";
        } else if (weight.dtype() == MNDType::Float16) {
            kernel_name = "embedding_lookup_fp16";
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "embedding: unsupported weight dtype");
        }

        id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);

        // Encode compute command
        id<MTLCommandBuffer> command_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:indices.buffer()->metal_buffer() offset:indices.offset() atIndex:0];
        [encoder setBuffer:weight.buffer()->metal_buffer() offset:weight.offset() atIndex:1];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:2];

        // Pass constants
        uint32_t constants[3] = {
            static_cast<uint32_t>(num_indices),
            static_cast<uint32_t>(vocab_size),
            static_cast<uint32_t>(embed_dim)
        };
        [encoder setBytes:&constants length:sizeof(constants) atIndex:3];

        // Dispatch threads
        MTLSize grid_size = MTLSizeMake(num_indices, 1, 1);
        MTLSize threadgroup_size = MTLSizeMake(
            std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        cmd_pipeline.commit();
    }

    return output;
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding (RoPE)
// ---------------------------------------------------------------------------

MNTensor rope_embedding(const MNTensor& input,
                        const MNTensor& cos_cache,
                        const MNTensor& sin_cache,
                        const MNTensor& position_ids,
                        MNDevice& device) {
    // Validate inputs
    MN_CHECK(input.ndim() == 4,
             MetalNativeError::InvalidArgument,
             "rope_embedding: input must be 4D [batch, seq_len, num_heads, head_dim]");

    MN_CHECK(cos_cache.ndim() == 2 && sin_cache.ndim() == 2,
             MetalNativeError::InvalidArgument,
             "rope_embedding: cos_cache and sin_cache must be 2D [max_seq_len, head_dim]");

    MN_CHECK(position_ids.ndim() == 2,
             MetalNativeError::InvalidArgument,
             "rope_embedding: position_ids must be 2D [batch, seq_len]");

    const int64_t batch = input.shape()[0];
    const int64_t seq_len = input.shape()[1];
    const int64_t num_heads = input.shape()[2];
    const int64_t head_dim = input.shape()[3];

    MN_CHECK(cos_cache.shape()[1] == head_dim && sin_cache.shape()[1] == head_dim,
             MetalNativeError::InvalidArgument,
             "rope_embedding: cache head_dim must match input head_dim");

    MN_CHECK(position_ids.shape()[0] == batch && position_ids.shape()[1] == seq_len,
             MetalNativeError::InvalidArgument,
             "rope_embedding: position_ids shape must match [batch, seq_len]");

    // Allocate output tensor (same shape as input)
    MNTensor output = MNTensor::empty(input.shape(), input.dtype(), device);

    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        // Select kernel based on dtype
        const char* kernel_name = nullptr;
        if (input.dtype() == MNDType::Float32) {
            kernel_name = "rope_embedding_fp32";
        } else if (input.dtype() == MNDType::Float16) {
            kernel_name = "rope_embedding_fp16";
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "rope_embedding: unsupported dtype");
        }

        id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);

        // Encode compute command
        id<MTLCommandBuffer> command_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input.buffer()->metal_buffer() offset:input.offset() atIndex:0];
        [encoder setBuffer:cos_cache.buffer()->metal_buffer() offset:cos_cache.offset() atIndex:1];
        [encoder setBuffer:sin_cache.buffer()->metal_buffer() offset:sin_cache.offset() atIndex:2];
        [encoder setBuffer:position_ids.buffer()->metal_buffer() offset:position_ids.offset() atIndex:3];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:4];

        // Pass constants
        uint32_t constants[4] = {
            static_cast<uint32_t>(batch),
            static_cast<uint32_t>(seq_len),
            static_cast<uint32_t>(num_heads),
            static_cast<uint32_t>(head_dim)
        };
        [encoder setBytes:&constants length:sizeof(constants) atIndex:5];

        // Dispatch threads: one thread per (batch, seq_len, head, dim/2) element
        const int64_t total_threads = batch * seq_len * num_heads * (head_dim / 2);
        MTLSize grid_size = MTLSizeMake(total_threads, 1, 1);
        MTLSize threadgroup_size = MTLSizeMake(
            std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        cmd_pipeline.commit();
    }

    return output;
}

} // namespace metal_native
