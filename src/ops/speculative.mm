/// @file speculative.mm
/// @brief Objective-C++ implementation of speculative decoding verification.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/ops/speculative.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/kernels/kernel_registry.h"
#include "metal_native/dispatch/command_pipeline.h"

namespace metal_native {

SpeculativeResult speculative_verify(
    const MNTensor& p_target,
    const MNTensor& p_draft,
    const MNTensor& draft_tokens,
    const MNTensor& random_vals)
{
    // Validate inputs
    MN_CHECK(p_target.ndim() == 3,
             MetalNativeError::InvalidArgument,
             "speculative_verify: p_target must be 3D [batch, K, vocab_size]");
    MN_CHECK(p_draft.ndim() == 3,
             MetalNativeError::InvalidArgument,
             "speculative_verify: p_draft must be 3D [batch, K, vocab_size]");
    MN_CHECK(draft_tokens.ndim() == 2,
             MetalNativeError::InvalidArgument,
             "speculative_verify: draft_tokens must be 2D [batch, K]");
    MN_CHECK(random_vals.ndim() == 2,
             MetalNativeError::InvalidArgument,
             "speculative_verify: random_vals must be 2D [batch, K]");

    // Extract dimensions
    int64_t batch_size = p_target.shape()[0];
    int64_t K = p_target.shape()[1];
    int64_t vocab_size = p_target.shape()[2];

    // Validate shape consistency
    MN_CHECK(p_draft.shape()[0] == batch_size && p_draft.shape()[1] == K && p_draft.shape()[2] == vocab_size,
             MetalNativeError::InvalidArgument,
             "speculative_verify: p_draft shape mismatch");
    MN_CHECK(draft_tokens.shape()[0] == batch_size && draft_tokens.shape()[1] == K,
             MetalNativeError::InvalidArgument,
             "speculative_verify: draft_tokens shape mismatch");
    MN_CHECK(random_vals.shape()[0] == batch_size && random_vals.shape()[1] == K,
             MetalNativeError::InvalidArgument,
             "speculative_verify: random_vals shape mismatch");

    // Validate dtypes
    MN_CHECK(p_target.dtype() == p_draft.dtype(),
             MetalNativeError::InvalidArgument,
             "speculative_verify: p_target and p_draft must have same dtype");
    MN_CHECK(p_target.dtype() == MNDType::Float32 || p_target.dtype() == MNDType::Float16,
             MetalNativeError::InvalidArgument,
             "speculative_verify: p_target/p_draft must be Float32 or Float16");
    MN_CHECK(draft_tokens.dtype() == MNDType::Int32,
             MetalNativeError::InvalidArgument,
             "speculative_verify: draft_tokens must be Int32");
    MN_CHECK(random_vals.dtype() == MNDType::Float32,
             MetalNativeError::InvalidArgument,
             "speculative_verify: random_vals must be Float32");

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();

        // Allocate output tensors
        MNTensor accepted = MNTensor::empty(
            MNShape({batch_size, K}),
            MNDType::Int32,
            device
        );
        MNTensor num_accepted = MNTensor::empty(
            MNShape({batch_size}),
            MNDType::Int32,
            device
        );

        // Select kernel based on dtype
        const char* kernel_name = nullptr;
        if (p_target.dtype() == MNDType::Float32) {
            kernel_name = "speculative_verify_fp32";
        } else if (p_target.dtype() == MNDType::Float16) {
            kernel_name = "speculative_verify_fp16";
        }

        CommandPipeline& cmd_pipeline = device.command_pipeline();
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();

        // Dispatch kernel
        {
            id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
            id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);
            [encoder setComputePipelineState:pipeline];

            // Set buffers
            [encoder setBuffer:p_target.buffer()->metal_buffer() offset:p_target.offset() atIndex:0];
            [encoder setBuffer:p_draft.buffer()->metal_buffer() offset:p_draft.offset() atIndex:1];
            [encoder setBuffer:draft_tokens.buffer()->metal_buffer() offset:draft_tokens.offset() atIndex:2];
            [encoder setBuffer:random_vals.buffer()->metal_buffer() offset:random_vals.offset() atIndex:3];
            [encoder setBuffer:accepted.buffer()->metal_buffer() offset:accepted.offset() atIndex:4];
            [encoder setBuffer:num_accepted.buffer()->metal_buffer() offset:num_accepted.offset() atIndex:5];

            // Set constants
            uint32_t K_u32 = static_cast<uint32_t>(K);
            uint32_t vocab_size_u32 = static_cast<uint32_t>(vocab_size);
            [encoder setBytes:&K_u32 length:sizeof(uint32_t) atIndex:6];
            [encoder setBytes:&vocab_size_u32 length:sizeof(uint32_t) atIndex:7];

            // Dispatch: one thread per batch
            MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1);
            MTLSize threadgroup_size = MTLSizeMake(1, 1, 1);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
            [encoder endEncoding];
        }

        // Commit and continue
        cmd_pipeline.commit_and_continue();

        return SpeculativeResult{
            .accepted = accepted,
            .num_accepted = num_accepted
        };
    }
}

} // namespace metal_native
