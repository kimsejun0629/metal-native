/// @file normalization.mm
/// @brief Objective-C++ implementation of normalization operations.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/ops/normalization.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/kernels/kernel_registry.h"
#include "metal_native/dispatch/command_pipeline.h"

namespace metal_native {

MNTensor layer_norm(const MNTensor& input,
                    const MNTensor& weight,
                    const MNTensor& bias,
                    float eps) {
    MN_CHECK(input.ndim() >= 1,
             MetalNativeError::InvalidArgument,
             "layer_norm: input must have at least 1 dimension");

    // Calculate batch_size and norm_size
    // norm_size is the last dimension, batch_size is product of all others
    const int64_t norm_size = input.shape()[input.ndim() - 1];
    int64_t batch_size = 1;
    for (size_t i = 0; i < input.ndim() - 1; ++i) {
        batch_size *= input.shape()[static_cast<int64_t>(i)];
    }

    MN_CHECK(weight.ndim() == 1 && weight.shape()[0] == norm_size,
             MetalNativeError::InvalidArgument,
             "layer_norm: weight shape must match normalized dimension");
    MN_CHECK(bias.ndim() == 1 && bias.shape()[0] == norm_size,
             MetalNativeError::InvalidArgument,
             "layer_norm: bias shape must match normalized dimension");
    MN_CHECK(input.dtype() == weight.dtype() && input.dtype() == bias.dtype(),
             MetalNativeError::InvalidArgument,
             "layer_norm: all inputs must have the same dtype");

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();
        StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
        MNTensor output = MNTensor::empty(input.shape(), input.dtype(), device, out_mode);

        // Kernel selection strategy:
        // - norm_size >= 2048: "large" kernels with 256 threads (8 SIMD groups) per row.
        //   Achieves ~20-40% GPU occupancy vs ~0.3% for the old 32-thread design.
        // - 128 <= norm_size < 2048: vectorized kernels (32 threads, float4/half4 loads)
        // - norm_size < 128: basic scalar kernels (32 threads)
        const bool use_large = (norm_size >= 2048);
        const bool use_vectorized = (!use_large && norm_size >= 128 && norm_size <= 8192);
        const bool use_regcache = use_large && (norm_size / 256 <= 32);  // max 8192

        const char* kernel_name = nullptr;
        if (input.dtype() == MNDType::Float32) {
            if (use_regcache) kernel_name = "layer_norm_large_regcache_fp32";
            else if (use_large) kernel_name = "layer_norm_large_fp32";
            else if (use_vectorized) kernel_name = "layer_norm_vec_fp32";
            else kernel_name = "layer_norm_kernel";
        } else if (input.dtype() == MNDType::Float16) {
            if (use_regcache) kernel_name = "layer_norm_large_regcache_fp16";
            else if (use_large) kernel_name = "layer_norm_large_fp16";
            else if (use_vectorized) kernel_name = "layer_norm_vec_fp16";
            else kernel_name = "layer_norm_kernel_fp16";
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "layer_norm: unsupported dtype (only Float32 and Float16)");
        }

        id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);
        CommandPipeline& cmd_pipeline = device.command_pipeline();
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input.buffer()->metal_buffer() offset:input.offset() atIndex:0];
        [encoder setBuffer:weight.buffer()->metal_buffer() offset:weight.offset() atIndex:1];
        [encoder setBuffer:bias.buffer()->metal_buffer() offset:bias.offset() atIndex:2];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:3];

        MN_CHECK(batch_size <= UINT32_MAX, MetalNativeError::InvalidArgument, "normalization: dimension exceeds uint32_t range");
        MN_CHECK(norm_size <= UINT32_MAX, MetalNativeError::InvalidArgument, "normalization: dimension exceeds uint32_t range");
        uint32_t batch_size_u32 = static_cast<uint32_t>(batch_size);
        uint32_t norm_size_u32 = static_cast<uint32_t>(norm_size);

        [encoder setBytes:&batch_size_u32 length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&norm_size_u32 length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&eps length:sizeof(float) atIndex:6];

        if (use_large) {
            // Large kernels: 256 threads per threadgroup, one threadgroup per batch row.
            // Threadgroup memory: 16 floats (8 for sum + 8 for sq_sum) for inter-SIMD reduction.
            [encoder setThreadgroupMemoryLength:16 * sizeof(float) atIndex:0];

            MTLSize grid_size = MTLSizeMake(batch_size, 1, 1);
            MTLSize threadgroup_size = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];
        } else {
            // Small/medium kernels: 32 threads (1 SIMD group) per batch row.
            if (use_vectorized) {
                [encoder setThreadgroupMemoryLength:norm_size * sizeof(float) atIndex:0];
            }
            MTLSize grid_size = MTLSizeMake(32, batch_size, 1);
            MTLSize threadgroup_size = MTLSizeMake(32, 1, 1);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        }

        [encoder endEncoding];
        // OPT-5: Use commit_and_continue to allow command buffer reuse.
        cmd_pipeline.commit_and_continue();

        return output;
    }
}

MNTensor rms_norm(const MNTensor& input,
                  const MNTensor& weight,
                  float eps) {
    MN_CHECK(input.ndim() >= 1,
             MetalNativeError::InvalidArgument,
             "rms_norm: input must have at least 1 dimension");

    const int64_t norm_size = input.shape()[input.ndim() - 1];
    int64_t batch_size = 1;
    for (size_t i = 0; i < input.ndim() - 1; ++i) {
        batch_size *= input.shape()[static_cast<int64_t>(i)];
    }

    MN_CHECK(weight.ndim() == 1 && weight.shape()[0] == norm_size,
             MetalNativeError::InvalidArgument,
             "rms_norm: weight shape must match normalized dimension");
    MN_CHECK(input.dtype() == weight.dtype(),
             MetalNativeError::InvalidArgument,
             "rms_norm: input and weight must have the same dtype");

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();
        StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
        MNTensor output = MNTensor::empty(input.shape(), input.dtype(), device, out_mode);

        // Kernel selection: same strategy as layer_norm
        const bool use_large = (norm_size >= 2048);
        const bool use_vectorized = (!use_large && norm_size >= 128 && norm_size <= 8192);
        const bool use_regcache = use_large && (norm_size / 256 <= 32);  // max 8192

        const char* kernel_name = nullptr;
        if (input.dtype() == MNDType::Float32) {
            if (use_regcache) kernel_name = "rms_norm_large_regcache_fp32";
            else if (use_large) kernel_name = "rms_norm_large_fp32";
            else if (use_vectorized) kernel_name = "rms_norm_vec_fp32";
            else kernel_name = "rms_norm_kernel";
        } else if (input.dtype() == MNDType::Float16) {
            if (use_regcache) kernel_name = "rms_norm_large_regcache_fp16";
            else if (use_large) kernel_name = "rms_norm_large_fp16";
            else if (use_vectorized) kernel_name = "rms_norm_vec_fp16";
            else kernel_name = "rms_norm_kernel_fp16";
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "rms_norm: unsupported dtype (only Float32 and Float16)");
        }

        id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);
        CommandPipeline& cmd_pipeline = device.command_pipeline();
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input.buffer()->metal_buffer() offset:input.offset() atIndex:0];
        [encoder setBuffer:weight.buffer()->metal_buffer() offset:weight.offset() atIndex:1];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:2];

        MN_CHECK(batch_size <= UINT32_MAX, MetalNativeError::InvalidArgument, "normalization: dimension exceeds uint32_t range");
        MN_CHECK(norm_size <= UINT32_MAX, MetalNativeError::InvalidArgument, "normalization: dimension exceeds uint32_t range");
        uint32_t batch_size_u32 = static_cast<uint32_t>(batch_size);
        uint32_t norm_size_u32 = static_cast<uint32_t>(norm_size);

        [encoder setBytes:&batch_size_u32 length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&norm_size_u32 length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&eps length:sizeof(float) atIndex:5];

        if (use_large) {
            // Large kernels: 256 threads, one threadgroup per row.
            // Threadgroup memory: 8 floats for inter-SIMD reduction (RMS only needs sq_sum).
            [encoder setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];

            MTLSize grid_size = MTLSizeMake(batch_size, 1, 1);
            MTLSize threadgroup_size = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];
        } else {
            if (use_vectorized) {
                [encoder setThreadgroupMemoryLength:norm_size * sizeof(float) atIndex:0];
            }
            MTLSize grid_size = MTLSizeMake(32, batch_size, 1);
            MTLSize threadgroup_size = MTLSizeMake(32, 1, 1);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        }

        [encoder endEncoding];
        // OPT-5: Use commit_and_continue to allow command buffer reuse.
        cmd_pipeline.commit_and_continue();

        return output;
    }
}

MNTensor batch_norm(const MNTensor& input,
                    MNTensor& running_mean,
                    MNTensor& running_var,
                    const MNTensor& weight,
                    const MNTensor& bias,
                    bool training,
                    float momentum,
                    float eps) {
    MN_CHECK(input.ndim() >= 2,
             MetalNativeError::InvalidArgument,
             "batch_norm: input must have at least 2 dimensions [batch, channels, ...]");

    const int64_t batch_size = input.shape()[0];
    const int64_t channels = input.shape()[1];

    // Calculate spatial size (product of all dimensions after channels)
    int64_t spatial_size = 1;
    for (size_t i = 2; i < input.ndim(); ++i) {
        spatial_size *= input.shape()[static_cast<int64_t>(i)];
    }

    MN_CHECK(running_mean.ndim() == 1 && running_mean.shape()[0] == channels,
             MetalNativeError::InvalidArgument,
             "batch_norm: running_mean shape must be [channels]");
    MN_CHECK(running_var.ndim() == 1 && running_var.shape()[0] == channels,
             MetalNativeError::InvalidArgument,
             "batch_norm: running_var shape must be [channels]");
    MN_CHECK(weight.ndim() == 1 && weight.shape()[0] == channels,
             MetalNativeError::InvalidArgument,
             "batch_norm: weight shape must be [channels]");
    MN_CHECK(bias.ndim() == 1 && bias.shape()[0] == channels,
             MetalNativeError::InvalidArgument,
             "batch_norm: bias shape must be [channels]");

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();
        StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
        MNTensor output = MNTensor::empty(input.shape(), input.dtype(), device, out_mode);

        CommandPipeline& cmd_pipeline = device.command_pipeline();
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();

        if (training) {
            id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline("batch_norm_training_kernel");
            id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input.buffer()->metal_buffer() offset:input.offset() atIndex:0];
            [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:1];
            [encoder setBuffer:running_mean.buffer()->metal_buffer() offset:running_mean.offset() atIndex:2];
            [encoder setBuffer:running_var.buffer()->metal_buffer() offset:running_var.offset() atIndex:3];
            [encoder setBuffer:weight.buffer()->metal_buffer() offset:weight.offset() atIndex:4];
            [encoder setBuffer:bias.buffer()->metal_buffer() offset:bias.offset() atIndex:5];

            MN_CHECK(batch_size <= UINT32_MAX, MetalNativeError::InvalidArgument, "normalization: dimension exceeds uint32_t range");
            MN_CHECK(channels <= UINT32_MAX, MetalNativeError::InvalidArgument, "normalization: dimension exceeds uint32_t range");
            MN_CHECK(spatial_size <= UINT32_MAX, MetalNativeError::InvalidArgument, "normalization: dimension exceeds uint32_t range");
            uint32_t batch_size_u32 = static_cast<uint32_t>(batch_size);
            uint32_t channels_u32 = static_cast<uint32_t>(channels);
            uint32_t spatial_size_u32 = static_cast<uint32_t>(spatial_size);

            [encoder setBytes:&batch_size_u32 length:sizeof(uint32_t) atIndex:6];
            [encoder setBytes:&channels_u32 length:sizeof(uint32_t) atIndex:7];
            [encoder setBytes:&spatial_size_u32 length:sizeof(uint32_t) atIndex:8];
            [encoder setBytes:&momentum length:sizeof(float) atIndex:9];
            [encoder setBytes:&eps length:sizeof(float) atIndex:10];

            MTLSize grid_size = MTLSizeMake(32, channels, 1);
            MTLSize threadgroup_size = MTLSizeMake(32, 1, 1);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];

            [encoder endEncoding];
        } else {
            id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline("batch_norm_inference_kernel");
            id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input.buffer()->metal_buffer() offset:input.offset() atIndex:0];
            [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:1];
            [encoder setBuffer:running_mean.buffer()->metal_buffer() offset:running_mean.offset() atIndex:2];
            [encoder setBuffer:running_var.buffer()->metal_buffer() offset:running_var.offset() atIndex:3];
            [encoder setBuffer:weight.buffer()->metal_buffer() offset:weight.offset() atIndex:4];
            [encoder setBuffer:bias.buffer()->metal_buffer() offset:bias.offset() atIndex:5];

            MN_CHECK(batch_size <= UINT32_MAX, MetalNativeError::InvalidArgument, "normalization: dimension exceeds uint32_t range");
            MN_CHECK(channels <= UINT32_MAX, MetalNativeError::InvalidArgument, "normalization: dimension exceeds uint32_t range");
            MN_CHECK(spatial_size <= UINT32_MAX, MetalNativeError::InvalidArgument, "normalization: dimension exceeds uint32_t range");
            uint32_t batch_size_u32 = static_cast<uint32_t>(batch_size);
            uint32_t channels_u32 = static_cast<uint32_t>(channels);
            uint32_t spatial_size_u32 = static_cast<uint32_t>(spatial_size);

            [encoder setBytes:&batch_size_u32 length:sizeof(uint32_t) atIndex:6];
            [encoder setBytes:&channels_u32 length:sizeof(uint32_t) atIndex:7];
            [encoder setBytes:&spatial_size_u32 length:sizeof(uint32_t) atIndex:8];
            [encoder setBytes:&eps length:sizeof(float) atIndex:9];

            MTLSize grid_size = MTLSizeMake(spatial_size, channels, batch_size);
            MTLSize threadgroup_size = MTLSizeMake(32, 1, 1);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];

            [encoder endEncoding];
        }

        // OPT-5: Use commit_and_continue for pipeline consistency with other normalization ops.
        // Previously used commit() which forces new buffer allocation on next operation.
        cmd_pipeline.commit_and_continue();

        return output;
    }
}

MNTensor group_norm(const MNTensor& input,
                    size_t num_groups,
                    const MNTensor& weight,
                    const MNTensor& bias,
                    float eps) {
    MN_CHECK(input.ndim() >= 2,
             MetalNativeError::InvalidArgument,
             "group_norm: input must have at least 2 dimensions [batch, channels, ...]");

    const int64_t batch_size = input.shape()[0];
    const int64_t channels = input.shape()[1];

    MN_CHECK(channels % num_groups == 0,
             MetalNativeError::InvalidArgument,
             "group_norm: channels must be divisible by num_groups");

    int64_t spatial_size = 1;
    for (size_t i = 2; i < input.ndim(); ++i) {
        spatial_size *= input.shape()[static_cast<int64_t>(i)];
    }

    MN_CHECK(weight.ndim() == 1 && weight.shape()[0] == channels,
             MetalNativeError::InvalidArgument,
             "group_norm: weight shape must be [channels]");
    MN_CHECK(bias.ndim() == 1 && bias.shape()[0] == channels,
             MetalNativeError::InvalidArgument,
             "group_norm: bias shape must be [channels]");

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();
        StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
        MNTensor output = MNTensor::empty(input.shape(), input.dtype(), device, out_mode);

        const char* kernel_name = nullptr;
        if (input.dtype() == MNDType::Float32) {
            kernel_name = "group_norm_kernel";
        } else if (input.dtype() == MNDType::Float16) {
            kernel_name = "group_norm_kernel_fp16";
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "group_norm: unsupported dtype (only Float32 and Float16)");
        }

        id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);
        CommandPipeline& cmd_pipeline = device.command_pipeline();
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input.buffer()->metal_buffer() offset:input.offset() atIndex:0];
        [encoder setBuffer:weight.buffer()->metal_buffer() offset:weight.offset() atIndex:1];
        [encoder setBuffer:bias.buffer()->metal_buffer() offset:bias.offset() atIndex:2];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:3];

        MN_CHECK(batch_size <= UINT32_MAX, MetalNativeError::InvalidArgument, "normalization: dimension exceeds uint32_t range");
        MN_CHECK(channels <= UINT32_MAX, MetalNativeError::InvalidArgument, "normalization: dimension exceeds uint32_t range");
        MN_CHECK(spatial_size <= UINT32_MAX, MetalNativeError::InvalidArgument, "normalization: dimension exceeds uint32_t range");
        MN_CHECK(num_groups <= UINT32_MAX, MetalNativeError::InvalidArgument, "normalization: dimension exceeds uint32_t range");
        uint32_t batch_size_u32 = static_cast<uint32_t>(batch_size);
        uint32_t channels_u32 = static_cast<uint32_t>(channels);
        uint32_t spatial_size_u32 = static_cast<uint32_t>(spatial_size);
        uint32_t num_groups_u32 = static_cast<uint32_t>(num_groups);

        [encoder setBytes:&batch_size_u32 length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&channels_u32 length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&spatial_size_u32 length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&num_groups_u32 length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&eps length:sizeof(float) atIndex:8];

        MTLSize grid_size = MTLSizeMake(32, num_groups, batch_size);
        MTLSize threadgroup_size = MTLSizeMake(32, 1, 1);
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];

        [encoder endEncoding];
        // OPT-5: Use commit_and_continue to allow command buffer reuse.
        cmd_pipeline.commit_and_continue();

        return output;
    }
}

} // namespace metal_native
