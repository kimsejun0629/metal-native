/// @file elementwise.mm
/// @brief Objective-C++ implementation of element-wise operations.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/ops/elementwise.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/kernels/kernel_registry.h"
#include "metal_native/dispatch/command_pipeline.h"

namespace metal_native {

namespace {

/// Helper to dispatch a binary element-wise operation with broadcasting support.
MNTensor dispatch_binary_op(const MNTensor& a,
                             const MNTensor& b,
                             const char* kernel_prefix,
                             MNDevice& device) {
    MN_CHECK(a.dtype() == b.dtype(),
             MetalNativeError::InvalidArgument,
             "elementwise: operands must have same dtype");

    MN_CHECK(a.is_contiguous() && b.is_contiguous(),
             MetalNativeError::InvalidArgument,
             "elementwise: operands must be contiguous");

    // Compute broadcast shape
    MNShape output_shape = a.shape().broadcast_with(b.shape());
    StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
    MNTensor output = MNTensor::empty(output_shape, a.dtype(), device, out_mode);

    const int64_t numel = output.numel();

    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        // Select kernel based on dtype
        std::string kernel_name;
        if (a.dtype() == MNDType::Float32) {
            kernel_name = std::string(kernel_prefix) + "_fp32";
        } else if (a.dtype() == MNDType::Float16) {
            kernel_name = std::string(kernel_prefix) + "_fp16";
        } else if (a.dtype() == MNDType::BFloat16) {
            kernel_name = std::string(kernel_prefix) + "_bf16";
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "elementwise: unsupported dtype");
        }

        id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name.c_str());

        id<MTLCommandBuffer> command_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a.buffer()->metal_buffer() offset:a.offset() atIndex:0];
        [encoder setBuffer:b.buffer()->metal_buffer() offset:b.offset() atIndex:1];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:2];

        // Pass num_elements to kernel (required at buffer(3))
        MN_CHECK(numel <= UINT32_MAX, MetalNativeError::InvalidArgument, "elementwise: dimension exceeds uint32_t range");
        uint32_t num_elements = static_cast<uint32_t>(numel);
        [encoder setBytes:&num_elements length:sizeof(uint32_t) atIndex:3];

        // Dispatch threads: kernel processes 4 elements per thread
        MTLSize grid_size = MTLSizeMake((numel + 3) / 4, 1, 1);
        MTLSize threadgroup_size = MTLSizeMake(
            std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        // OPT-5: Use commit_and_continue to allow command buffer reuse.
        cmd_pipeline.commit_and_continue();
    }

    return output;
}

/// Helper to dispatch a unary element-wise operation.
MNTensor dispatch_unary_op(const MNTensor& x,
                           const char* kernel_prefix,
                           MNDevice& device) {
    MN_CHECK(x.is_contiguous(),
             MetalNativeError::InvalidArgument,
             "elementwise: input must be contiguous");

    StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
    MNTensor output = MNTensor::empty(x.shape(), x.dtype(), device, out_mode);

    const int64_t numel = x.numel();

    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        std::string kernel_name;
        if (x.dtype() == MNDType::Float32) {
            kernel_name = std::string(kernel_prefix) + "_fp32";
        } else if (x.dtype() == MNDType::Float16) {
            kernel_name = std::string(kernel_prefix) + "_fp16";
        } else if (x.dtype() == MNDType::BFloat16) {
            kernel_name = std::string(kernel_prefix) + "_bf16";
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "elementwise: unsupported dtype");
        }

        id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name.c_str());

        id<MTLCommandBuffer> command_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:x.buffer()->metal_buffer() offset:x.offset() atIndex:0];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:1];

        // Pass num_elements to kernel (required at buffer(2))
        MN_CHECK(numel <= UINT32_MAX, MetalNativeError::InvalidArgument, "elementwise: dimension exceeds uint32_t range");
        uint32_t num_elements = static_cast<uint32_t>(numel);
        [encoder setBytes:&num_elements length:sizeof(uint32_t) atIndex:2];

        // Dispatch threads: kernel processes 4 elements per thread
        MTLSize grid_size = MTLSizeMake((numel + 3) / 4, 1, 1);
        MTLSize threadgroup_size = MTLSizeMake(
            std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        // OPT-5: Use commit_and_continue to allow command buffer reuse.
        cmd_pipeline.commit_and_continue();
    }

    return output;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Binary operations
// ---------------------------------------------------------------------------

MNTensor add(const MNTensor& a, const MNTensor& b, MNDevice& device) {
    return dispatch_binary_op(a, b, "add", device);
}

MNTensor sub(const MNTensor& a, const MNTensor& b, MNDevice& device) {
    return dispatch_binary_op(a, b, "sub", device);
}

MNTensor mul(const MNTensor& a, const MNTensor& b, MNDevice& device) {
    return dispatch_binary_op(a, b, "mul", device);
}

MNTensor div(const MNTensor& a, const MNTensor& b, MNDevice& device) {
    return dispatch_binary_op(a, b, "div", device);
}

// ---------------------------------------------------------------------------
// Unary operations
// ---------------------------------------------------------------------------

MNTensor exp(const MNTensor& x, MNDevice& device) {
    return dispatch_unary_op(x, "exp", device);
}

MNTensor log(const MNTensor& x, MNDevice& device) {
    return dispatch_unary_op(x, "log", device);
}

MNTensor sqrt(const MNTensor& x, MNDevice& device) {
    MN_CHECK(x.is_contiguous(),
             MetalNativeError::InvalidArgument,
             "sqrt: input must be contiguous");

    StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
    MNTensor output = MNTensor::empty(x.shape(), x.dtype(), device, out_mode);
    const int64_t numel = x.numel();

    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        const char* kernel_name = nullptr;
        if (x.dtype() == MNDType::Float32) {
            kernel_name = "sqrt_fp32";
        } else if (x.dtype() == MNDType::Float16) {
            kernel_name = "sqrt_fp16";
        } else if (x.dtype() == MNDType::BFloat16) {
            kernel_name = "sqrt_bf16";
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "sqrt: unsupported dtype");
        }

        id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);

        id<MTLCommandBuffer> command_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:x.buffer()->metal_buffer() offset:x.offset() atIndex:0];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:1];

        // Pass num_elements to kernel (required at buffer(2))
        MN_CHECK(numel <= UINT32_MAX, MetalNativeError::InvalidArgument, "elementwise: dimension exceeds uint32_t range");
        uint32_t num_elements = static_cast<uint32_t>(numel);
        [encoder setBytes:&num_elements length:sizeof(uint32_t) atIndex:2];

        // Dispatch threads: kernel processes 4 elements per thread
        MTLSize grid_size = MTLSizeMake((numel + 3) / 4, 1, 1);
        MTLSize threadgroup_size = MTLSizeMake(
            std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        // OPT-5: Use commit_and_continue to allow command buffer reuse.
        cmd_pipeline.commit_and_continue();
    }

    return output;
}

MNTensor abs(const MNTensor& x, MNDevice& device) {
    return dispatch_unary_op(x, "abs", device);
}

MNTensor neg(const MNTensor& x, MNDevice& device) {
    return dispatch_unary_op(x, "neg", device);
}

// ---------------------------------------------------------------------------
// Conditional operations
// ---------------------------------------------------------------------------

MNTensor clamp(const MNTensor& x, float min_val, float max_val, MNDevice& device) {
    MN_CHECK(x.is_contiguous(),
             MetalNativeError::InvalidArgument,
             "clamp: input must be contiguous");

    StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
    MNTensor output = MNTensor::empty(x.shape(), x.dtype(), device, out_mode);
    const int64_t numel = x.numel();

    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        const char* kernel_name = nullptr;
        if (x.dtype() == MNDType::Float32) {
            kernel_name = "clamp_fp32";
        } else if (x.dtype() == MNDType::Float16) {
            kernel_name = "clamp_fp16";
        } else if (x.dtype() == MNDType::BFloat16) {
            kernel_name = "clamp_bf16";
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "clamp: unsupported dtype");
        }

        id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);

        id<MTLCommandBuffer> command_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:x.buffer()->metal_buffer() offset:x.offset() atIndex:0];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:1];

        float bounds[2] = {min_val, max_val};
        [encoder setBytes:&bounds length:sizeof(bounds) atIndex:2];

        // Pass num_elements to kernel (required at buffer(3))
        MN_CHECK(numel <= UINT32_MAX, MetalNativeError::InvalidArgument, "elementwise: dimension exceeds uint32_t range");
        uint32_t num_elements = static_cast<uint32_t>(numel);
        [encoder setBytes:&num_elements length:sizeof(uint32_t) atIndex:3];

        // Dispatch threads: kernel processes 4 elements per thread
        MTLSize grid_size = MTLSizeMake((numel + 3) / 4, 1, 1);
        MTLSize threadgroup_size = MTLSizeMake(
            std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        // OPT-5: Use commit_and_continue to allow command buffer reuse.
        cmd_pipeline.commit_and_continue();
    }

    return output;
}

MNTensor where(const MNTensor& condition,
               const MNTensor& x,
               const MNTensor& y,
               MNDevice& device) {
    MN_CHECK(condition.dtype() == MNDType::Bool,
             MetalNativeError::InvalidArgument,
             "where: condition must be Bool dtype");

    MN_CHECK(x.dtype() == y.dtype(),
             MetalNativeError::InvalidArgument,
             "where: x and y must have same dtype");

    MN_CHECK(condition.is_contiguous() && x.is_contiguous() && y.is_contiguous(),
             MetalNativeError::InvalidArgument,
             "where: all inputs must be contiguous");

    // Compute broadcast shape
    MNShape shape1 = condition.shape().broadcast_with(x.shape());
    MNShape output_shape = shape1.broadcast_with(y.shape());

    StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
    MNTensor output = MNTensor::empty(output_shape, x.dtype(), device, out_mode);
    const int64_t numel = output.numel();

    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        const char* kernel_name = nullptr;
        if (x.dtype() == MNDType::Float32) {
            kernel_name = "where_fp32";
        } else if (x.dtype() == MNDType::Float16) {
            kernel_name = "where_fp16";
        } else if (x.dtype() == MNDType::BFloat16) {
            kernel_name = "where_bf16";
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "where: unsupported dtype");
        }

        id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);

        id<MTLCommandBuffer> command_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:condition.buffer()->metal_buffer() offset:condition.offset() atIndex:0];
        [encoder setBuffer:x.buffer()->metal_buffer() offset:x.offset() atIndex:1];
        [encoder setBuffer:y.buffer()->metal_buffer() offset:y.offset() atIndex:2];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:3];

        MTLSize grid_size = MTLSizeMake(numel, 1, 1);
        MTLSize threadgroup_size = MTLSizeMake(
            std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        // OPT-5: Use commit_and_continue to allow command buffer reuse.
        cmd_pipeline.commit_and_continue();
    }

    return output;
}

} // namespace metal_native
