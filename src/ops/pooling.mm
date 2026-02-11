/// @file pooling.mm
/// @brief Objective-C++ implementation of pooling operators.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>

#include "metal_native/ops/pooling.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

namespace metal_native {

namespace {

MPSDataType to_mps_datatype(MNDType dtype) {
    switch (dtype) {
        case MNDType::Float32:  return MPSDataTypeFloat32;
        case MNDType::Float16:  return MPSDataTypeFloat16;
        case MNDType::BFloat16: return MPSDataTypeBFloat16;
        default:
            MN_THROW(MetalNativeError::InvalidArgument,
                     "pooling: unsupported dtype");
    }
}

NSArray<NSNumber*>* to_ns_shape(const MNShape& shape) {
    NSMutableArray<NSNumber*>* ns_shape = [NSMutableArray arrayWithCapacity:shape.ndim()];
    for (size_t i = 0; i < shape.ndim(); ++i) {
        [ns_shape addObject:@(shape[static_cast<int64_t>(i)])];
    }
    return ns_shape;
}

} // anonymous namespace

MNTensor max_pool2d(const MNTensor& input,
                    const std::vector<size_t>& kernel_size,
                    const std::vector<size_t>& stride,
                    const std::vector<size_t>& padding) {
    MN_CHECK(input.ndim() == 4,
             MetalNativeError::InvalidArgument,
             "max_pool2d: input must be 4D [N, C, H, W]");
    MN_CHECK(kernel_size.size() == 2,
             MetalNativeError::InvalidArgument,
             "max_pool2d: kernel_size must have 2 elements");
    MN_CHECK(padding.size() == 2,
             MetalNativeError::InvalidArgument,
             "max_pool2d: padding must have 2 elements");

    // Default stride to kernel_size if not provided
    std::vector<size_t> actual_stride = stride.empty() ? kernel_size : stride;
    MN_CHECK(actual_stride.size() == 2,
             MetalNativeError::InvalidArgument,
             "max_pool2d: stride must have 2 elements");

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();

        // Create MPSGraph
        MPSGraph* graph = [[MPSGraph alloc] init];

        // Create placeholder tensor
        MPSGraphTensor* input_tensor = [graph placeholderWithShape:to_ns_shape(input.shape())
                                                           dataType:to_mps_datatype(input.dtype())
                                                               name:@"input"];

        // Create max pooling descriptor
        MPSGraphPooling2DOpDescriptor* desc = [[MPSGraphPooling2DOpDescriptor alloc] init];
        desc.kernelWidth = kernel_size[1];
        desc.kernelHeight = kernel_size[0];
        desc.strideInX = actual_stride[1];
        desc.strideInY = actual_stride[0];
        desc.paddingLeft = padding[1];
        desc.paddingRight = padding[1];
        desc.paddingTop = padding[0];
        desc.paddingBottom = padding[0];
        desc.paddingStyle = MPSGraphPaddingStyleExplicit;
        desc.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;

        // Perform max pooling
        MPSGraphTensor* result_tensor = [graph maxPooling2DWithSourceTensor:input_tensor
                                                                 descriptor:desc
                                                                       name:@"max_pool2d"];

        MN_CHECK(result_tensor != nil,
                 MetalNativeError::InternalError,
                 "max_pool2d: MPSGraph operation failed");

        // Create MPSGraphTensorData for input
        MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input.buffer()->metal_buffer()
                       shape:to_ns_shape(input.shape())
                    dataType:to_mps_datatype(input.dtype())];

        // Compute output shape
        int64_t N = input.shape()[0];
        int64_t C = input.shape()[1];
        int64_t H_in = input.shape()[2];
        int64_t W_in = input.shape()[3];

        int64_t H_out = (H_in + 2 * padding[0] - kernel_size[0]) / actual_stride[0] + 1;
        int64_t W_out = (W_in + 2 * padding[1] - kernel_size[1]) / actual_stride[1] + 1;

        MNShape output_shape({N, C, H_out, W_out});

        // Allocate output tensor
        MNTensor result = MNTensor::empty(output_shape, input.dtype(), device);


        // Execute the graph
        id<MTLCommandQueue> queue = device.command_queue();
        MPSGraphExecutionDescriptor* exec_desc = [[MPSGraphExecutionDescriptor alloc] init];
        exec_desc.waitUntilCompleted = NO;

        NSDictionary* feeds = @{input_tensor: input_data};

        NSDictionary* results = [graph runAsyncWithMTLCommandQueue:queue
                                                             feeds:feeds
                                                    targetTensors:@[result_tensor]
                                                 targetOperations:nil
                                                executionDescriptor:exec_desc];

        MN_CHECK(results != nil && results[result_tensor] != nil,
                 MetalNativeError::InternalError,
                 "max_pool2d: graph execution failed");

        return result;
    }
}

MNTensor avg_pool2d(const MNTensor& input,
                    const std::vector<size_t>& kernel_size,
                    const std::vector<size_t>& stride,
                    const std::vector<size_t>& padding) {
    MN_CHECK(input.ndim() == 4,
             MetalNativeError::InvalidArgument,
             "avg_pool2d: input must be 4D [N, C, H, W]");
    MN_CHECK(kernel_size.size() == 2,
             MetalNativeError::InvalidArgument,
             "avg_pool2d: kernel_size must have 2 elements");
    MN_CHECK(padding.size() == 2,
             MetalNativeError::InvalidArgument,
             "avg_pool2d: padding must have 2 elements");

    // Default stride to kernel_size if not provided
    std::vector<size_t> actual_stride = stride.empty() ? kernel_size : stride;
    MN_CHECK(actual_stride.size() == 2,
             MetalNativeError::InvalidArgument,
             "avg_pool2d: stride must have 2 elements");

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();

        // Create MPSGraph
        MPSGraph* graph = [[MPSGraph alloc] init];

        // Create placeholder tensor
        MPSGraphTensor* input_tensor = [graph placeholderWithShape:to_ns_shape(input.shape())
                                                           dataType:to_mps_datatype(input.dtype())
                                                               name:@"input"];

        // Create avg pooling descriptor
        MPSGraphPooling2DOpDescriptor* desc = [[MPSGraphPooling2DOpDescriptor alloc] init];
        desc.kernelWidth = kernel_size[1];
        desc.kernelHeight = kernel_size[0];
        desc.strideInX = actual_stride[1];
        desc.strideInY = actual_stride[0];
        desc.paddingLeft = padding[1];
        desc.paddingRight = padding[1];
        desc.paddingTop = padding[0];
        desc.paddingBottom = padding[0];
        desc.paddingStyle = MPSGraphPaddingStyleExplicit;
        desc.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;

        // Perform average pooling
        MPSGraphTensor* result_tensor = [graph avgPooling2DWithSourceTensor:input_tensor
                                                                 descriptor:desc
                                                                       name:@"avg_pool2d"];

        MN_CHECK(result_tensor != nil,
                 MetalNativeError::InternalError,
                 "avg_pool2d: MPSGraph operation failed");

        // Create MPSGraphTensorData for input
        MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input.buffer()->metal_buffer()
                       shape:to_ns_shape(input.shape())
                    dataType:to_mps_datatype(input.dtype())];

        // Compute output shape
        int64_t N = input.shape()[0];
        int64_t C = input.shape()[1];
        int64_t H_in = input.shape()[2];
        int64_t W_in = input.shape()[3];

        int64_t H_out = (H_in + 2 * padding[0] - kernel_size[0]) / actual_stride[0] + 1;
        int64_t W_out = (W_in + 2 * padding[1] - kernel_size[1]) / actual_stride[1] + 1;

        MNShape output_shape({N, C, H_out, W_out});

        // Allocate output tensor
        MNTensor result = MNTensor::empty(output_shape, input.dtype(), device);


        // Execute the graph
        id<MTLCommandQueue> queue = device.command_queue();
        MPSGraphExecutionDescriptor* exec_desc = [[MPSGraphExecutionDescriptor alloc] init];
        exec_desc.waitUntilCompleted = NO;

        NSDictionary* feeds = @{input_tensor: input_data};

        NSDictionary* results = [graph runAsyncWithMTLCommandQueue:queue
                                                             feeds:feeds
                                                    targetTensors:@[result_tensor]
                                                 targetOperations:nil
                                                executionDescriptor:exec_desc];

        MN_CHECK(results != nil && results[result_tensor] != nil,
                 MetalNativeError::InternalError,
                 "avg_pool2d: graph execution failed");

        return result;
    }
}

MNTensor adaptive_avg_pool2d(const MNTensor& input,
                             const std::vector<size_t>& output_size) {
    MN_CHECK(input.ndim() == 4,
             MetalNativeError::InvalidArgument,
             "adaptive_avg_pool2d: input must be 4D [N, C, H, W]");
    MN_CHECK(output_size.size() == 2,
             MetalNativeError::InvalidArgument,
             "adaptive_avg_pool2d: output_size must have 2 elements");

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();

        // Create MPSGraph
        MPSGraph* graph = [[MPSGraph alloc] init];

        // Create placeholder tensor
        MPSGraphTensor* input_tensor = [graph placeholderWithShape:to_ns_shape(input.shape())
                                                           dataType:to_mps_datatype(input.dtype())
                                                               name:@"input"];

        // Use resize operation to achieve adaptive pooling
        NSArray<NSNumber*>* target_shape = @[
            @(input.shape()[0]),
            @(input.shape()[1]),
            @(output_size[0]),
            @(output_size[1])
        ];

        MPSGraphTensor* result_tensor = [graph resizeTensor:input_tensor
                                                       size:target_shape
                                                       mode:MPSGraphResizeNearest
                                                 centerResult:YES
                                                 alignCorners:NO
                                                     layout:MPSGraphTensorNamedDataLayoutNCHW
                                                       name:@"adaptive_avg_pool2d"];

        MN_CHECK(result_tensor != nil,
                 MetalNativeError::InternalError,
                 "adaptive_avg_pool2d: MPSGraph operation failed");

        // Create MPSGraphTensorData for input
        MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input.buffer()->metal_buffer()
                       shape:to_ns_shape(input.shape())
                    dataType:to_mps_datatype(input.dtype())];

        // Compute output shape
        int64_t N = input.shape()[0];
        int64_t C = input.shape()[1];
        int64_t H_out = output_size[0];
        int64_t W_out = output_size[1];

        MNShape output_shape({N, C, H_out, W_out});

        // Allocate output tensor
        MNTensor result = MNTensor::empty(output_shape, input.dtype(), device);


        // Execute the graph
        id<MTLCommandQueue> queue = device.command_queue();
        MPSGraphExecutionDescriptor* exec_desc = [[MPSGraphExecutionDescriptor alloc] init];
        exec_desc.waitUntilCompleted = NO;

        NSDictionary* feeds = @{input_tensor: input_data};

        NSDictionary* results = [graph runAsyncWithMTLCommandQueue:queue
                                                             feeds:feeds
                                                    targetTensors:@[result_tensor]
                                                 targetOperations:nil
                                                executionDescriptor:exec_desc];

        MN_CHECK(results != nil && results[result_tensor] != nil,
                 MetalNativeError::InternalError,
                 "adaptive_avg_pool2d: graph execution failed");

        return result;
    }
}

} // namespace metal_native
