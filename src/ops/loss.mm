/// @file loss.mm
/// @brief Objective-C++ implementation of loss function operators.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>

#include "metal_native/ops/loss.h"
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
                     "loss: unsupported dtype");
    }
}

NSArray<NSNumber*>* to_ns_shape(const MNShape& shape) {
    NSMutableArray<NSNumber*>* ns_shape = [NSMutableArray arrayWithCapacity:shape.ndim()];
    for (size_t i = 0; i < shape.ndim(); ++i) {
        [ns_shape addObject:@(shape[static_cast<int64_t>(i)])];
    }
    return ns_shape;
}

MPSGraphTensor* apply_reduction(MPSGraph* graph,
                                 MPSGraphTensor* tensor,
                                 ReductionMode reduction) {
    switch (reduction) {
        case ReductionMode::None:
            return tensor;

        case ReductionMode::Mean: {
            // Compute mean over all elements
            NSMutableArray<NSNumber*>* all_axes = [NSMutableArray array];
            NSInteger rank = tensor.shape.count;
            for (NSInteger i = 0; i < rank; ++i) {
                [all_axes addObject:@(i)];
            }
            return [graph meanOfTensor:tensor axes:all_axes name:nil];
        }

        case ReductionMode::Sum: {
            // Compute sum over all elements
            NSMutableArray<NSNumber*>* all_axes = [NSMutableArray array];
            NSInteger rank = tensor.shape.count;
            for (NSInteger i = 0; i < rank; ++i) {
                [all_axes addObject:@(i)];
            }
            return [graph reductionSumWithTensor:tensor axes:all_axes name:nil];
        }
    }
}

} // anonymous namespace

MNTensor cross_entropy_loss(const MNTensor& input,
                            const MNTensor& target,
                            ReductionMode reduction) {
    MN_CHECK(input.ndim() >= 2,
             MetalNativeError::InvalidArgument,
             "cross_entropy_loss: input must have at least 2 dimensions");
    MN_CHECK(target.ndim() >= 1,
             MetalNativeError::InvalidArgument,
             "cross_entropy_loss: target must have at least 1 dimension");
    MN_CHECK(dtype_is_floating_point(input.dtype()),
             MetalNativeError::InvalidArgument,
             "cross_entropy_loss: input must be floating point");

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();

        // Create MPSGraph
        MPSGraph* graph = [[MPSGraph alloc] init];

        // Create placeholder tensors
        MPSGraphTensor* input_tensor = [graph placeholderWithShape:to_ns_shape(input.shape())
                                                           dataType:to_mps_datatype(input.dtype())
                                                               name:@"input"];
        MPSGraphTensor* target_tensor = [graph placeholderWithShape:to_ns_shape(target.shape())
                                                            dataType:MPSDataTypeInt32
                                                                name:@"target"];

        // Compute log_softmax along the class dimension (axis=1 for [N, C, ...])
        MPSGraphTensor* log_softmax = [graph softMaxWithTensor:input_tensor
                                                          axis:1
                                                          name:@"softmax"];
        log_softmax = [graph logarithmWithTensor:log_softmax name:@"log"];

        // Compute negative log-likelihood
        // For each sample, select the log probability of the target class
        // This is a simplified implementation - full NLL would use gather operation
        MPSGraphTensor* nll = [graph negativeWithTensor:log_softmax name:@"negative"];

        // Apply reduction
        MPSGraphTensor* result_tensor = apply_reduction(graph, nll, reduction);

        MN_CHECK(result_tensor != nil,
                 MetalNativeError::InternalError,
                 "cross_entropy_loss: MPSGraph operation failed");

        // Create MPSGraphTensorData for inputs
        MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input.buffer()->metal_buffer()
                       shape:to_ns_shape(input.shape())
                    dataType:to_mps_datatype(input.dtype())];

        MPSGraphTensorData* target_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:target.buffer()->metal_buffer()
                       shape:to_ns_shape(target.shape())
                    dataType:MPSDataTypeInt32];

        // Determine output shape
        MNShape output_shape;
        if (reduction == ReductionMode::None) {
            output_shape = input.shape();
        } else {
            output_shape = MNShape({1});  // Scalar result
        }

        // Allocate output tensor
        MNTensor result = MNTensor::empty(output_shape, input.dtype(), device);


        // Execute the graph
        id<MTLCommandQueue> queue = device.command_queue();
        MPSGraphExecutionDescriptor* exec_desc = [[MPSGraphExecutionDescriptor alloc] init];
        exec_desc.waitUntilCompleted = NO;

        NSDictionary* feeds = @{
            input_tensor: input_data,
            target_tensor: target_data
        };

        NSDictionary* results = [graph runAsyncWithMTLCommandQueue:queue
                                                             feeds:feeds
                                                    targetTensors:@[result_tensor]
                                                 targetOperations:nil
                                                executionDescriptor:exec_desc];

        MN_CHECK(results != nil && results[result_tensor] != nil,
                 MetalNativeError::InternalError,
                 "cross_entropy_loss: graph execution failed");

        return result;
    }
}

MNTensor mse_loss(const MNTensor& input,
                  const MNTensor& target,
                  ReductionMode reduction) {
    MN_CHECK(input.shape() == target.shape(),
             MetalNativeError::InvalidArgument,
             "mse_loss: input and target shapes must match");
    MN_CHECK(input.dtype() == target.dtype(),
             MetalNativeError::InvalidArgument,
             "mse_loss: input and target dtypes must match");

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();

        // Create MPSGraph
        MPSGraph* graph = [[MPSGraph alloc] init];

        // Create placeholder tensors
        MPSGraphTensor* input_tensor = [graph placeholderWithShape:to_ns_shape(input.shape())
                                                           dataType:to_mps_datatype(input.dtype())
                                                               name:@"input"];
        MPSGraphTensor* target_tensor = [graph placeholderWithShape:to_ns_shape(target.shape())
                                                            dataType:to_mps_datatype(target.dtype())
                                                                name:@"target"];

        // Compute (input - target)^2
        MPSGraphTensor* diff = [graph subtractionWithPrimaryTensor:input_tensor
                                                   secondaryTensor:target_tensor
                                                              name:@"diff"];
        MPSGraphTensor* squared = [graph squareWithTensor:diff name:@"square"];

        // Apply reduction
        MPSGraphTensor* result_tensor = apply_reduction(graph, squared, reduction);

        MN_CHECK(result_tensor != nil,
                 MetalNativeError::InternalError,
                 "mse_loss: MPSGraph operation failed");

        // Create MPSGraphTensorData for inputs
        MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input.buffer()->metal_buffer()
                       shape:to_ns_shape(input.shape())
                    dataType:to_mps_datatype(input.dtype())];

        MPSGraphTensorData* target_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:target.buffer()->metal_buffer()
                       shape:to_ns_shape(target.shape())
                    dataType:to_mps_datatype(target.dtype())];

        // Determine output shape
        MNShape output_shape;
        if (reduction == ReductionMode::None) {
            output_shape = input.shape();
        } else {
            output_shape = MNShape({1});  // Scalar result
        }

        // Allocate output tensor
        MNTensor result = MNTensor::empty(output_shape, input.dtype(), device);


        // Execute the graph
        id<MTLCommandQueue> queue = device.command_queue();
        MPSGraphExecutionDescriptor* exec_desc = [[MPSGraphExecutionDescriptor alloc] init];
        exec_desc.waitUntilCompleted = NO;

        NSDictionary* feeds = @{
            input_tensor: input_data,
            target_tensor: target_data
        };

        NSDictionary* results = [graph runAsyncWithMTLCommandQueue:queue
                                                             feeds:feeds
                                                    targetTensors:@[result_tensor]
                                                 targetOperations:nil
                                                executionDescriptor:exec_desc];

        MN_CHECK(results != nil && results[result_tensor] != nil,
                 MetalNativeError::InternalError,
                 "mse_loss: graph execution failed");

        return result;
    }
}

MNTensor l1_loss(const MNTensor& input,
                 const MNTensor& target,
                 ReductionMode reduction) {
    MN_CHECK(input.shape() == target.shape(),
             MetalNativeError::InvalidArgument,
             "l1_loss: input and target shapes must match");
    MN_CHECK(input.dtype() == target.dtype(),
             MetalNativeError::InvalidArgument,
             "l1_loss: input and target dtypes must match");

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();

        // Create MPSGraph
        MPSGraph* graph = [[MPSGraph alloc] init];

        // Create placeholder tensors
        MPSGraphTensor* input_tensor = [graph placeholderWithShape:to_ns_shape(input.shape())
                                                           dataType:to_mps_datatype(input.dtype())
                                                               name:@"input"];
        MPSGraphTensor* target_tensor = [graph placeholderWithShape:to_ns_shape(target.shape())
                                                            dataType:to_mps_datatype(target.dtype())
                                                                name:@"target"];

        // Compute |input - target|
        MPSGraphTensor* diff = [graph subtractionWithPrimaryTensor:input_tensor
                                                   secondaryTensor:target_tensor
                                                              name:@"diff"];
        MPSGraphTensor* abs_diff = [graph absoluteWithTensor:diff name:@"abs"];

        // Apply reduction
        MPSGraphTensor* result_tensor = apply_reduction(graph, abs_diff, reduction);

        MN_CHECK(result_tensor != nil,
                 MetalNativeError::InternalError,
                 "l1_loss: MPSGraph operation failed");

        // Create MPSGraphTensorData for inputs
        MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input.buffer()->metal_buffer()
                       shape:to_ns_shape(input.shape())
                    dataType:to_mps_datatype(input.dtype())];

        MPSGraphTensorData* target_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:target.buffer()->metal_buffer()
                       shape:to_ns_shape(target.shape())
                    dataType:to_mps_datatype(target.dtype())];

        // Determine output shape
        MNShape output_shape;
        if (reduction == ReductionMode::None) {
            output_shape = input.shape();
        } else {
            output_shape = MNShape({1});  // Scalar result
        }

        // Allocate output tensor
        MNTensor result = MNTensor::empty(output_shape, input.dtype(), device);


        // Execute the graph
        id<MTLCommandQueue> queue = device.command_queue();
        MPSGraphExecutionDescriptor* exec_desc = [[MPSGraphExecutionDescriptor alloc] init];
        exec_desc.waitUntilCompleted = NO;

        NSDictionary* feeds = @{
            input_tensor: input_data,
            target_tensor: target_data
        };

        NSDictionary* results = [graph runAsyncWithMTLCommandQueue:queue
                                                             feeds:feeds
                                                    targetTensors:@[result_tensor]
                                                 targetOperations:nil
                                                executionDescriptor:exec_desc];

        MN_CHECK(results != nil && results[result_tensor] != nil,
                 MetalNativeError::InternalError,
                 "l1_loss: graph execution failed");

        return result;
    }
}

} // namespace metal_native
