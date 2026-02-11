/// @file conv.mm
/// @brief Objective-C++ implementation of convolution operators.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>

#include "metal_native/ops/conv.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/graph/graph_cache.h"
#include "metal_native/memory/budget_controller.h"

#include <mutex>
#include <unordered_map>
#include <mach/mach_time.h>

namespace metal_native {

namespace {

MPSDataType to_mps_datatype(MNDType dtype) {
    switch (dtype) {
        case MNDType::Float32:  return MPSDataTypeFloat32;
        case MNDType::Float16:  return MPSDataTypeFloat16;
        case MNDType::BFloat16: return MPSDataTypeBFloat16;
        default:
            MN_THROW(MetalNativeError::InvalidArgument,
                     "conv2d: unsupported dtype");
    }
}

NSArray<NSNumber*>* to_ns_shape(const MNShape& shape) {
    NSMutableArray<NSNumber*>* ns_shape = [NSMutableArray arrayWithCapacity:shape.ndim()];
    for (size_t i = 0; i < shape.ndim(); ++i) {
        [ns_shape addObject:@(shape[static_cast<int64_t>(i)])];
    }
    return ns_shape;
}

// Cache entry for conv2d graphs
struct Conv2dCacheEntry {
    MPSGraph* graph;
    MPSGraphTensor* input_placeholder;
    MPSGraphTensor* weight_placeholder;
    MPSGraphTensor* bias_placeholder;  // may be nil if no bias
    MPSGraphTensor* result_tensor;
    bool has_bias;
    uint64_t last_access;
};

static uint64_t conv_monotonic_tick() {
    return mach_absolute_time();
}

static constexpr size_t kConvDefaultMaxCacheEntries = 64;

std::mutex conv2d_cache_mu;
std::unordered_map<GraphCacheKey, Conv2dCacheEntry> conv2d_graph_cache;
size_t conv2d_cache_max_entries = kConvDefaultMaxCacheEntries;

static void conv_evict_lru_if_needed() {
    if (conv2d_graph_cache.size() < conv2d_cache_max_entries) return;
    auto oldest = conv2d_graph_cache.begin();
    for (auto it = conv2d_graph_cache.begin(); it != conv2d_graph_cache.end(); ++it) {
        if (it->second.last_access < oldest->second.last_access) {
            oldest = it;
        }
    }
    conv2d_graph_cache.erase(oldest);
}

// Compute cache key from operation parameters
GraphCacheKey make_conv2d_key(const MNShape& input_shape, const MNShape& weight_shape,
                               bool has_bias, MNDType dtype,
                               const std::vector<size_t>& stride,
                               const std::vector<size_t>& padding,
                               const std::vector<size_t>& dilation,
                               size_t groups) {
    GraphCacheKey key;
    // Use a simple hash for conv2d topology: op_type + stride + padding + dilation + groups
    key.topology_hash = 0x434F4E5632440000ULL;  // "CONV2D\0\0"
    key.topology_hash ^= (stride[0] << 32) | (stride[1] << 24);
    key.topology_hash ^= (padding[0] << 16) | (padding[1] << 8);
    key.topology_hash ^= (dilation[0] << 4) | (dilation[1]);
    key.topology_hash ^= (groups << 48);
    key.topology_hash ^= (has_bias ? 0x1ULL : 0x0ULL);

    // Shape tuple: flatten all input shapes
    for (size_t i = 0; i < input_shape.ndim(); ++i) {
        key.shape_tuple.push_back(static_cast<size_t>(input_shape[static_cast<int64_t>(i)]));
    }
    key.shape_tuple.push_back(0);  // separator
    for (size_t i = 0; i < weight_shape.ndim(); ++i) {
        key.shape_tuple.push_back(static_cast<size_t>(weight_shape[static_cast<int64_t>(i)]));
    }
    key.dtype = dtype;
    return key;
}

} // anonymous namespace

MNTensor conv2d(const MNTensor& input,
                const MNTensor& weight,
                const MNTensor& bias,
                const std::vector<size_t>& stride,
                const std::vector<size_t>& padding,
                const std::vector<size_t>& dilation,
                size_t groups) {
    MN_CHECK(input.ndim() == 4,
             MetalNativeError::InvalidArgument,
             "conv2d: input must be 4D [N, C_in, H, W]");
    MN_CHECK(weight.ndim() == 4,
             MetalNativeError::InvalidArgument,
             "conv2d: weight must be 4D [C_out, C_in/groups, kH, kW]");
    MN_CHECK(stride.size() == 2,
             MetalNativeError::InvalidArgument,
             "conv2d: stride must have 2 elements");
    MN_CHECK(padding.size() == 2,
             MetalNativeError::InvalidArgument,
             "conv2d: padding must have 2 elements");
    MN_CHECK(dilation.size() == 2,
             MetalNativeError::InvalidArgument,
             "conv2d: dilation must have 2 elements");
    MN_CHECK(input.dtype() == weight.dtype(),
             MetalNativeError::InvalidArgument,
             "conv2d: input and weight dtypes must match");

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();

        // Check cache first
        bool has_bias = (bias.numel() > 0);
        GraphCacheKey cache_key = make_conv2d_key(input.shape(), weight.shape(), has_bias,
                                                   input.dtype(), stride, padding, dilation, groups);

        MPSGraph* graph = nil;
        MPSGraphTensor* input_tensor = nil;
        MPSGraphTensor* weight_tensor = nil;
        MPSGraphTensor* bias_tensor = nil;
        MPSGraphTensor* result_tensor = nil;

        {
            std::lock_guard<std::mutex> lock(conv2d_cache_mu);
            auto it = conv2d_graph_cache.find(cache_key);
            if (it != conv2d_graph_cache.end()) {
                // Cache hit: reuse the cached graph
                graph = it->second.graph;
                input_tensor = it->second.input_placeholder;
                weight_tensor = it->second.weight_placeholder;
                bias_tensor = it->second.bias_placeholder;
                result_tensor = it->second.result_tensor;
                it->second.last_access = conv_monotonic_tick();
            }
        }

        // Cache miss: build the graph
        if (graph == nil) {
            graph = [[MPSGraph alloc] init];

            // Create placeholder tensors
            input_tensor = [graph placeholderWithShape:to_ns_shape(input.shape())
                                               dataType:to_mps_datatype(input.dtype())
                                                   name:@"input"];
            weight_tensor = [graph placeholderWithShape:to_ns_shape(weight.shape())
                                                dataType:to_mps_datatype(weight.dtype())
                                                    name:@"weight"];

            // Create convolution descriptor
            MPSGraphConvolution2DOpDescriptor* desc =
                [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride[1]
                                                                  strideInY:stride[0]
                                                            dilationRateInX:dilation[1]
                                                            dilationRateInY:dilation[0]
                                                                     groups:groups
                                                                paddingLeft:padding[1]
                                                               paddingRight:padding[1]
                                                                 paddingTop:padding[0]
                                                              paddingBottom:padding[0]
                                                               paddingStyle:MPSGraphPaddingStyleExplicit
                                                                 dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                              weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

            // Perform convolution
            result_tensor = [graph convolution2DWithSourceTensor:input_tensor
                                                   weightsTensor:weight_tensor
                                                      descriptor:desc
                                                            name:@"conv2d"];

            // Add bias if provided
            if (has_bias) {
                MN_CHECK(bias.ndim() == 1,
                         MetalNativeError::InvalidArgument,
                         "conv2d: bias must be 1D");
                MN_CHECK(bias.dtype() == input.dtype(),
                         MetalNativeError::InvalidArgument,
                         "conv2d: bias dtype must match input dtype");

                bias_tensor = [graph placeholderWithShape:to_ns_shape(bias.shape())
                                                  dataType:to_mps_datatype(bias.dtype())
                                                      name:@"bias"];

                // Reshape bias to [1, C, 1, 1] for broadcasting
                NSArray<NSNumber*>* bias_shape = @[@1, @(bias.shape()[0]), @1, @1];
                MPSGraphTensor* bias_reshaped = [graph reshapeTensor:bias_tensor
                                                           withShape:bias_shape
                                                                name:nil];

                result_tensor = [graph additionWithPrimaryTensor:result_tensor
                                                 secondaryTensor:bias_reshaped
                                                            name:@"conv2d_bias"];
            }

            MN_CHECK(result_tensor != nil,
                     MetalNativeError::InternalError,
                     "conv2d: MPSGraph operation failed");

            // Update cache capacity from budget controller
            auto& bc = MemoryBudgetController::instance();
            float mult = bc.pressure_multiplier();
            conv2d_cache_max_entries = std::max<size_t>(4, static_cast<size_t>(kConvDefaultMaxCacheEntries * mult));

            // Store in cache
            std::lock_guard<std::mutex> lock(conv2d_cache_mu);
            conv_evict_lru_if_needed();
            Conv2dCacheEntry entry;
            entry.graph = graph;
            entry.input_placeholder = input_tensor;
            entry.weight_placeholder = weight_tensor;
            entry.bias_placeholder = bias_tensor;
            entry.result_tensor = result_tensor;
            entry.has_bias = has_bias;
            entry.last_access = conv_monotonic_tick();
            conv2d_graph_cache[cache_key] = entry;
        }

        // Create MPSGraphTensorData for inputs
        MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input.buffer()->metal_buffer()
                       shape:to_ns_shape(input.shape())
                    dataType:to_mps_datatype(input.dtype())];

        MPSGraphTensorData* weight_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:weight.buffer()->metal_buffer()
                       shape:to_ns_shape(weight.shape())
                    dataType:to_mps_datatype(weight.dtype())];

        // Compute output shape
        int64_t N = input.shape()[0];
        int64_t C_out = weight.shape()[0];
        int64_t H_in = input.shape()[2];
        int64_t W_in = input.shape()[3];
        int64_t kH = weight.shape()[2];
        int64_t kW = weight.shape()[3];

        int64_t H_out = (H_in + 2 * padding[0] - dilation[0] * (kH - 1) - 1) / stride[0] + 1;
        int64_t W_out = (W_in + 2 * padding[1] - dilation[1] * (kW - 1) - 1) / stride[1] + 1;

        MNShape output_shape({N, C_out, H_out, W_out});

        // Allocate output tensor
        MNTensor result = MNTensor::empty(output_shape, input.dtype(), device);


        // Execute the graph
        id<MTLCommandQueue> queue = device.command_queue();
        MPSGraphExecutionDescriptor* exec_desc = [[MPSGraphExecutionDescriptor alloc] init];
        exec_desc.waitUntilCompleted = NO;

        NSMutableDictionary* feeds = [NSMutableDictionary dictionaryWithDictionary:@{
            input_tensor: input_data,
            weight_tensor: weight_data
        }];

        if (has_bias) {
            MPSGraphTensorData* bias_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:bias.buffer()->metal_buffer()
                           shape:to_ns_shape(bias.shape())
                        dataType:to_mps_datatype(bias.dtype())];
            feeds[bias_tensor] = bias_data;
        }

        NSDictionary* results = [graph runAsyncWithMTLCommandQueue:queue
                                                             feeds:feeds
                                                    targetTensors:@[result_tensor]
                                                 targetOperations:nil
                                                executionDescriptor:exec_desc];

        MN_CHECK(results != nil && results[result_tensor] != nil,
                 MetalNativeError::InternalError,
                 "conv2d: graph execution failed");

        return result;
    }
}

MNTensor conv1d(const MNTensor& input,
                const MNTensor& weight,
                const MNTensor& bias,
                size_t stride,
                size_t padding,
                size_t dilation,
                size_t groups) {
    (void)input;
    (void)weight;
    (void)bias;
    (void)stride;
    (void)padding;
    (void)dilation;
    (void)groups;

    MN_THROW(MetalNativeError::InternalError,
             "conv1d: not implemented");
}

MNTensor conv3d(const MNTensor& input,
                const MNTensor& weight,
                const MNTensor& bias,
                const std::vector<size_t>& stride,
                const std::vector<size_t>& padding,
                const std::vector<size_t>& dilation,
                size_t groups) {
    (void)input;
    (void)weight;
    (void)bias;
    (void)stride;
    (void)padding;
    (void)dilation;
    (void)groups;

    MN_THROW(MetalNativeError::InternalError,
             "conv3d: not implemented");
}

} // namespace metal_native
