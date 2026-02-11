/// @file graph_builder.mm
/// @brief Objective-C++ implementation of GraphBuilder.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>

#include "metal_native/graph/graph_builder.h"
#include "metal_native/graph/fusion_patterns.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace metal_native {

// ---------------------------------------------------------------------------
// GraphExecutable::Impl
// ---------------------------------------------------------------------------

struct GraphExecutable::Impl {
    MPSGraphExecutable* executable = nil;

    explicit Impl(MPSGraphExecutable* exec) : executable(exec) {}
    ~Impl() {
        executable = nil;
    }
};

GraphExecutable::GraphExecutable() : impl_(std::make_unique<Impl>(nil)) {}

GraphExecutable::GraphExecutable(MPSGraphExecutable* exec)
    : impl_(std::make_unique<Impl>(exec)) {}

GraphExecutable::~GraphExecutable() = default;

GraphExecutable::GraphExecutable(GraphExecutable&&) noexcept = default;
GraphExecutable& GraphExecutable::operator=(GraphExecutable&&) noexcept = default;

MPSGraphExecutable* GraphExecutable::executable() const noexcept {
    return impl_->executable;
}

// ---------------------------------------------------------------------------
// GraphBuilder::Impl
// ---------------------------------------------------------------------------

struct GraphBuilder::Impl {
    MNDevice& device;
    MPSGraph* graph = nil;
    std::unordered_map<std::string, MPSGraphTensor*> tensor_map;
    NSMutableArray<MPSGraphTensor*>* output_tensors = nil;
    mutable std::mutex mu;

    // Recorded operation sequence for fusion analysis
    struct RecordedOp {
        OpType type;
        void* tensor_handle;  // MPSGraphTensor* as void*
    };
    std::vector<RecordedOp> recorded_ops;
    size_t fusions_applied = 0;

    explicit Impl(MNDevice& dev) : device(dev) {
        @autoreleasepool {
            graph = [[MPSGraph alloc] init];
            output_tensors = [[NSMutableArray alloc] init];
            MN_CHECK(graph != nil,
                     MetalNativeError::InternalError,
                     "GraphBuilder: failed to create MPSGraph");
        }
    }

    ~Impl() {
        graph = nil;
        output_tensors = nil;
    }

    // -- Helpers -------------------------------------------------------------

    MPSGraphTensor* get_tensor(void* handle) {
        return (__bridge MPSGraphTensor*)(handle);
    }

    size_t optimize_locked() {
        if (recorded_ops.empty()) return 0;
        std::vector<OpType> op_sequence;
        op_sequence.reserve(recorded_ops.size());
        for (const auto& rec : recorded_ops) {
            op_sequence.push_back(rec.type);
        }
        auto matches = FusionRegistry::instance().find_all_fusions(op_sequence);
        fusions_applied = matches.size();
        return fusions_applied;
    }

    MPSDataType to_mps_datatype(MNDType dtype) {
        switch (dtype) {
            case MNDType::Float32:  return MPSDataTypeFloat32;
            case MNDType::Float16:  return MPSDataTypeFloat16;
            case MNDType::BFloat16: return MPSDataTypeBFloat16;
            case MNDType::Int64:    return MPSDataTypeInt64;
            case MNDType::Int32:    return MPSDataTypeInt32;
            case MNDType::Int16:    return MPSDataTypeInt16;
            case MNDType::Int8:     return MPSDataTypeInt8;
            case MNDType::UInt8:    return MPSDataTypeUInt8;
            case MNDType::Bool:     return MPSDataTypeBool;
        }
        MN_THROW(MetalNativeError::InvalidArgument,
                 "GraphBuilder: unsupported dtype");
    }

    NSArray<NSNumber*>* to_ns_shape(const MNShape& shape) {
        NSMutableArray<NSNumber*>* ns_shape = [NSMutableArray arrayWithCapacity:shape.ndim()];
        for (size_t i = 0; i < shape.ndim(); ++i) {
            [ns_shape addObject:@(shape[static_cast<int64_t>(i)])];
        }
        return ns_shape;
    }
};

// ---------------------------------------------------------------------------
// GraphBuilder public API
// ---------------------------------------------------------------------------

GraphBuilder::GraphBuilder(MNDevice& device)
    : impl_(std::make_unique<Impl>(device)) {}

GraphBuilder::~GraphBuilder() = default;

void* GraphBuilder::add_placeholder(const std::string& name,
                                     const MNShape& shape,
                                     MNDType dtype) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    @autoreleasepool {
        NSArray<NSNumber*>* ns_shape = impl_->to_ns_shape(shape);
        MPSDataType mps_dtype = impl_->to_mps_datatype(dtype);

        MPSGraphTensor* tensor = [impl_->graph placeholderWithShape:ns_shape
                                                           dataType:mps_dtype
                                                               name:@(name.c_str())];
        MN_CHECK(tensor != nil,
                 MetalNativeError::InternalError,
                 "GraphBuilder: failed to create placeholder '" + name + "'");

        impl_->tensor_map[name] = tensor;
        return (__bridge void*)(tensor);
    }
}

void* GraphBuilder::add_matmul(void* lhs, void* rhs) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    @autoreleasepool {
        MPSGraphTensor* lhs_tensor = impl_->get_tensor(lhs);
        MPSGraphTensor* rhs_tensor = impl_->get_tensor(rhs);

        MPSGraphTensor* result = [impl_->graph matrixMultiplicationWithPrimaryTensor:lhs_tensor
                                                                     secondaryTensor:rhs_tensor
                                                                                name:nil];
        MN_CHECK(result != nil,
                 MetalNativeError::InternalError,
                 "GraphBuilder: matmul operation failed");

        impl_->recorded_ops.push_back({OpType::MatMul, (__bridge void*)(result)});
        return (__bridge void*)(result);
    }
}

void* GraphBuilder::add_conv2d(void* input,
                                void* weight,
                                size_t stride_h,
                                size_t stride_w,
                                size_t padding_h,
                                size_t padding_w,
                                size_t dilation_h,
                                size_t dilation_w) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    @autoreleasepool {
        MPSGraphTensor* input_tensor = impl_->get_tensor(input);
        MPSGraphTensor* weight_tensor = impl_->get_tensor(weight);

        MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride_w
                                                                                                   strideInY:stride_h
                                                                                             dilationRateInX:dilation_w
                                                                                             dilationRateInY:dilation_h
                                                                                                      groups:1
                                                                                                paddingLeft:padding_w
                                                                                                paddingRight:padding_w
                                                                                                 paddingTop:padding_h
                                                                                              paddingBottom:padding_h
                                                                                                paddingStyle:MPSGraphPaddingStyleExplicit
                                                                                                  dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                                                              weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

        MPSGraphTensor* result = [impl_->graph convolution2DWithSourceTensor:input_tensor
                                                               weightsTensor:weight_tensor
                                                                  descriptor:desc
                                                                        name:nil];
        MN_CHECK(result != nil,
                 MetalNativeError::InternalError,
                 "GraphBuilder: conv2d operation failed");

        impl_->recorded_ops.push_back({OpType::Conv2D, (__bridge void*)(result)});
        return (__bridge void*)(result);
    }
}

void* GraphBuilder::add_relu(void* input) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    @autoreleasepool {
        MPSGraphTensor* input_tensor = impl_->get_tensor(input);
        MPSGraphTensor* result = [impl_->graph reLUWithTensor:input_tensor
                                                         name:nil];
        MN_CHECK(result != nil,
                 MetalNativeError::InternalError,
                 "GraphBuilder: relu operation failed");

        impl_->recorded_ops.push_back({OpType::ReLU, (__bridge void*)(result)});
        return (__bridge void*)(result);
    }
}

void* GraphBuilder::add_gelu(void* input) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    @autoreleasepool {
        MPSGraphTensor* input_tensor = impl_->get_tensor(input);

        // GELU(x) = x * Φ(x) where Φ is the cumulative distribution function
        // of the standard normal distribution.
        // MPSGraph doesn't have a direct GELU op, so we approximate:
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

        MPSGraphTensor* x = input_tensor;

        // Constants
        MPSGraphTensor* half = [impl_->graph constantWithScalar:0.5
                                                        dataType:x.dataType];
        MPSGraphTensor* one = [impl_->graph constantWithScalar:1.0
                                                      dataType:x.dataType];
        MPSGraphTensor* sqrt_2_over_pi = [impl_->graph constantWithScalar:0.7978845608
                                                                 dataType:x.dataType];
        MPSGraphTensor* coeff = [impl_->graph constantWithScalar:0.044715
                                                        dataType:x.dataType];
        MPSGraphTensor* three = [impl_->graph constantWithScalar:3.0
                                                        dataType:x.dataType];

        // x^3
        MPSGraphTensor* x_cubed = [impl_->graph powerWithPrimaryTensor:x
                                                       secondaryTensor:three
                                                                  name:nil];

        // 0.044715 * x^3
        MPSGraphTensor* term = [impl_->graph multiplicationWithPrimaryTensor:coeff
                                                             secondaryTensor:x_cubed
                                                                        name:nil];

        // x + 0.044715 * x^3
        MPSGraphTensor* inner_sum = [impl_->graph additionWithPrimaryTensor:x
                                                            secondaryTensor:term
                                                                       name:nil];

        // sqrt(2/π) * (x + 0.044715 * x^3)
        MPSGraphTensor* scaled = [impl_->graph multiplicationWithPrimaryTensor:sqrt_2_over_pi
                                                               secondaryTensor:inner_sum
                                                                          name:nil];

        // tanh(sqrt(2/π) * (x + 0.044715 * x^3))
        MPSGraphTensor* tanh_result = [impl_->graph tanhWithTensor:scaled
                                                              name:nil];

        // 1 + tanh(...)
        MPSGraphTensor* one_plus_tanh = [impl_->graph additionWithPrimaryTensor:one
                                                                secondaryTensor:tanh_result
                                                                           name:nil];

        // x * (1 + tanh(...))
        MPSGraphTensor* x_scaled = [impl_->graph multiplicationWithPrimaryTensor:x
                                                                 secondaryTensor:one_plus_tanh
                                                                            name:nil];

        // 0.5 * x * (1 + tanh(...))
        MPSGraphTensor* result = [impl_->graph multiplicationWithPrimaryTensor:half
                                                               secondaryTensor:x_scaled
                                                                          name:nil];

        MN_CHECK(result != nil,
                 MetalNativeError::InternalError,
                 "GraphBuilder: gelu operation failed");

        impl_->recorded_ops.push_back({OpType::GELU, (__bridge void*)(result)});
        return (__bridge void*)(result);
    }
}

void* GraphBuilder::add_softmax(void* input) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    @autoreleasepool {
        MPSGraphTensor* input_tensor = impl_->get_tensor(input);

        // Softmax along the last axis (axis = -1)
        MPSGraphTensor* result = [impl_->graph softMaxWithTensor:input_tensor
                                                            axis:-1
                                                            name:nil];
        MN_CHECK(result != nil,
                 MetalNativeError::InternalError,
                 "GraphBuilder: softmax operation failed");

        impl_->recorded_ops.push_back({OpType::Softmax, (__bridge void*)(result)});
        return (__bridge void*)(result);
    }
}

void* GraphBuilder::add_layer_norm(void* input,
                                    const std::vector<int64_t>& axes,
                                    float eps) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    @autoreleasepool {
        MPSGraphTensor* input_tensor = impl_->get_tensor(input);

        // Convert axes to NSArray
        NSMutableArray<NSNumber*>* ns_axes = [NSMutableArray arrayWithCapacity:axes.size()];
        for (int64_t axis : axes) {
            [ns_axes addObject:@(axis)];
        }

        // Compute mean
        MPSGraphTensor* mean = [impl_->graph meanOfTensor:input_tensor
                                                     axes:ns_axes
                                                     name:nil];

        // x - mean
        MPSGraphTensor* centered = [impl_->graph subtractionWithPrimaryTensor:input_tensor
                                                               secondaryTensor:mean
                                                                          name:nil];

        // (x - mean)^2
        MPSGraphTensor* squared = [impl_->graph squareWithTensor:centered
                                                            name:nil];

        // variance = mean((x - mean)^2)
        MPSGraphTensor* variance = [impl_->graph meanOfTensor:squared
                                                         axes:ns_axes
                                                         name:nil];

        // variance + eps
        MPSGraphTensor* eps_tensor = [impl_->graph constantWithScalar:eps
                                                             dataType:input_tensor.dataType];
        MPSGraphTensor* var_eps = [impl_->graph additionWithPrimaryTensor:variance
                                                          secondaryTensor:eps_tensor
                                                                     name:nil];

        // rsqrt(variance + eps)
        MPSGraphTensor* rsqrt = [impl_->graph reciprocalSquareRootWithTensor:var_eps
                                                                         name:nil];

        // (x - mean) * rsqrt(variance + eps)
        MPSGraphTensor* result = [impl_->graph multiplicationWithPrimaryTensor:centered
                                                               secondaryTensor:rsqrt
                                                                          name:nil];

        MN_CHECK(result != nil,
                 MetalNativeError::InternalError,
                 "GraphBuilder: layer_norm operation failed");

        impl_->recorded_ops.push_back({OpType::LayerNorm, (__bridge void*)(result)});
        return (__bridge void*)(result);
    }
}

void* GraphBuilder::add_add(void* lhs, void* rhs) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    @autoreleasepool {
        MPSGraphTensor* lhs_tensor = impl_->get_tensor(lhs);
        MPSGraphTensor* rhs_tensor = impl_->get_tensor(rhs);

        MPSGraphTensor* result = [impl_->graph additionWithPrimaryTensor:lhs_tensor
                                                         secondaryTensor:rhs_tensor
                                                                    name:nil];
        MN_CHECK(result != nil,
                 MetalNativeError::InternalError,
                 "GraphBuilder: add operation failed");

        impl_->recorded_ops.push_back({OpType::Add, (__bridge void*)(result)});
        return (__bridge void*)(result);
    }
}

void* GraphBuilder::add_mul(void* lhs, void* rhs) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    @autoreleasepool {
        MPSGraphTensor* lhs_tensor = impl_->get_tensor(lhs);
        MPSGraphTensor* rhs_tensor = impl_->get_tensor(rhs);

        MPSGraphTensor* result = [impl_->graph multiplicationWithPrimaryTensor:lhs_tensor
                                                               secondaryTensor:rhs_tensor
                                                                          name:nil];
        MN_CHECK(result != nil,
                 MetalNativeError::InternalError,
                 "GraphBuilder: mul operation failed");

        impl_->recorded_ops.push_back({OpType::Mul, (__bridge void*)(result)});
        return (__bridge void*)(result);
    }
}

void GraphBuilder::mark_output(const std::string& name) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    auto it = impl_->tensor_map.find(name);
    MN_CHECK(it != impl_->tensor_map.end(),
             MetalNativeError::InvalidArgument,
             "GraphBuilder: tensor '" + name + "' not found");

    [impl_->output_tensors addObject:it->second];
}

void GraphBuilder::record_op(OpType type, void* result_handle) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->recorded_ops.push_back({type, result_handle});
}

size_t GraphBuilder::optimize() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->optimize_locked();
}

GraphExecutable GraphBuilder::build() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    // Run fusion optimization pass
    impl_->optimize_locked();

    MN_CHECK(impl_->output_tensors.count > 0,
             MetalNativeError::InvalidArgument,
             "GraphBuilder: no output tensors marked");

    @autoreleasepool {
        // Create the executable
        id<MTLDevice> metal_device = impl_->device.metal_device();

        MPSGraphExecutionDescriptor* exec_desc = [[MPSGraphExecutionDescriptor alloc] init];
        exec_desc.waitUntilCompleted = NO;

        // Compile the graph
        MPSGraphExecutable* executable = [impl_->graph compileWithDevice:metal_device
                                                                   feeds:@{}
                                                           targetTensors:impl_->output_tensors
                                                        targetOperations:nil
                                                       compilationDescriptor:nil];

        MN_CHECK(executable != nil,
                 MetalNativeError::InternalError,
                 "GraphBuilder: failed to compile graph");

        return GraphExecutable(executable);
    }
}

} // namespace metal_native
