#pragma once

/// @file graph_builder.h
/// @brief Fluent API for constructing MPSGraph computation graphs.
///
/// GraphBuilder provides a high-level interface for building MPSGraph instances
/// with automatic dtype propagation. It supports common operations (matmul,
/// convolution, activations, layer norm, etc.) and produces an executable
/// graph that can be cached and reused across multiple invocations.
///
/// Example usage:
/// @code
///   GraphBuilder builder(device);
///   auto x = builder.add_placeholder("x", {1, 256}, MNDType::Float32);
///   auto w = builder.add_placeholder("w", {256, 512}, MNDType::Float32);
///   auto y = builder.add_matmul(x, w);
///   auto z = builder.add_relu(y);
///   builder.mark_output("z");
///   auto executable = builder.build();
/// @endcode
///
/// Thread-safe: all public methods are guarded by a mutex.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "metal_native/core/dtype.h"
#include "metal_native/core/shape.h"
#include "metal_native/graph/fusion_patterns.h"

#ifdef __OBJC__
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#endif

namespace metal_native {

class MNDevice;

/// Wrapper around MPSGraphExecutable with RAII semantics.
class GraphExecutable {
public:
    GraphExecutable();
    ~GraphExecutable();

    // Non-copyable, movable.
    GraphExecutable(const GraphExecutable&) = delete;
    GraphExecutable& operator=(const GraphExecutable&) = delete;
    GraphExecutable(GraphExecutable&&) noexcept;
    GraphExecutable& operator=(GraphExecutable&&) noexcept;

#ifdef __OBJC__
    /// Access the underlying MPSGraphExecutable.
    MPSGraphExecutable* executable() const noexcept;

    /// Initialize from an existing MPSGraphExecutable.
    explicit GraphExecutable(MPSGraphExecutable* exec);
#else
    /// Opaque handle to MPSGraphExecutable.
    void* executable() const noexcept;

    /// Initialize from an opaque handle.
    explicit GraphExecutable(void* exec);
#endif

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Fluent builder for MPSGraph computation graphs.
class GraphBuilder {
public:
    /// Construct a builder that will create graphs for the given device.
    explicit GraphBuilder(MNDevice& device);
    ~GraphBuilder();

    // Non-copyable, non-movable.
    GraphBuilder(const GraphBuilder&) = delete;
    GraphBuilder& operator=(const GraphBuilder&) = delete;
    GraphBuilder(GraphBuilder&&) = delete;
    GraphBuilder& operator=(GraphBuilder&&) = delete;

    // -- Placeholder tensors -------------------------------------------------

    /// Add a placeholder (input) tensor to the graph.
    ///
    /// @param name   Unique identifier for this tensor.
    /// @param shape  Shape of the tensor.
    /// @param dtype  Data type of the tensor.
    /// @return       Opaque tensor handle (MPSGraphTensor* in .mm files).
    void* add_placeholder(const std::string& name,
                          const MNShape& shape,
                          MNDType dtype);

    // -- Linear algebra operations -------------------------------------------

    /// Add a matrix multiplication: C = A @ B.
    ///
    /// @param lhs    Left operand tensor.
    /// @param rhs    Right operand tensor.
    /// @return       Result tensor handle.
    void* add_matmul(void* lhs, void* rhs);

    /// Add a 2D convolution operation.
    ///
    /// @param input         Input tensor (NCHW or NHWC).
    /// @param weight        Weight tensor (OIHW).
    /// @param stride_h      Vertical stride.
    /// @param stride_w      Horizontal stride.
    /// @param padding_h     Vertical padding.
    /// @param padding_w     Horizontal padding.
    /// @param dilation_h    Vertical dilation.
    /// @param dilation_w    Horizontal dilation.
    /// @return              Result tensor handle.
    void* add_conv2d(void* input,
                     void* weight,
                     size_t stride_h = 1,
                     size_t stride_w = 1,
                     size_t padding_h = 0,
                     size_t padding_w = 0,
                     size_t dilation_h = 1,
                     size_t dilation_w = 1);

    // -- Activation functions ------------------------------------------------

    /// Add a ReLU activation: y = max(0, x).
    void* add_relu(void* input);

    /// Add a GELU activation (Gaussian Error Linear Unit).
    void* add_gelu(void* input);

    /// Add a softmax operation along the last dimension.
    void* add_softmax(void* input);

    // -- Normalization -------------------------------------------------------

    /// Add a layer normalization operation.
    ///
    /// @param input   Input tensor.
    /// @param axes    Axes to normalize over (e.g., {-1} for last dimension).
    /// @param eps     Small constant for numerical stability.
    /// @return        Result tensor handle.
    void* add_layer_norm(void* input,
                         const std::vector<int64_t>& axes,
                         float eps = 1e-5f);

    // -- Element-wise operations ---------------------------------------------

    /// Add an element-wise addition: z = x + y.
    void* add_add(void* lhs, void* rhs);

    /// Add an element-wise multiplication: z = x * y.
    void* add_mul(void* lhs, void* rhs);

    // -- Output marking ------------------------------------------------------

    /// Mark a tensor as an output of the graph.
    ///
    /// @param name   The name assigned to the tensor (from add_placeholder
    ///               or internal naming).
    void mark_output(const std::string& name);

    // -- Graph compilation ---------------------------------------------------

    /// Compile the graph and return an executable.
    ///
    /// This finalizes the graph structure, compiles it for the target device,
    /// and returns a GraphExecutable that can be used for inference.
    ///
    /// @return  An executable graph wrapper.
    /// @throws  MNException on compilation failure.
    GraphExecutable build();

    // -- Operation recording (for fusion) ------------------------------------

    /// Record an operation for deferred fusion analysis.
    /// Call this instead of (or in addition to) the individual add_* methods
    /// when building a sequence of operations that might be fusible.
    void record_op(OpType type, void* result_handle);

    /// Analyze recorded operations and apply fusion where possible.
    /// This scans the recorded op sequence using FusionRegistry, identifies
    /// fusible patterns, and replaces them with optimized fused operations.
    /// Must be called before build().
    /// @return Number of fusions applied.
    size_t optimize();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
