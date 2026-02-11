#pragma once

/// @file op_registry.h
/// @brief Central dispatch registry for all MetalNative operators.
///
/// OpRegistry provides a global dispatch table that maps operator names to
/// their forward and backward implementations. All high-level ops (matmul,
/// conv, pooling, etc.) are registered here, enabling dynamic dispatch and
/// extensibility.
///
/// Example usage:
/// @code
///   auto& registry = OpRegistry::instance();
///
///   // Register an operator
///   OpSchema schema;
///   schema.name = "add";
///   schema.input_types = {MNDType::Float32, MNDType::Float32};
///   schema.output_types = {MNDType::Float32};
///   schema.supports_backward = true;
///
///   registry.register_op("add", schema, forward_fn, backward_fn);
///
///   // Dispatch an operator
///   std::vector<MNTensor> inputs = {a, b};
///   MNTensor result = registry.dispatch("add", inputs);
/// @endcode

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "metal_native/core/dtype.h"

namespace metal_native {

// Forward declaration
class MNTensor;

/// Schema describing an operator's signature and capabilities.
struct OpSchema {
    std::string name;
    std::vector<MNDType> input_types;
    std::vector<MNDType> output_types;
    bool supports_backward = false;
};

/// Function signature for operator forward pass.
using ForwardFn = std::function<MNTensor(const std::vector<MNTensor>&)>;

/// Function signature for operator backward pass.
using BackwardFn = std::function<std::vector<MNTensor>(const std::vector<MNTensor>&)>;

/// Central registry for all operators in the framework.
///
/// This class maintains a global dispatch table mapping operator names to
/// their implementations. Thread-safe singleton pattern.
class OpRegistry {
public:
    // -- Singleton access ----------------------------------------------------

    /// Return the process-wide OpRegistry instance.
    static OpRegistry& instance();

    // -- Registration --------------------------------------------------------

    /// Register an operator with its schema and implementation functions.
    ///
    /// @param name          Unique identifier for the operator.
    /// @param schema        Operator schema (types, backward support).
    /// @param forward_fn    Forward pass implementation.
    /// @param backward_fn   Optional backward pass implementation.
    /// @throws MNException(InvalidArgument) if @p name is already registered.
    void register_op(const std::string& name,
                     const OpSchema& schema,
                     ForwardFn forward_fn,
                     BackwardFn backward_fn = nullptr);

    // -- Dispatch ------------------------------------------------------------

    /// Dispatch an operator by name with the given inputs.
    ///
    /// @param name    Operator name (must be registered).
    /// @param inputs  Input tensors.
    /// @return        Result tensor from the forward pass.
    /// @throws MNException(InvalidArgument) if @p name is not found.
    MNTensor dispatch(const std::string& name,
                      const std::vector<MNTensor>& inputs);

    /// Dispatch backward pass for an operator.
    ///
    /// @param name           Operator name (must be registered).
    /// @param grad_outputs   Gradient outputs from downstream.
    /// @return               Gradient inputs for upstream.
    /// @throws MNException(InvalidArgument) if @p name is not found or
    ///         does not support backward.
    std::vector<MNTensor> dispatch_backward(const std::string& name,
                                            const std::vector<MNTensor>& grad_outputs);

    // -- Introspection -------------------------------------------------------

    /// Check whether an operator is registered.
    bool has_op(const std::string& name) const;

    /// List all registered operator names.
    std::vector<std::string> list_ops() const;

    /// Retrieve the schema for a registered operator.
    ///
    /// @param name  Operator name.
    /// @return      The operator's schema.
    /// @throws MNException(InvalidArgument) if @p name is not found.
    const OpSchema& get_schema(const std::string& name) const;

    // -- Non-copyable / non-movable ------------------------------------------
    OpRegistry(const OpRegistry&) = delete;
    OpRegistry& operator=(const OpRegistry&) = delete;
    OpRegistry(OpRegistry&&) = delete;
    OpRegistry& operator=(OpRegistry&&) = delete;

private:
    OpRegistry();
    ~OpRegistry();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
