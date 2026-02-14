#pragma once

/// @file lazy_graph.h
/// @brief Lazy evaluation framework for deferred computation.
///
/// LazyGraph records operations without executing them, building a DAG
/// of deferred computations. When a value is needed (e.g., tensor_to_numpy,
/// item(), synchronize()), the graph is materialized by traversing the DAG,
/// applying optimizations (fusion, constant folding), and dispatching to GPU.
///
/// Inspired by MLX's lazy evaluation architecture.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "metal_native/core/dtype.h"
#include "metal_native/core/shape.h"
#include "metal_native/graph/fusion_patterns.h"

namespace metal_native {

class MNTensor;
class MNDevice;

/// Unique identifier for a node in the lazy graph.
using LazyNodeId = uint64_t;

/// Status of a lazy node.
enum class LazyNodeStatus : uint8_t {
    Pending,      ///< Not yet evaluated.
    Scheduled,    ///< Scheduled for evaluation.
    Materialized, ///< Evaluation complete, result available.
};

/// A node in the lazy computation graph.
struct LazyNode {
    LazyNodeId id = 0;
    OpType op_type = OpType::Unknown;
    LazyNodeStatus status = LazyNodeStatus::Pending;

    /// Output shape (computed eagerly for shape inference).
    MNShape output_shape;
    MNDType output_dtype = MNDType::Float32;

    /// Input node IDs (edges in the DAG).
    std::vector<LazyNodeId> inputs;

    /// Scalar parameters (e.g., epsilon for layer_norm).
    std::vector<double> scalar_params;

    /// Materialized result (set after evaluation).
    std::shared_ptr<MNTensor> result;

    /// Reference count (how many downstream nodes depend on this).
    uint32_t ref_count = 0;

    /// Whether this node is marked as an output (protected from DCE).
    bool is_output = false;
};

/// Lazy evaluation graph that records and optimizes computations.
class LazyGraph {
public:
    /// Access the global lazy graph (thread-local for safety).
    static LazyGraph& instance();

    // -- Recording operations ------------------------------------------------

    /// Record a unary operation (relu, gelu, softmax, etc.).
    LazyNodeId record_unary(OpType op, LazyNodeId input,
                            const MNShape& output_shape, MNDType dtype);

    /// Record a binary operation (add, mul, matmul, etc.).
    LazyNodeId record_binary(OpType op, LazyNodeId lhs, LazyNodeId rhs,
                             const MNShape& output_shape, MNDType dtype);

    /// Record a constant/input tensor (already materialized).
    LazyNodeId record_input(std::shared_ptr<MNTensor> tensor);

    // -- Evaluation ----------------------------------------------------------

    /// Materialize a specific node and all its dependencies.
    /// Returns the materialized tensor.
    MNTensor& eval(LazyNodeId node_id);

    /// Materialize all pending nodes.
    void eval_all();

    /// Check if a node has been materialized.
    bool is_materialized(LazyNodeId node_id) const;

    /// Mark a node as an output (protects it from dead code elimination).
    void mark_output(LazyNodeId node_id);

    /// Unmark a node as output.
    void unmark_output(LazyNodeId node_id);

    // -- Optimization passes -------------------------------------------------

    /// Run fusion analysis on pending nodes.
    /// @return Number of fusion opportunities found.
    size_t optimize_fusions();

    /// Run dead code elimination (remove nodes with zero ref_count
    /// that are not output nodes).
    size_t eliminate_dead_code();

    // -- Graph management ----------------------------------------------------

    /// Clear all nodes from the graph.
    void reset();

    /// Number of nodes in the graph.
    size_t node_count() const;

    /// Number of pending (unevaluated) nodes.
    size_t pending_count() const;

    /// Enable/disable lazy evaluation globally.
    static void set_enabled(bool enabled);
    static bool is_enabled();

private:
    LazyGraph();
    ~LazyGraph();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
