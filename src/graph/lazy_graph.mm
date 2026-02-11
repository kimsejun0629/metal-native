/// @file lazy_graph.mm
/// @brief Implementation of lazy evaluation graph.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/graph/lazy_graph.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/graph/fusion_patterns.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

namespace metal_native {

// Global enable flag
static std::atomic<bool> g_lazy_enabled{false};

struct LazyGraph::Impl {
    std::unordered_map<LazyNodeId, LazyNode> nodes;
    LazyNodeId next_id = 1;
    mutable std::mutex mu;

    LazyNodeId allocate_id() {
        return next_id++;
    }
};

// Thread-local instance for safety
LazyGraph& LazyGraph::instance() {
    thread_local LazyGraph graph;
    return graph;
}

LazyGraph::LazyGraph() : impl_(std::make_unique<Impl>()) {}
LazyGraph::~LazyGraph() = default;

void LazyGraph::set_enabled(bool enabled) {
    g_lazy_enabled.store(enabled, std::memory_order_release);
}

bool LazyGraph::is_enabled() {
    return g_lazy_enabled.load(std::memory_order_acquire);
}

LazyNodeId LazyGraph::record_unary(OpType op, LazyNodeId input,
                                    const MNShape& output_shape, MNDType dtype) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    LazyNode node;
    node.id = impl_->allocate_id();
    node.op_type = op;
    node.output_shape = output_shape;
    node.output_dtype = dtype;
    node.inputs = {input};

    // Increment ref count of input
    auto it = impl_->nodes.find(input);
    if (it != impl_->nodes.end()) {
        it->second.ref_count++;
    }

    LazyNodeId id = node.id;
    impl_->nodes[id] = std::move(node);
    return id;
}

LazyNodeId LazyGraph::record_binary(OpType op, LazyNodeId lhs, LazyNodeId rhs,
                                     const MNShape& output_shape, MNDType dtype) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    LazyNode node;
    node.id = impl_->allocate_id();
    node.op_type = op;
    node.output_shape = output_shape;
    node.output_dtype = dtype;
    node.inputs = {lhs, rhs};

    // Increment ref counts
    for (LazyNodeId inp : {lhs, rhs}) {
        auto it = impl_->nodes.find(inp);
        if (it != impl_->nodes.end()) {
            it->second.ref_count++;
        }
    }

    LazyNodeId id = node.id;
    impl_->nodes[id] = std::move(node);
    return id;
}

LazyNodeId LazyGraph::record_input(std::shared_ptr<MNTensor> tensor) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    LazyNode node;
    node.id = impl_->allocate_id();
    node.op_type = OpType::Unknown;  // Input nodes have no op
    node.status = LazyNodeStatus::Materialized;
    node.output_shape = tensor->shape();
    node.output_dtype = tensor->dtype();
    node.result = std::move(tensor);

    LazyNodeId id = node.id;
    impl_->nodes[id] = std::move(node);
    return id;
}

MNTensor& LazyGraph::eval(LazyNodeId node_id) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    auto it = impl_->nodes.find(node_id);
    MN_CHECK(it != impl_->nodes.end(),
             MetalNativeError::InvalidArgument,
             "LazyGraph::eval: node not found");

    if (it->second.status == LazyNodeStatus::Materialized) {
        return *(it->second.result);
    }

    // TODO: Full implementation would:
    // 1. Topological sort from this node back to inputs
    // 2. Run fusion optimization on the subgraph
    // 3. Build MPSGraph from the optimized subgraph
    // 4. Execute and store results
    // For now, throw - operations still execute eagerly
    MN_THROW(MetalNativeError::NotImplemented,
             "LazyGraph::eval: deferred evaluation not yet implemented. "
             "Use LazyGraph::set_enabled(false) for eager mode.");
}

void LazyGraph::eval_all() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    // Find all leaf nodes (nodes with ref_count == 0 that aren't materialized)
    for (auto& [id, node] : impl_->nodes) {
        if (node.status == LazyNodeStatus::Pending && node.ref_count == 0) {
            // This is an output node - would need to evaluate
            // TODO: implement batch evaluation
        }
    }
}

bool LazyGraph::is_materialized(LazyNodeId node_id) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    auto it = impl_->nodes.find(node_id);
    if (it == impl_->nodes.end()) return false;
    return it->second.status == LazyNodeStatus::Materialized;
}

size_t LazyGraph::optimize_fusions() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    // Collect pending ops in topological order
    std::vector<OpType> op_sequence;
    std::vector<LazyNodeId> node_order;

    for (const auto& [id, node] : impl_->nodes) {
        if (node.status == LazyNodeStatus::Pending) {
            op_sequence.push_back(node.op_type);
            node_order.push_back(id);
        }
    }

    if (op_sequence.empty()) return 0;

    auto& registry = FusionRegistry::instance();
    auto matches = registry.find_all_fusions(op_sequence);

    // TODO: Rewrite fused sequences into single nodes
    return matches.size();
}

size_t LazyGraph::eliminate_dead_code() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    size_t eliminated = 0;
    std::vector<LazyNodeId> to_remove;

    for (const auto& [id, node] : impl_->nodes) {
        if (node.status == LazyNodeStatus::Pending && node.ref_count == 0) {
            // Check if this is truly dead (not an output)
            // For now, skip - need output marking to determine this
        }
    }

    for (LazyNodeId id : to_remove) {
        // Decrement ref counts of inputs before removing
        auto it = impl_->nodes.find(id);
        if (it != impl_->nodes.end()) {
            for (LazyNodeId inp : it->second.inputs) {
                auto inp_it = impl_->nodes.find(inp);
                if (inp_it != impl_->nodes.end() && inp_it->second.ref_count > 0) {
                    inp_it->second.ref_count--;
                }
            }
            impl_->nodes.erase(it);
            eliminated++;
        }
    }

    return eliminated;
}

void LazyGraph::reset() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->nodes.clear();
    impl_->next_id = 1;
}

size_t LazyGraph::node_count() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->nodes.size();
}

size_t LazyGraph::pending_count() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    size_t count = 0;
    for (const auto& [id, node] : impl_->nodes) {
        if (node.status == LazyNodeStatus::Pending) count++;
    }
    return count;
}

} // namespace metal_native
