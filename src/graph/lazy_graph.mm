/// @file lazy_graph.mm
/// @brief Implementation of lazy evaluation graph.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/graph/lazy_graph.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/graph/fusion_patterns.h"
#include "metal_native/ops/elementwise.h"
#include "metal_native/ops/normalization.h"
#include "metal_native/ops/softmax.h"
#include "metal_native/ops/matmul.h"
#include "metal_native/future/fast_ops.h"

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <cmath>

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

    std::shared_ptr<MNTensor> execute_node(LazyNode& node) {
        // Input nodes are already materialized
        if (node.op_type == OpType::Unknown) {
            return node.result;
        }

        // Gather materialized inputs
        std::vector<std::shared_ptr<MNTensor>> inputs;
        for (LazyNodeId inp_id : node.inputs) {
            auto it = nodes.find(inp_id);
            if (it == nodes.end() || it->second.status != LazyNodeStatus::Materialized) {
                return nullptr; // Input not ready
            }
            inputs.push_back(it->second.result);
        }

        // Get default device
        MNDevice& device = MNDevice::instance();

        // Dispatch based on op type
        switch (node.op_type) {
            case OpType::Add: {
                if (inputs.size() == 2) {
                    auto result = std::make_shared<MNTensor>(
                        add(*inputs[0], *inputs[1], device));
                    return result;
                }
                break;
            }
            case OpType::Mul: {
                if (inputs.size() == 2) {
                    auto result = std::make_shared<MNTensor>(
                        mul(*inputs[0], *inputs[1], device));
                    return result;
                }
                break;
            }
            case OpType::ReLU: {
                if (inputs.size() == 1) {
                    auto result = std::make_shared<MNTensor>(
                        clamp(*inputs[0], 0.0f, HUGE_VALF, device));
                    return result;
                }
                break;
            }
            case OpType::Softmax: {
                if (inputs.size() == 1) {
                    auto result = std::make_shared<MNTensor>(
                        softmax(*inputs[0], -1));
                    return result;
                }
                break;
            }
            case OpType::RMSNorm: {
                if (inputs.size() >= 2) {
                    float eps = (node.scalar_params.size() > 0) ?
                        static_cast<float>(node.scalar_params[0]) : 1e-6f;
                    auto result = std::make_shared<MNTensor>(
                        fast::rms_norm(*inputs[0], *inputs[1], eps));
                    return result;
                }
                break;
            }
            case OpType::LayerNorm: {
                if (inputs.size() >= 3) {
                    float eps = (node.scalar_params.size() > 0) ?
                        static_cast<float>(node.scalar_params[0]) : 1e-5f;
                    auto result = std::make_shared<MNTensor>(
                        fast::layer_norm(*inputs[0], *inputs[1], *inputs[2], eps));
                    return result;
                }
                break;
            }
            case OpType::SwiGLU: {
                if (inputs.size() == 2) {
                    auto result = std::make_shared<MNTensor>(
                        fast::swiglu(*inputs[0], *inputs[1]));
                    return result;
                }
                break;
            }
            case OpType::MatMul: {
                if (inputs.size() == 2) {
                    auto result = std::make_shared<MNTensor>(
                        matmul(*inputs[0], *inputs[1]));
                    return result;
                }
                break;
            }
            default:
                break;
        }

        return nullptr; // Unsupported op
    }

    std::vector<LazyNodeId> topological_sort(LazyNodeId target) {
        std::vector<LazyNodeId> order;
        std::unordered_set<LazyNodeId> visited;
        std::unordered_set<LazyNodeId> in_stack;  // cycle detection

        std::function<bool(LazyNodeId)> dfs = [&](LazyNodeId id) -> bool {
            if (visited.count(id)) return true;
            if (in_stack.count(id)) return false; // cycle!

            in_stack.insert(id);

            auto it = nodes.find(id);
            if (it == nodes.end()) return false;

            // Already materialized - no need to traverse further
            if (it->second.status == LazyNodeStatus::Materialized) {
                visited.insert(id);
                in_stack.erase(id);
                return true;
            }

            // Visit inputs first
            for (LazyNodeId inp : it->second.inputs) {
                if (!dfs(inp)) return false;
            }

            visited.insert(id);
            in_stack.erase(id);
            order.push_back(id);
            return true;
        };

        dfs(target);
        return order;
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

    // Already materialized - return directly
    if (it->second.status == LazyNodeStatus::Materialized) {
        return *(it->second.result);
    }

    // Phase A: Topological sort + sequential execution
    auto order = impl_->topological_sort(node_id);

    MN_CHECK(!order.empty(),
             MetalNativeError::InternalError,
             "LazyGraph::eval: topological sort produced empty order (cycle detected?)");

    // Execute each node in topological order
    for (LazyNodeId exec_id : order) {
        auto node_it = impl_->nodes.find(exec_id);
        if (node_it == impl_->nodes.end()) continue;
        if (node_it->second.status == LazyNodeStatus::Materialized) continue;

        node_it->second.status = LazyNodeStatus::Scheduled;

        auto result = impl_->execute_node(node_it->second);
        MN_CHECK(result != nullptr,
                 MetalNativeError::InternalError,
                 "LazyGraph::eval: failed to execute node " +
                 std::string(op_type_name(node_it->second.op_type)));

        node_it->second.result = std::move(result);
        node_it->second.status = LazyNodeStatus::Materialized;
    }

    // Return the target node's result
    it = impl_->nodes.find(node_id);
    MN_CHECK(it->second.status == LazyNodeStatus::Materialized,
             MetalNativeError::InternalError,
             "LazyGraph::eval: node not materialized after execution");

    return *(it->second.result);
}

void LazyGraph::eval_all() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    // Find all leaf nodes (nodes with ref_count == 0 that aren't materialized)
    std::vector<LazyNodeId> leaves;
    for (auto& [id, node] : impl_->nodes) {
        if (node.status == LazyNodeStatus::Pending && node.ref_count == 0) {
            leaves.push_back(id);
        }
    }

    // Evaluate each leaf (which will pull in all dependencies)
    for (LazyNodeId leaf_id : leaves) {
        auto order = impl_->topological_sort(leaf_id);

        for (LazyNodeId exec_id : order) {
            auto node_it = impl_->nodes.find(exec_id);
            if (node_it == impl_->nodes.end()) continue;
            if (node_it->second.status == LazyNodeStatus::Materialized) continue;

            node_it->second.status = LazyNodeStatus::Scheduled;
            auto result = impl_->execute_node(node_it->second);
            if (result) {
                node_it->second.result = std::move(result);
                node_it->second.status = LazyNodeStatus::Materialized;
            }
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
