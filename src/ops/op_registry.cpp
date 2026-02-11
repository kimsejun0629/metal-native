/// @file op_registry.cpp
/// @brief Implementation of the operator registry.

#include "metal_native/ops/op_registry.h"
#include "metal_native/core/error.h"
#include "metal_native/core/tensor.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace metal_native {

// ---------------------------------------------------------------------------
// OpRegistry::Impl
// ---------------------------------------------------------------------------

struct OpEntry {
    OpSchema schema;
    ForwardFn forward_fn;
    BackwardFn backward_fn;
};

struct OpRegistry::Impl {
    std::unordered_map<std::string, OpEntry> ops;
    mutable std::mutex mu;
};

// ---------------------------------------------------------------------------
// OpRegistry public API
// ---------------------------------------------------------------------------

OpRegistry::OpRegistry() : impl_(std::make_unique<Impl>()) {}

OpRegistry::~OpRegistry() = default;

OpRegistry& OpRegistry::instance() {
    static OpRegistry registry;
    return registry;
}

void OpRegistry::register_op(const std::string& name,
                              const OpSchema& schema,
                              ForwardFn forward_fn,
                              BackwardFn backward_fn) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    MN_CHECK(impl_->ops.find(name) == impl_->ops.end(),
             MetalNativeError::InvalidArgument,
             "OpRegistry: operator '" + name + "' is already registered");

    MN_CHECK(forward_fn != nullptr,
             MetalNativeError::InvalidArgument,
             "OpRegistry: forward_fn cannot be null for operator '" + name + "'");

    OpEntry entry;
    entry.schema = schema;
    entry.forward_fn = forward_fn;
    entry.backward_fn = backward_fn;

    impl_->ops[name] = entry;
}

MNTensor OpRegistry::dispatch(const std::string& name,
                              const std::vector<MNTensor>& inputs) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    auto it = impl_->ops.find(name);
    MN_CHECK(it != impl_->ops.end(),
             MetalNativeError::InvalidArgument,
             "OpRegistry: operator '" + name + "' is not registered");

    return it->second.forward_fn(inputs);
}

std::vector<MNTensor> OpRegistry::dispatch_backward(
    const std::string& name,
    const std::vector<MNTensor>& grad_outputs) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    auto it = impl_->ops.find(name);
    MN_CHECK(it != impl_->ops.end(),
             MetalNativeError::InvalidArgument,
             "OpRegistry: operator '" + name + "' is not registered");

    MN_CHECK(it->second.backward_fn != nullptr,
             MetalNativeError::InvalidArgument,
             "OpRegistry: operator '" + name + "' does not support backward");

    return it->second.backward_fn(grad_outputs);
}

bool OpRegistry::has_op(const std::string& name) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->ops.find(name) != impl_->ops.end();
}

std::vector<std::string> OpRegistry::list_ops() const {
    std::lock_guard<std::mutex> lock(impl_->mu);

    std::vector<std::string> names;
    names.reserve(impl_->ops.size());

    for (const auto& kv : impl_->ops) {
        names.push_back(kv.first);
    }

    std::sort(names.begin(), names.end());
    return names;
}

const OpSchema& OpRegistry::get_schema(const std::string& name) const {
    std::lock_guard<std::mutex> lock(impl_->mu);

    auto it = impl_->ops.find(name);
    MN_CHECK(it != impl_->ops.end(),
             MetalNativeError::InvalidArgument,
             "OpRegistry: operator '" + name + "' is not registered");

    return it->second.schema;
}

} // namespace metal_native
