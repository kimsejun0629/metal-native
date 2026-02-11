#import <Foundation/Foundation.h>
#include "metal_native/memory/activation_cache.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/error.h"
#include "metal_native/memory/budget_controller.h"

#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

namespace metal_native {

struct ActivationCache::Impl {
    struct Entry {
        std::string key;
        MNTensor tensor;
        size_t byte_size;
    };

    mutable std::mutex mu;
    size_t max_entries;
    std::list<Entry> order;  // front = most recently used
    std::unordered_map<std::string, std::list<Entry>::iterator> map;
};

ActivationCache::ActivationCache(size_t max_entries)
    : impl_(std::make_unique<Impl>()) {
    impl_->max_entries = max_entries;
}

ActivationCache::~ActivationCache() = default;

void ActivationCache::store(const std::string& key, const MNTensor& activation) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    // If key exists, move to front and update
    auto it = impl_->map.find(key);
    if (it != impl_->map.end()) {
        impl_->order.erase(it->second);
        impl_->map.erase(it);
    }

    // Check budget-aware capacity
    auto& bc = MemoryBudgetController::instance();
    float mult = bc.pressure_multiplier();
    size_t effective_max = std::max<size_t>(2, static_cast<size_t>(impl_->max_entries * mult));

    // Evict LRU entries if at capacity
    while (impl_->order.size() >= effective_max) {
        auto& back = impl_->order.back();
        impl_->map.erase(back.key);
        impl_->order.pop_back();
    }

    // Calculate byte size: numel * dtype_size
    size_t dtype_size = 4; // default FP32
    if (activation.dtype() == MNDType::Float16) dtype_size = 2;
    size_t byte_size = activation.numel() * dtype_size;

    // Insert at front
    impl_->order.push_front({key, activation, byte_size});
    impl_->map[key] = impl_->order.begin();
}

std::optional<MNTensor> ActivationCache::lookup(const std::string& key) const {
    std::lock_guard<std::mutex> lock(impl_->mu);

    auto it = impl_->map.find(key);
    if (it == impl_->map.end()) {
        return std::nullopt;
    }

    // Move to front (promote)
    auto list_it = it->second;
    if (list_it != impl_->order.begin()) {
        impl_->order.splice(impl_->order.begin(), impl_->order, list_it);
    }

    return list_it->tensor;
}

void ActivationCache::evict_to_budget(size_t max_bytes) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    size_t total = 0;
    for (auto& e : impl_->order) {
        total += e.byte_size;
    }

    while (total > max_bytes && !impl_->order.empty()) {
        auto& back = impl_->order.back();
        total -= back.byte_size;
        impl_->map.erase(back.key);
        impl_->order.pop_back();
    }
}

void ActivationCache::clear() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->order.clear();
    impl_->map.clear();
}

size_t ActivationCache::memory_footprint() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    size_t total = 0;
    for (auto& e : impl_->order) {
        total += e.byte_size;
    }
    return total;
}

size_t ActivationCache::size() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->order.size();
}

} // namespace metal_native
