#pragma once

/// @file activation_cache.h
/// @brief LRU cache for layer activations (inference prefix caching).

#include "metal_native/core/dtype.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace metal_native {

class MNTensor;

class ActivationCache {
public:
    explicit ActivationCache(size_t max_entries = 32);
    ~ActivationCache();

    /// Store a layer output keyed by (layer_id, input_hash)
    void store(const std::string& key, const MNTensor& activation);

    /// Lookup a cached activation. Returns std::nullopt if not found.
    std::optional<MNTensor> lookup(const std::string& key) const;

    /// Evict entries to fit within budget
    void evict_to_budget(size_t max_bytes);

    /// Evict all entries
    void clear();

    /// Current memory footprint
    size_t memory_footprint() const;

    /// Number of cached entries
    size_t size() const;

    ActivationCache(const ActivationCache&) = delete;
    ActivationCache& operator=(const ActivationCache&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
