#pragma once

/// @file kv_cache.h
/// @brief Pre-allocated KV cache for autoregressive generation.

#include "metal_native/core/dtype.h"
#include <cstddef>
#include <cstdint>
#include <memory>

namespace metal_native {

class MNDevice;
class MNTensor;

struct KVCacheConfig {
    size_t num_layers;
    size_t num_heads;
    size_t head_dim;
    size_t max_seq_len;
    MNDType dtype = MNDType::Float16;
};

class KVCache {
public:
    explicit KVCache(const KVCacheConfig& config, MNDevice& device);
    ~KVCache();

    /// Append new K/V for a given layer at a given position
    void append(size_t layer, const MNTensor& new_key, const MNTensor& new_value, size_t position);

    /// Get the K tensor for a layer (up to current_seq_len)
    MNTensor get_key(size_t layer) const;

    /// Get the V tensor for a layer (up to current_seq_len)
    MNTensor get_value(size_t layer) const;

    /// Current sequence length (number of appended tokens)
    size_t current_seq_len() const;

    /// Resize under memory pressure (shrinks max_seq_len, copies existing data)
    void resize(size_t new_max_seq_len);

    /// Reset for new generation (keeps buffers allocated)
    void reset();

    /// Memory footprint in bytes
    size_t memory_footprint() const;

    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
