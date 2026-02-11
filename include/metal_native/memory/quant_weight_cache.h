#pragma once

/// @file quant_weight_cache.h
/// @brief Cache for quantized weights with optional dequantized FP16 copies.

#include "metal_native/core/dtype.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace metal_native {

class MNTensor;
class MNDevice;

enum class QuantFormat : uint8_t {
    INT8 = 0,
    INT4 = 1
};

class QuantWeightCache {
public:
    explicit QuantWeightCache(MNDevice& device);
    ~QuantWeightCache();

    /// Register a weight with its quantized representation
    void register_weight(const std::string& name, const MNTensor& quantized,
                         const MNTensor& scales, QuantFormat format);

    /// Get dequantized weight (returns cached FP16 or dequantizes on-the-fly)
    MNTensor get_dequantized(const std::string& name);

    /// Evict dequantized copies under pressure (keeps quantized)
    void evict_dequantized();

    /// Memory footprint of dequantized copies only
    size_t dequantized_footprint() const;

    /// Total memory footprint
    size_t total_footprint() const;

    QuantWeightCache(const QuantWeightCache&) = delete;
    QuantWeightCache& operator=(const QuantWeightCache&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
