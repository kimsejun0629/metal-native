#pragma once

/// @file batch_advisor.h
/// @brief Advisory API for optimal batch size based on available memory.

#include "metal_native/core/dtype.h"
#include <cstddef>
#include <cstdint>

namespace metal_native {

struct BatchSizeRecommendation {
    size_t recommended_batch;     // Optimal batch size for current conditions
    size_t max_batch;             // Maximum batch that would fit in memory
    size_t min_batch;             // Minimum useful batch (always 1)
    size_t memory_per_sample;     // Estimated bytes per sample
    float  utilization;           // Expected GPU memory utilization (0.0-1.0)
};

class BatchSizeAdvisor {
public:
    /// Recommend batch size given model parameters and current memory state.
    static BatchSizeRecommendation recommend(
        size_t model_bytes,
        size_t seq_len,
        size_t hidden_dim,
        MNDType dtype);
};

} // namespace metal_native
