/// @file batch_advisor.mm
/// @brief Implementation of batch size advisor.

#import <Foundation/Foundation.h>
#include "metal_native/ops/batch_advisor.h"
#include "metal_native/memory/budget_controller.h"
#include <algorithm>

namespace metal_native {

BatchSizeRecommendation BatchSizeAdvisor::recommend(
    size_t model_bytes,
    size_t seq_len,
    size_t hidden_dim,
    MNDType dtype) {

    // Get current memory headroom from budget controller
    BudgetSnapshot snap = MemoryBudgetController::instance().snapshot();
    size_t headroom = snap.distributable;

    // Calculate bytes per element
    size_t dtype_size = 4; // default Float32
    if (dtype == MNDType::Float16) dtype_size = 2;
    else if (dtype == MNDType::BFloat16) dtype_size = 2;

    // Activation memory per sample: ~4x hidden for KV + intermediate activations
    // seq_len * hidden_dim * dtype_size * 4
    size_t activation_per_sample = seq_len * hidden_dim * dtype_size * 4;

    BatchSizeRecommendation rec;
    rec.memory_per_sample = activation_per_sample;
    rec.min_batch = 1;

    if (activation_per_sample == 0) {
        rec.max_batch = 1;
        rec.recommended_batch = 1;
        rec.utilization = 0.0f;
        return rec;
    }

    rec.max_batch = headroom / activation_per_sample;
    if (rec.max_batch == 0) rec.max_batch = 1;

    // Target 75% of max to leave headroom
    rec.recommended_batch = std::max<size_t>(1, static_cast<size_t>(rec.max_batch * 0.75));

    // Expected utilization
    rec.utilization = static_cast<float>(rec.recommended_batch * activation_per_sample) /
                      static_cast<float>(headroom > 0 ? headroom : 1);

    return rec;
}

} // namespace metal_native
