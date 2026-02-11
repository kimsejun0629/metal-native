/// @file fusion_patterns.mm
/// @brief Objective-C++ implementation of FusionRegistry.

#import <Foundation/Foundation.h>

#include "metal_native/graph/fusion_patterns.h"

#include <algorithm>
#include <mutex>
#include <vector>

namespace metal_native {

// ---------------------------------------------------------------------------
// OpType helpers
// ---------------------------------------------------------------------------

const char* op_type_name(OpType type) noexcept {
    switch (type) {
        case OpType::Unknown:   return "Unknown";
        case OpType::Conv2D:    return "Conv2D";
        case OpType::BatchNorm: return "BatchNorm";
        case OpType::ReLU:      return "ReLU";
        case OpType::GELU:      return "GELU";
        case OpType::Softmax:   return "Softmax";
        case OpType::MatMul:    return "MatMul";
        case OpType::Linear:    return "Linear";
        case OpType::BiasAdd:   return "BiasAdd";
        case OpType::LayerNorm: return "LayerNorm";
        case OpType::Add:       return "Add";
        case OpType::Mul:       return "Mul";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// FusionRegistry::Impl
// ---------------------------------------------------------------------------

struct FusionRegistry::Impl {
    std::vector<FusionPattern> patterns;
    mutable std::mutex mu;

    Impl() = default;

    // -- Pattern matching helpers --------------------------------------------

    bool matches_at(const std::vector<OpType>& ops,
                    size_t start,
                    const FusionPattern& pattern) const {
        if (start + pattern.sequence.size() > ops.size()) {
            return false;
        }

        for (size_t i = 0; i < pattern.sequence.size(); ++i) {
            if (ops[start + i] != pattern.sequence[i]) {
                return false;
            }
        }

        return true;
    }
};

// ---------------------------------------------------------------------------
// FusionRegistry public API
// ---------------------------------------------------------------------------

FusionRegistry& FusionRegistry::instance() {
    static FusionRegistry registry;
    return registry;
}

FusionRegistry::FusionRegistry() : impl_(std::make_unique<Impl>()) {
    // Automatically register built-in patterns on first use.
    register_builtin_patterns();
}

FusionRegistry::~FusionRegistry() = default;

void FusionRegistry::register_pattern(FusionPattern pattern) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->patterns.push_back(std::move(pattern));
}

void FusionRegistry::clear_patterns() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->patterns.clear();
}

FusionMatch FusionRegistry::try_fuse(const std::vector<OpType>& ops,
                                     size_t start_index) const {
    std::lock_guard<std::mutex> lock(impl_->mu);

    if (start_index >= ops.size()) {
        return FusionMatch{};
    }

    // Try patterns in order (longest first, typically).
    for (const auto& pattern : impl_->patterns) {
        if (impl_->matches_at(ops, start_index, pattern)) {
            FusionMatch match;
            match.matched = true;
            match.start_index = start_index;
            match.length = pattern.sequence.size();
            match.pattern = &pattern;
            return match;
        }
    }

    return FusionMatch{};
}

std::vector<FusionMatch> FusionRegistry::find_all_fusions(
        const std::vector<OpType>& ops) const {
    std::vector<FusionMatch> matches;

    size_t i = 0;
    while (i < ops.size()) {
        FusionMatch match = try_fuse(ops, i);
        if (match.matched) {
            matches.push_back(match);
            i += match.length; // Skip over the fused sequence.
        } else {
            ++i; // Move to the next operation.
        }
    }

    return matches;
}

void FusionRegistry::register_builtin_patterns() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    // Clear existing patterns to avoid duplicates on repeated calls.
    impl_->patterns.clear();

    // Pattern 1: Conv2D + BatchNorm + ReLU
    // This is a very common pattern in CNNs (ResNet, MobileNet, etc.).
    impl_->patterns.emplace_back(
        std::vector<OpType>{OpType::Conv2D, OpType::BatchNorm, OpType::ReLU},
        "FusedConvBNReLU",
        1.3f  // ~30% speedup
    );

    // Pattern 2: Conv2D + ReLU
    // Simpler variant without batch normalization.
    impl_->patterns.emplace_back(
        std::vector<OpType>{OpType::Conv2D, OpType::ReLU},
        "FusedConvReLU",
        1.2f  // ~20% speedup
    );

    // Pattern 3: Linear + GELU
    // Common in transformers (BERT, GPT).
    impl_->patterns.emplace_back(
        std::vector<OpType>{OpType::Linear, OpType::GELU},
        "FusedLinearGELU",
        1.25f  // ~25% speedup
    );

    // Pattern 4: MatMul + BiasAdd + ReLU
    // General GEMM fusion with bias and activation.
    impl_->patterns.emplace_back(
        std::vector<OpType>{OpType::MatMul, OpType::BiasAdd, OpType::ReLU},
        "FusedGEMMBiasReLU",
        1.3f  // ~30% speedup
    );

    // Pattern 5: MatMul + BiasAdd + GELU
    // Transformer FFN layers.
    impl_->patterns.emplace_back(
        std::vector<OpType>{OpType::MatMul, OpType::BiasAdd, OpType::GELU},
        "FusedGEMMBiasGELU",
        1.3f  // ~30% speedup
    );

    // Pattern 6: MatMul + BiasAdd
    // Basic GEMM with bias.
    impl_->patterns.emplace_back(
        std::vector<OpType>{OpType::MatMul, OpType::BiasAdd},
        "FusedGEMMBias",
        1.15f  // ~15% speedup
    );

    // Pattern 7: Conv2D + BatchNorm
    // Without activation (used in some architectures).
    impl_->patterns.emplace_back(
        std::vector<OpType>{OpType::Conv2D, OpType::BatchNorm},
        "FusedConvBN",
        1.2f  // ~20% speedup
    );

    // Patterns are matched in the order registered. Longer patterns are
    // registered first to ensure greedy matching picks the most specific
    // fusion opportunity.
}

size_t FusionRegistry::pattern_count() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->patterns.size();
}

} // namespace metal_native
