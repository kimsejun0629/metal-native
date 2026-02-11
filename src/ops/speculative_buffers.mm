/// @file speculative_buffers.mm
/// @brief Implementation of speculative decoding buffer manager.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/ops/speculative_buffers.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/memory/budget_controller.h"
#include <algorithm>
#include <optional>

namespace metal_native {

struct SpeculativeDecodingBufferManager::Impl {
    MNDevice& device;
    SpeculativeConfig config;
    size_t current_max_draft;

    std::optional<MNTensor> draft_logits;
    std::optional<MNTensor> verify_logits;
    std::optional<MNTensor> draft_tokens;

    explicit Impl(MNDevice& dev) : device(dev), current_max_draft(0) {}

    void allocate_buffers() {
        // Draft logits: [max_draft_tokens, vocab_size]
        draft_logits = MNTensor::empty(
            MNShape({static_cast<int64_t>(current_max_draft),
                     static_cast<int64_t>(config.vocab_size)}),
            config.dtype, device);

        // Verification logits: [max_draft_tokens + 1, vocab_size]
        verify_logits = MNTensor::empty(
            MNShape({static_cast<int64_t>(current_max_draft + 1),
                     static_cast<int64_t>(config.vocab_size)}),
            config.dtype, device);

        // Draft tokens: [max_draft_tokens] as Int64
        draft_tokens = MNTensor::empty(
            MNShape({static_cast<int64_t>(current_max_draft)}),
            MNDType::Int64, device);
    }
};

SpeculativeDecodingBufferManager::SpeculativeDecodingBufferManager(
    const SpeculativeConfig& config, MNDevice& device)
    : impl_(std::make_unique<Impl>(device)) {

    impl_->config = config;

    // Query budget
    size_t dtype_size = (config.dtype == MNDType::Float16) ? 2 : 4;
    size_t total_bytes = config.max_draft_tokens * config.vocab_size * dtype_size * 2  // draft + verify
                       + (config.max_draft_tokens + 1) * config.vocab_size * dtype_size
                       + config.max_draft_tokens * 8;  // Int64 tokens

    auto budget = MemoryBudgetController::instance().request_budget(
        BudgetStrategy::SpeculativeBuffers, total_bytes);

    if (budget.approved && budget.bytes >= total_bytes) {
        impl_->current_max_draft = config.max_draft_tokens;
    } else if (budget.approved) {
        // Reduce proportionally
        double ratio = static_cast<double>(budget.bytes) / static_cast<double>(total_bytes);
        impl_->current_max_draft = std::max<size_t>(1,
            static_cast<size_t>(config.max_draft_tokens * ratio));
    } else {
        impl_->current_max_draft = 1;  // minimum
    }

    impl_->allocate_buffers();
}

SpeculativeDecodingBufferManager::~SpeculativeDecodingBufferManager() = default;

MNTensor SpeculativeDecodingBufferManager::draft_logits_buffer() {
    return *impl_->draft_logits;
}

MNTensor SpeculativeDecodingBufferManager::verification_logits_buffer() {
    return *impl_->verify_logits;
}

MNTensor SpeculativeDecodingBufferManager::draft_tokens_buffer() {
    return *impl_->draft_tokens;
}

void SpeculativeDecodingBufferManager::resize(size_t new_max_draft_tokens) {
    if (new_max_draft_tokens >= impl_->current_max_draft) return;
    new_max_draft_tokens = std::max<size_t>(1, new_max_draft_tokens);
    impl_->current_max_draft = new_max_draft_tokens;
    impl_->allocate_buffers();
}

size_t SpeculativeDecodingBufferManager::memory_footprint() const {
    size_t dtype_size = (impl_->config.dtype == MNDType::Float16) ? 2 : 4;
    size_t d = impl_->current_max_draft;
    size_t v = impl_->config.vocab_size;
    return d * v * dtype_size                // draft logits
         + (d + 1) * v * dtype_size          // verify logits
         + d * 8;                             // draft tokens (Int64)
}

} // namespace metal_native
