#pragma once

/// @file speculative_buffers.h
/// @brief Pre-allocated buffer pool for speculative decoding.

#include "metal_native/core/dtype.h"
#include <cstddef>
#include <cstdint>
#include <memory>

namespace metal_native {

class MNDevice;
class MNTensor;

struct SpeculativeConfig {
    size_t max_draft_tokens;     // Maximum speculative draft tokens (e.g. 5)
    size_t vocab_size;
    size_t hidden_dim;
    MNDType dtype = MNDType::Float16;
};

class SpeculativeDecodingBufferManager {
public:
    explicit SpeculativeDecodingBufferManager(const SpeculativeConfig& config, MNDevice& device);
    ~SpeculativeDecodingBufferManager();

    /// Get pre-allocated buffer for draft logits: [max_draft_tokens, vocab_size]
    MNTensor draft_logits_buffer();

    /// Get pre-allocated buffer for verification logits: [max_draft_tokens + 1, vocab_size]
    MNTensor verification_logits_buffer();

    /// Get pre-allocated buffer for draft tokens: [max_draft_tokens]
    MNTensor draft_tokens_buffer();

    /// Resize under pressure (reduce max_draft_tokens)
    void resize(size_t new_max_draft_tokens);

    /// Memory footprint
    size_t memory_footprint() const;

    SpeculativeDecodingBufferManager(const SpeculativeDecodingBufferManager&) = delete;
    SpeculativeDecodingBufferManager& operator=(const SpeculativeDecodingBufferManager&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
