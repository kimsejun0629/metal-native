#pragma once

/// @file speculative.h
/// @brief Speculative decoding verification kernel.

#include <cstdint>
#include <vector>
#include "metal_native/core/tensor.h"

namespace metal_native {

/// Result of speculative decoding verification.
struct SpeculativeResult {
    /// Per-position acceptance mask [batch, K]. 1 = accepted, 0 = rejected.
    MNTensor accepted;
    /// Number of accepted tokens per batch item [batch].
    MNTensor num_accepted;
};

/// Verify draft tokens against target model probabilities.
///
/// @param p_target     Target model probabilities [batch, K, vocab_size].
/// @param p_draft      Draft model probabilities [batch, K, vocab_size].
/// @param draft_tokens Draft token indices [batch, K].
/// @param random_vals  Pre-generated uniform random values [batch, K].
/// @return             SpeculativeResult with acceptance mask and count.
SpeculativeResult speculative_verify(
    const MNTensor& p_target,
    const MNTensor& p_draft,
    const MNTensor& draft_tokens,
    const MNTensor& random_vals);

} // namespace metal_native
