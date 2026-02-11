#include <metal_stdlib>
using namespace metal;

/// Speculative decoding verification kernel.
/// For each draft token position i (0..K-1):
///   - Compute acceptance probability: min(1, p_target[i] / p_draft[i])
///   - Compare against random number to accept/reject
///   - All tokens after first rejection are rejected
///
/// @param p_target     Target model probabilities [batch, K, vocab_size]
/// @param p_draft      Draft model probabilities [batch, K, vocab_size]
/// @param draft_tokens Draft token indices [batch, K]
/// @param random_vals  Pre-generated uniform random values [batch, K]
/// @param accepted     Output: 1 if accepted, 0 if rejected [batch, K]
/// @param num_accepted Output: number of accepted tokens per batch [batch]
/// @param K            Number of draft tokens
/// @param vocab_size   Vocabulary size

kernel void speculative_verify_fp32(
    device const float*   p_target      [[buffer(0)]],
    device const float*   p_draft       [[buffer(1)]],
    device const int*     draft_tokens  [[buffer(2)]],
    device const float*   random_vals   [[buffer(3)]],
    device int*           accepted      [[buffer(4)]],
    device int*           num_accepted  [[buffer(5)]],
    constant uint&        K             [[buffer(6)]],
    constant uint&        vocab_size    [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint batch_idx = tid;

    uint count = 0;
    bool still_accepting = true;

    for (uint i = 0; i < K; i++) {
        if (!still_accepting) {
            accepted[batch_idx * K + i] = 0;
            continue;
        }

        int token = draft_tokens[batch_idx * K + i];

        // Get probabilities for the drafted token
        float p_t = p_target[batch_idx * K * vocab_size + i * vocab_size + token];
        float p_d = p_draft[batch_idx * K * vocab_size + i * vocab_size + token];

        // Acceptance probability: min(1, p_target / p_draft)
        float accept_prob = (p_d > 0.0f) ? min(1.0f, p_t / p_d) : 0.0f;

        float r = random_vals[batch_idx * K + i];

        if (r < accept_prob) {
            accepted[batch_idx * K + i] = 1;
            count++;
        } else {
            accepted[batch_idx * K + i] = 0;
            still_accepting = false;
        }
    }

    num_accepted[batch_idx] = count;
}

// FP16 variant (probabilities in half, compute in float)
kernel void speculative_verify_fp16(
    device const half*    p_target      [[buffer(0)]],
    device const half*    p_draft       [[buffer(1)]],
    device const int*     draft_tokens  [[buffer(2)]],
    device const float*   random_vals   [[buffer(3)]],
    device int*           accepted      [[buffer(4)]],
    device int*           num_accepted  [[buffer(5)]],
    constant uint&        K             [[buffer(6)]],
    constant uint&        vocab_size    [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint batch_idx = tid;

    uint count = 0;
    bool still_accepting = true;

    for (uint i = 0; i < K; i++) {
        if (!still_accepting) {
            accepted[batch_idx * K + i] = 0;
            continue;
        }

        int token = draft_tokens[batch_idx * K + i];

        float p_t = float(p_target[batch_idx * K * vocab_size + i * vocab_size + token]);
        float p_d = float(p_draft[batch_idx * K * vocab_size + i * vocab_size + token]);

        float accept_prob = (p_d > 0.0f) ? min(1.0f, p_t / p_d) : 0.0f;
        float r = random_vals[batch_idx * K + i];

        if (r < accept_prob) {
            accepted[batch_idx * K + i] = 1;
            count++;
        } else {
            accepted[batch_idx * K + i] = 0;
            still_accepting = false;
        }
    }

    num_accepted[batch_idx] = count;
}
