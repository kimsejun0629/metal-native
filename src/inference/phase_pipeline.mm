/// @file phase_pipeline.mm
/// @brief Implementation of PhasePipeline for prefill/decode phase optimization.

#include "metal_native/inference/phase_pipeline.h"
#include "metal_native/core/device.h"
#include "metal_native/dispatch/command_pipeline.h"
#include "metal_native/core/error.h"

#include <mutex>

namespace metal_native {

// ---------------------------------------------------------------------------
// PhasePipeline::Impl
// ---------------------------------------------------------------------------

struct PhasePipeline::Impl {
    MNDevice& device;
    InferencePhase current_phase;
    PhaseConfig prefill_cfg;
    PhaseConfig decode_cfg;
    mutable std::mutex mutex;

    explicit Impl(MNDevice& dev)
        : device(dev),
          current_phase(InferencePhase::Unknown) {
        // Initialize prefill config (large batch/sequence)
        prefill_cfg.matmul_strategy = "tiled64";
        prefill_cfg.attention_variant = "standard";
        prefill_cfg.encoder_max_ops = 64;
        prefill_cfg.auto_flush_threshold = 128;
        prefill_cfg.use_activation_pool = true;

        // Initialize decode config (small batch/sequence, autoregressive)
        decode_cfg.matmul_strategy = "vecmat";
        decode_cfg.attention_variant = "4simd";
        decode_cfg.encoder_max_ops = 16;
        decode_cfg.auto_flush_threshold = 32;
        decode_cfg.use_activation_pool = true;
    }

    InferencePhase detect_phase(size_t batch_size, size_t seq_len) const {
        // Decode: single token generation (seq_len == 1) or very short sequences
        // with single batch (autoregressive generation)
        if (seq_len == 1 || (seq_len <= 4 && batch_size == 1)) {
            return InferencePhase::Decode;
        }

        // Prefill: processing multiple tokens at once (seq > 4 or batch > 1)
        return InferencePhase::Prefill;
    }

    const PhaseConfig& get_active_config() const {
        switch (current_phase) {
            case InferencePhase::Prefill:
                return prefill_cfg;
            case InferencePhase::Decode:
                return decode_cfg;
            case InferencePhase::Unknown:
            default:
                // Fall back to prefill config for unknown phase
                return prefill_cfg;
        }
    }
};

// ---------------------------------------------------------------------------
// PhasePipeline public API
// ---------------------------------------------------------------------------

PhasePipeline::PhasePipeline(MNDevice& device)
    : impl_(std::make_unique<Impl>(device)) {}

PhasePipeline::~PhasePipeline() = default;

void PhasePipeline::set_phase(InferencePhase phase) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->current_phase = phase;
}

void PhasePipeline::auto_detect_phase(size_t batch_size, size_t seq_len) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->current_phase = impl_->detect_phase(batch_size, seq_len);
}

InferencePhase PhasePipeline::current_phase() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->current_phase;
}

const PhaseConfig& PhasePipeline::prefill_config() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->prefill_cfg;
}

const PhaseConfig& PhasePipeline::decode_config() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->decode_cfg;
}

const PhaseConfig& PhasePipeline::active_config() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->get_active_config();
}

void PhasePipeline::set_prefill_config(const PhaseConfig& config) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->prefill_cfg = config;
}

void PhasePipeline::set_decode_config(const PhaseConfig& config) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->decode_cfg = config;
}

void PhasePipeline::configure_for_model(size_t num_layers, size_t hidden_dim,
                                          size_t num_heads, size_t head_dim) {
    // Tune based on model size categories
    PhaseConfig prefill;
    PhaseConfig decode;

    if (hidden_dim <= 2048) {
        // Small models (e.g., Qwen2.5-0.5B, GPT2-Small)
        // Fewer ops per layer, smaller buffers
        prefill.matmul_strategy = "tiled64";
        prefill.attention_variant = "standard";
        prefill.encoder_max_ops = 16;
        prefill.auto_flush_threshold = 32;
        prefill.use_activation_pool = true;

        decode.matmul_strategy = "vecmat";
        decode.attention_variant = (head_dim <= 128) ? "4simd" : "standard";
        decode.encoder_max_ops = 12;
        decode.auto_flush_threshold = 24;
        decode.use_activation_pool = true;
    } else if (hidden_dim <= 4096) {
        // Medium models (e.g., Llama-3.2-3B, Qwen2.5-3B)
        prefill.matmul_strategy = "tiled64";
        prefill.attention_variant = "standard";
        prefill.encoder_max_ops = 32;
        prefill.auto_flush_threshold = 64;
        prefill.use_activation_pool = true;

        decode.matmul_strategy = "vecmat";
        decode.attention_variant = (head_dim <= 128) ? "4simd" : "standard";
        decode.encoder_max_ops = 24;
        decode.auto_flush_threshold = 48;
        decode.use_activation_pool = true;
    } else {
        // Large models (e.g., Qwen2.5-7B, Llama-7B)
        // More ops per layer, benefit from larger batches
        prefill.matmul_strategy = "tiled64";
        prefill.attention_variant = "standard";
        prefill.encoder_max_ops = 48;
        prefill.auto_flush_threshold = 96;
        prefill.use_activation_pool = true;

        decode.matmul_strategy = "vecmat";
        decode.attention_variant = (head_dim <= 128) ? "4simd" : "standard";
        decode.encoder_max_ops = 32;
        decode.auto_flush_threshold = 64;
        decode.use_activation_pool = true;
    }

    set_prefill_config(prefill);
    set_decode_config(decode);
}

void PhasePipeline::apply_to_pipeline(CommandPipeline& pipeline) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const PhaseConfig& cfg = impl_->get_active_config();

    // Apply encoder and auto-flush settings
    pipeline.set_encoder_max_ops(cfg.encoder_max_ops);
    pipeline.set_auto_flush_threshold(cfg.auto_flush_threshold);

    // Note: matmul_strategy, attention_variant, and use_activation_pool
    // are advisory fields that operations can query via active_config().
    // They are not directly applied to CommandPipeline because those
    // decisions happen at the operation level (matmul.mm, attention.mm, etc.).
}

} // namespace metal_native
