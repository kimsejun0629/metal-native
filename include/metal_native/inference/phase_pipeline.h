#pragma once

/// @file phase_pipeline.h
/// @brief Inference pipeline that automatically detects and optimizes for prefill vs decode phases.
///
/// PhasePipeline routes inference operations through phase-specific configurations:
/// - Prefill: large batch/sequence → use tiled matmul, standard attention, larger buffers
/// - Decode: small batch/sequence → use vecmat matmul, optimized attention, smaller buffers
///
/// Auto-detection is based on batch_size and seq_len heuristics. Manual override is also supported.
///
/// Thread-safe: all public methods are guarded by a mutex.

#include <cstddef>
#include <memory>
#include <string>

namespace metal_native {

class MNDevice;
class CommandPipeline;

/// Inference phase enumeration
enum class InferencePhase {
    Prefill,   ///< Large batch/sequence (initial token processing)
    Decode,    ///< Small batch/sequence (autoregressive generation)
    Unknown    ///< Phase not yet determined
};

/// Phase-specific configuration for inference operations
struct PhaseConfig {
    /// Matmul strategy: "vecmat" (M=1), "simd32", "tiled64", or "mpsgraph"
    std::string matmul_strategy;

    /// Attention variant: "standard", "gqa", or "4simd"
    std::string attention_variant;

    /// Maximum operations per encoder scope before auto-flush
    size_t encoder_max_ops;

    /// Auto-flush threshold for command buffer operation count
    size_t auto_flush_threshold;

    /// Whether to use activation pool for buffer reuse
    bool use_activation_pool;

    PhaseConfig()
        : matmul_strategy("tiled64"),
          attention_variant("standard"),
          encoder_max_ops(32),
          auto_flush_threshold(64),
          use_activation_pool(true) {}
};

/// Pipeline that automatically configures inference based on prefill vs decode phase
class PhasePipeline {
public:
    /// Construct a phase pipeline for the given device
    explicit PhasePipeline(MNDevice& device);
    ~PhasePipeline();

    // Non-copyable, non-movable
    PhasePipeline(const PhasePipeline&) = delete;
    PhasePipeline& operator=(const PhasePipeline&) = delete;
    PhasePipeline(PhasePipeline&&) = delete;
    PhasePipeline& operator=(PhasePipeline&&) = delete;

    // -- Phase management ----------------------------------------------------

    /// Manually set the inference phase
    void set_phase(InferencePhase phase);

    /// Auto-detect phase based on batch size and sequence length
    ///
    /// Heuristic:
    /// - seq_len > 1 && batch_size >= 1 → Prefill
    /// - seq_len == 1 || (seq_len <= 4 && batch_size == 1) → Decode
    /// - Otherwise → Unknown (falls back to Prefill config)
    void auto_detect_phase(size_t batch_size, size_t seq_len);

    /// Get the current inference phase
    InferencePhase current_phase() const noexcept;

    // -- Configuration access ------------------------------------------------

    /// Get the configuration for prefill phase
    const PhaseConfig& prefill_config() const noexcept;

    /// Get the configuration for decode phase
    const PhaseConfig& decode_config() const noexcept;

    /// Get the active configuration (based on current phase)
    const PhaseConfig& active_config() const noexcept;

    // -- Configuration update ------------------------------------------------

    /// Update prefill configuration
    void set_prefill_config(const PhaseConfig& config);

    /// Update decode configuration
    void set_decode_config(const PhaseConfig& config);

    // -- Model-aware configuration -------------------------------------------

    /// Configure pipeline parameters based on model architecture.
    ///
    /// Automatically tunes encoder_max_ops, auto_flush_threshold, and
    /// attention_variant for both prefill and decode phases.
    ///
    /// @param num_layers   Number of transformer layers
    /// @param hidden_dim   Model hidden dimension
    /// @param num_heads    Number of attention heads
    /// @param head_dim     Dimension per attention head
    void configure_for_model(size_t num_layers, size_t hidden_dim,
                              size_t num_heads, size_t head_dim);

    // -- Pipeline application ------------------------------------------------

    /// Apply the current phase configuration to the device's command pipeline
    ///
    /// This sets encoder_max_ops and auto_flush_threshold on the CommandPipeline
    /// based on the active phase config.
    void apply_to_pipeline(CommandPipeline& pipeline);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
