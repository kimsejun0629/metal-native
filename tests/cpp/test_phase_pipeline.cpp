/// @file test_phase_pipeline.cpp
/// @brief Unit tests for PhasePipeline prefill/decode detection and configuration.

#include <gtest/gtest.h>
#include "metal_native/inference/phase_pipeline.h"
#include "metal_native/core/device.h"
#include "metal_native/dispatch/command_pipeline.h"

using namespace metal_native;

class PhasePipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = &MNDevice::instance();
        pipeline_ = std::make_unique<PhasePipeline>(*device_);
    }

    MNDevice* device_;
    std::unique_ptr<PhasePipeline> pipeline_;
};

// -- Auto-detection tests ----------------------------------------------------

TEST_F(PhasePipelineTest, AutoDetectPrefill) {
    // seq_len > 1 && batch_size >= 1 → Prefill
    pipeline_->auto_detect_phase(1, 128);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Prefill);

    pipeline_->auto_detect_phase(4, 256);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Prefill);

    pipeline_->auto_detect_phase(8, 512);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Prefill);
}

TEST_F(PhasePipelineTest, AutoDetectDecode) {
    // seq_len == 1 → Decode
    pipeline_->auto_detect_phase(1, 1);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Decode);

    pipeline_->auto_detect_phase(4, 1);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Decode);

    // seq_len <= 4 && batch_size == 1 → Decode
    pipeline_->auto_detect_phase(1, 2);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Decode);

    pipeline_->auto_detect_phase(1, 4);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Decode);
}

TEST_F(PhasePipelineTest, AutoDetectEdgeCases) {
    // batch=1, seq=5 → Prefill (seq > 4)
    pipeline_->auto_detect_phase(1, 5);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Prefill);

    // batch=2, seq=2 → Prefill (batch > 1, even though seq <= 4)
    pipeline_->auto_detect_phase(2, 2);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Prefill);

    // batch=1, seq=1 → Decode (classic autoregressive)
    pipeline_->auto_detect_phase(1, 1);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Decode);
}

// -- Manual phase setting tests ----------------------------------------------

TEST_F(PhasePipelineTest, ManualPhaseSet) {
    // Initial phase is Unknown
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Unknown);

    // Set to Prefill
    pipeline_->set_phase(InferencePhase::Prefill);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Prefill);

    // Set to Decode
    pipeline_->set_phase(InferencePhase::Decode);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Decode);

    // Set back to Unknown
    pipeline_->set_phase(InferencePhase::Unknown);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Unknown);
}

TEST_F(PhasePipelineTest, ManualOverrideAutoDetect) {
    // Auto-detect sets phase to Prefill
    pipeline_->auto_detect_phase(1, 128);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Prefill);

    // Manual override to Decode
    pipeline_->set_phase(InferencePhase::Decode);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Decode);

    // Auto-detect again (should update phase)
    pipeline_->auto_detect_phase(1, 1);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Decode);
}

// -- Configuration tests -----------------------------------------------------

TEST_F(PhasePipelineTest, DefaultPrefillConfig) {
    const PhaseConfig& cfg = pipeline_->prefill_config();

    EXPECT_EQ(cfg.matmul_strategy, "tiled64");
    EXPECT_EQ(cfg.attention_variant, "standard");
    EXPECT_EQ(cfg.encoder_max_ops, 64u);
    EXPECT_EQ(cfg.auto_flush_threshold, 128u);
    EXPECT_TRUE(cfg.use_activation_pool);
}

TEST_F(PhasePipelineTest, DefaultDecodeConfig) {
    const PhaseConfig& cfg = pipeline_->decode_config();

    EXPECT_EQ(cfg.matmul_strategy, "vecmat");
    EXPECT_EQ(cfg.attention_variant, "4simd");
    EXPECT_EQ(cfg.encoder_max_ops, 16u);
    EXPECT_EQ(cfg.auto_flush_threshold, 32u);
    EXPECT_TRUE(cfg.use_activation_pool);
}

TEST_F(PhasePipelineTest, ActiveConfigPrefill) {
    pipeline_->set_phase(InferencePhase::Prefill);
    const PhaseConfig& cfg = pipeline_->active_config();

    EXPECT_EQ(cfg.matmul_strategy, "tiled64");
    EXPECT_EQ(cfg.attention_variant, "standard");
}

TEST_F(PhasePipelineTest, ActiveConfigDecode) {
    pipeline_->set_phase(InferencePhase::Decode);
    const PhaseConfig& cfg = pipeline_->active_config();

    EXPECT_EQ(cfg.matmul_strategy, "vecmat");
    EXPECT_EQ(cfg.attention_variant, "4simd");
}

TEST_F(PhasePipelineTest, ActiveConfigUnknown) {
    // Unknown phase should fall back to prefill config
    pipeline_->set_phase(InferencePhase::Unknown);
    const PhaseConfig& cfg = pipeline_->active_config();

    EXPECT_EQ(cfg.matmul_strategy, "tiled64");
    EXPECT_EQ(cfg.attention_variant, "standard");
}

// -- Custom configuration tests ----------------------------------------------

TEST_F(PhasePipelineTest, CustomPrefillConfig) {
    PhaseConfig custom;
    custom.matmul_strategy = "mpsgraph";
    custom.attention_variant = "gqa";
    custom.encoder_max_ops = 128;
    custom.auto_flush_threshold = 256;
    custom.use_activation_pool = false;

    pipeline_->set_prefill_config(custom);

    const PhaseConfig& cfg = pipeline_->prefill_config();
    EXPECT_EQ(cfg.matmul_strategy, "mpsgraph");
    EXPECT_EQ(cfg.attention_variant, "gqa");
    EXPECT_EQ(cfg.encoder_max_ops, 128u);
    EXPECT_EQ(cfg.auto_flush_threshold, 256u);
    EXPECT_FALSE(cfg.use_activation_pool);
}

TEST_F(PhasePipelineTest, CustomDecodeConfig) {
    PhaseConfig custom;
    custom.matmul_strategy = "simd32";
    custom.attention_variant = "gqa";
    custom.encoder_max_ops = 8;
    custom.auto_flush_threshold = 16;
    custom.use_activation_pool = false;

    pipeline_->set_decode_config(custom);

    const PhaseConfig& cfg = pipeline_->decode_config();
    EXPECT_EQ(cfg.matmul_strategy, "simd32");
    EXPECT_EQ(cfg.attention_variant, "gqa");
    EXPECT_EQ(cfg.encoder_max_ops, 8u);
    EXPECT_EQ(cfg.auto_flush_threshold, 16u);
    EXPECT_FALSE(cfg.use_activation_pool);
}

// -- Pipeline application tests ----------------------------------------------

TEST_F(PhasePipelineTest, ApplyPrefillToPipeline) {
    CommandPipeline& cmd_pipeline = device_->command_pipeline();

    // Set to prefill phase
    pipeline_->set_phase(InferencePhase::Prefill);
    pipeline_->apply_to_pipeline(cmd_pipeline);

    // Check that command pipeline settings match prefill config
    EXPECT_EQ(cmd_pipeline.encoder_max_ops(), 64u);
    EXPECT_EQ(cmd_pipeline.auto_flush_threshold(), 128u);
}

TEST_F(PhasePipelineTest, ApplyDecodeToPipeline) {
    CommandPipeline& cmd_pipeline = device_->command_pipeline();

    // Set to decode phase
    pipeline_->set_phase(InferencePhase::Decode);
    pipeline_->apply_to_pipeline(cmd_pipeline);

    // Check that command pipeline settings match decode config
    EXPECT_EQ(cmd_pipeline.encoder_max_ops(), 16u);
    EXPECT_EQ(cmd_pipeline.auto_flush_threshold(), 32u);
}

TEST_F(PhasePipelineTest, ApplyCustomConfigToPipeline) {
    CommandPipeline& cmd_pipeline = device_->command_pipeline();

    // Create custom config
    PhaseConfig custom;
    custom.encoder_max_ops = 99;
    custom.auto_flush_threshold = 199;

    pipeline_->set_prefill_config(custom);
    pipeline_->set_phase(InferencePhase::Prefill);
    pipeline_->apply_to_pipeline(cmd_pipeline);

    // Verify custom settings were applied
    EXPECT_EQ(cmd_pipeline.encoder_max_ops(), 99u);
    EXPECT_EQ(cmd_pipeline.auto_flush_threshold(), 199u);
}

// -- Phase transition tests --------------------------------------------------

TEST_F(PhasePipelineTest, PrefillToDecodeTransition) {
    CommandPipeline& cmd_pipeline = device_->command_pipeline();

    // Start in prefill
    pipeline_->set_phase(InferencePhase::Prefill);
    pipeline_->apply_to_pipeline(cmd_pipeline);
    EXPECT_EQ(cmd_pipeline.encoder_max_ops(), 64u);

    // Transition to decode
    pipeline_->set_phase(InferencePhase::Decode);
    pipeline_->apply_to_pipeline(cmd_pipeline);
    EXPECT_EQ(cmd_pipeline.encoder_max_ops(), 16u);

    // Back to prefill
    pipeline_->set_phase(InferencePhase::Prefill);
    pipeline_->apply_to_pipeline(cmd_pipeline);
    EXPECT_EQ(cmd_pipeline.encoder_max_ops(), 64u);
}

TEST_F(PhasePipelineTest, AutoTransitionDuringInference) {
    CommandPipeline& cmd_pipeline = device_->command_pipeline();

    // Simulate prefill: batch=1, seq=128
    pipeline_->auto_detect_phase(1, 128);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Prefill);
    pipeline_->apply_to_pipeline(cmd_pipeline);
    EXPECT_EQ(cmd_pipeline.encoder_max_ops(), 64u);

    // Simulate decode: batch=1, seq=1 (autoregressive generation)
    pipeline_->auto_detect_phase(1, 1);
    EXPECT_EQ(pipeline_->current_phase(), InferencePhase::Decode);
    pipeline_->apply_to_pipeline(cmd_pipeline);
    EXPECT_EQ(cmd_pipeline.encoder_max_ops(), 16u);
}
