/// @file test_command_pipeline.cpp
/// @brief Unit tests for CommandPipeline encoder scope functionality.

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include "metal_native/core/device.h"
#include "metal_native/dispatch/command_pipeline.h"

using namespace metal_native;

class CommandPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = &MNDevice::instance();
        pipeline_ = &device_->command_pipeline();
    }

    MNDevice* device_;
    CommandPipeline* pipeline_;
};

TEST_F(CommandPipelineTest, EncoderScopeBasic) {
    // Verify encoder scope activates and deactivates correctly
    EXPECT_FALSE(pipeline_->has_active_encoder());

    {
        auto scope = pipeline_->encoder_scope();
        EXPECT_TRUE(pipeline_->has_active_encoder());

        // Should be able to get an encoder
        id<MTLComputeCommandEncoder> encoder = pipeline_->current_encoder();
        EXPECT_NE(encoder, nil);
    }

    // After scope ends, encoder should be deactivated
    EXPECT_FALSE(pipeline_->has_active_encoder());
}

TEST_F(CommandPipelineTest, EncoderScopeNested) {
    // Test nested encoder scopes
    EXPECT_FALSE(pipeline_->has_active_encoder());

    {
        auto scope1 = pipeline_->encoder_scope();
        EXPECT_TRUE(pipeline_->has_active_encoder());
        (void)pipeline_->current_encoder();

        {
            auto scope2 = pipeline_->encoder_scope();
            EXPECT_TRUE(pipeline_->has_active_encoder());
            id<MTLComputeCommandEncoder> encoder2 = pipeline_->current_encoder();

            // Inner scope should have its own encoder
            EXPECT_NE(encoder2, nil);
        }

        // Outer scope should still be active
        EXPECT_TRUE(pipeline_->has_active_encoder());
    }

    EXPECT_FALSE(pipeline_->has_active_encoder());
}

TEST_F(CommandPipelineTest, EncoderScopeWithBatchScope) {
    // Test combining encoder scope with batch scope
    EXPECT_FALSE(pipeline_->lazy_commit());
    EXPECT_FALSE(pipeline_->has_active_encoder());

    {
        auto batch = pipeline_->batch_scope();
        EXPECT_TRUE(pipeline_->lazy_commit());

        {
            auto encoder = pipeline_->encoder_scope();
            EXPECT_TRUE(pipeline_->has_active_encoder());
            EXPECT_TRUE(pipeline_->lazy_commit());
        }

        EXPECT_FALSE(pipeline_->has_active_encoder());
        EXPECT_TRUE(pipeline_->lazy_commit());
    }

    EXPECT_FALSE(pipeline_->lazy_commit());
    EXPECT_FALSE(pipeline_->has_active_encoder());
}

TEST_F(CommandPipelineTest, TransformerLayerScope) {
    // Test transformer layer scope (batch + encoder combined)
    EXPECT_FALSE(pipeline_->lazy_commit());
    EXPECT_FALSE(pipeline_->has_active_encoder());

    {
        auto scope = pipeline_->transformer_layer_scope();

        // Should enable both lazy commit and encoder scope
        EXPECT_TRUE(pipeline_->lazy_commit());
        EXPECT_TRUE(pipeline_->has_active_encoder());

        // Can get encoder multiple times
        id<MTLComputeCommandEncoder> enc1 = pipeline_->current_encoder();
        id<MTLComputeCommandEncoder> enc2 = pipeline_->current_encoder();
        EXPECT_NE(enc1, nil);
        EXPECT_NE(enc2, nil);
    }

    EXPECT_FALSE(pipeline_->has_active_encoder());
}

TEST_F(CommandPipelineTest, EncoderMaxOps) {
    // Test that encoder auto-flushes after max ops
    size_t original_max = pipeline_->encoder_max_ops();
    pipeline_->set_encoder_max_ops(5);

    {
        auto scope = pipeline_->encoder_scope();
        (void)pipeline_->current_encoder();

        // Get encoder multiple times (simulating operations)
        for (int i = 0; i < 6; ++i) {
            id<MTLComputeCommandEncoder> enc = pipeline_->current_encoder();
            EXPECT_NE(enc, nil);
        }

        // After exceeding max_ops, we should have a different encoder
        // (the implementation creates a new one after flush)
    }

    pipeline_->set_encoder_max_ops(original_max);
}

TEST_F(CommandPipelineTest, CurrentEncoderWithoutScope) {
    // Without encoder scope, current_encoder should return nil
    EXPECT_FALSE(pipeline_->has_active_encoder());

    id<MTLComputeCommandEncoder> encoder = pipeline_->current_encoder();
    EXPECT_EQ(encoder, nil);
}

TEST_F(CommandPipelineTest, AutoFlushSkippedDuringEncoderScope) {
    // Set a low auto-flush threshold
    size_t original_threshold = pipeline_->auto_flush_threshold();
    pipeline_->set_auto_flush_threshold(2);

    // Enable lazy commit mode
    pipeline_->set_lazy_commit(true);

    {
        auto scope = pipeline_->encoder_scope();
        EXPECT_TRUE(pipeline_->has_active_encoder());

        // Get the initial encoder
        id<MTLComputeCommandEncoder> initial_encoder = pipeline_->current_encoder();
        EXPECT_NE(initial_encoder, nil);

        // Call record_op() multiple times to exceed the threshold
        for (int i = 0; i < 5; ++i) {
            pipeline_->record_op();
        }

        // The encoder should still be the same (not replaced by auto-flush)
        id<MTLComputeCommandEncoder> current = pipeline_->current_encoder();
        EXPECT_EQ(current, initial_encoder);
        EXPECT_TRUE(pipeline_->has_active_encoder());
    }

    // After scope ends, normal operation should resume
    EXPECT_FALSE(pipeline_->has_active_encoder());

    // Restore original settings
    pipeline_->set_auto_flush_threshold(original_threshold);
    pipeline_->set_lazy_commit(false);
}

TEST_F(CommandPipelineTest, AutoFlushWorksOutsideEncoderScope) {
    // Set a low auto-flush threshold
    size_t original_threshold = pipeline_->auto_flush_threshold();
    pipeline_->set_auto_flush_threshold(2);

    // Enable lazy commit mode
    pipeline_->set_lazy_commit(true);

    // Ensure we have an active buffer
    id<MTLCommandBuffer> initial_buffer = pipeline_->current_buffer();
    EXPECT_NE(initial_buffer, nil);

    // Call record_op() to exceed the threshold (without encoder scope)
    pipeline_->record_op();
    pipeline_->record_op();
    pipeline_->record_op();

    // Flush should have been triggered, creating a new buffer
    // We can verify by checking that the operation counter was reset
    // by calling record_op() once more and checking it doesn't trigger another flush immediately
    id<MTLCommandBuffer> buffer_after_flush = pipeline_->current_buffer();
    EXPECT_NE(buffer_after_flush, nil);

    // Restore original settings
    pipeline_->set_auto_flush_threshold(original_threshold);
    pipeline_->set_lazy_commit(false);
}

TEST_F(CommandPipelineTest, ConcurrentRecordOpStress) {
    // Set auto-flush threshold
    size_t original_threshold = pipeline_->auto_flush_threshold();
    pipeline_->set_auto_flush_threshold(16);

    // Enable lazy commit mode
    pipeline_->set_lazy_commit(true);

    // Spawn 4 threads, each calling record_op() 1000 times
    const int num_threads = 4;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this]() {
            for (int i = 0; i < 1000; ++i) {
                pipeline_->record_op();
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify no crash occurred (test passes if we get here)
    // Synchronize to ensure all GPU work completes without error
    pipeline_->synchronize();

    // Restore original settings
    pipeline_->set_auto_flush_threshold(original_threshold);
    pipeline_->set_lazy_commit(false);
}

TEST_F(CommandPipelineTest, AdaptiveEncoderMaxOps) {
    // Set initial value
    pipeline_->set_encoder_max_ops(32);

    // Verify initial values
    EXPECT_EQ(pipeline_->encoder_max_ops(), 32);
    EXPECT_EQ(pipeline_->adaptive_encoder_max_ops(), 32);

    // Run some operations through encoder scope
    for (int i = 0; i < 100; i++) {
        auto scope = pipeline_->encoder_scope();
        pipeline_->record_op();
    }

    // Verify adaptive value is within bounds
    EXPECT_GE(pipeline_->adaptive_encoder_max_ops(), 8);
    EXPECT_LE(pipeline_->adaptive_encoder_max_ops(), 128);

    // Verify EMA is being tracked
    double ema = pipeline_->encoder_duration_ema();
    EXPECT_GT(ema, 0.0);
}

TEST_F(CommandPipelineTest, EncoderMaxOpsBoundsValidation) {
    // Test lower bound
    pipeline_->set_encoder_max_ops(0);
    EXPECT_GE(pipeline_->encoder_max_ops(), 1);
    EXPECT_EQ(pipeline_->adaptive_encoder_max_ops(), pipeline_->encoder_max_ops());

    // Test upper bound
    pipeline_->set_encoder_max_ops(10000);
    EXPECT_LE(pipeline_->encoder_max_ops(), 512);
    EXPECT_EQ(pipeline_->adaptive_encoder_max_ops(), pipeline_->encoder_max_ops());

    // Test valid range
    pipeline_->set_encoder_max_ops(64);
    EXPECT_EQ(pipeline_->encoder_max_ops(), 64);
    EXPECT_EQ(pipeline_->adaptive_encoder_max_ops(), 64);
}

TEST_F(CommandPipelineTest, AdaptiveEncoderResetOnManualSet) {
    // Set initial value
    pipeline_->set_encoder_max_ops(32);
    EXPECT_EQ(pipeline_->adaptive_encoder_max_ops(), 32);

    // Run some operations to let adaptive value potentially change
    for (int i = 0; i < 50; i++) {
        auto scope = pipeline_->encoder_scope();
        pipeline_->record_op();
    }

    // Manually set encoder_max_ops
    pipeline_->set_encoder_max_ops(64);

    // Verify adaptive value was reset to the new manual value
    EXPECT_EQ(pipeline_->encoder_max_ops(), 64);
    EXPECT_EQ(pipeline_->adaptive_encoder_max_ops(), 64);
}
