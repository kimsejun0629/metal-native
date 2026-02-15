/// @file test_activation_pool.cpp
/// @brief Unit tests for ActivationPool transformer memory reuse.

#include <gtest/gtest.h>
#include "metal_native/memory/activation_pool.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include <thread>
#include <vector>

using namespace metal_native;

// Test fixture for activation pool tests
class ActivationPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            device_ = &MNDevice::instance();
            has_metal_ = (device_->metal_device() != nullptr);
        } catch (...) {
            has_metal_ = false;
        }

        if (!has_metal_) {
            GTEST_SKIP() << "Metal device not available, skipping hardware tests";
        }

        // 512 MB budget for testing
        pool_ = std::make_unique<ActivationPool>(*device_, 512 * 1024 * 1024);
    }

    void TearDown() override {
        if (pool_) {
            pool_->purge_unused();
        }
    }

    MNDevice* device_ = nullptr;
    std::unique_ptr<ActivationPool> pool_;
    bool has_metal_ = false;
};

// -- Basic allocation tests ---------------------------------------------------

TEST_F(ActivationPoolTest, AcquireSingleBuffer) {
    auto handle = pool_->acquire(1024 * 1024, "test_buffer");

    EXPECT_NE(handle.metal_buffer, nullptr);
    EXPECT_GE(handle.size, 1024u * 1024u);

    pool_->release(handle);
}

TEST_F(ActivationPoolTest, AcquireMultipleBuffers) {
    auto h1 = pool_->acquire(512 * 1024, "buffer1");
    auto h2 = pool_->acquire(1024 * 1024, "buffer2");
    auto h3 = pool_->acquire(2048 * 1024, "buffer3");

    EXPECT_NE(h1.metal_buffer, nullptr);
    EXPECT_NE(h2.metal_buffer, nullptr);
    EXPECT_NE(h3.metal_buffer, nullptr);

    EXPECT_GE(h1.size, 512u * 1024u);
    EXPECT_GE(h2.size, 1024u * 1024u);
    EXPECT_GE(h3.size, 2048u * 1024u);

    pool_->release(h1);
    pool_->release(h2);
    pool_->release(h3);
}

// -- Reuse and caching tests --------------------------------------------------

TEST_F(ActivationPoolTest, BufferReuse) {
    // First acquire/release cycle
    auto h1 = pool_->acquire(1024 * 1024, "first");
    void* first_ptr = h1.metal_buffer;
    pool_->release(h1);

    // Second acquire should reuse the same buffer
    auto h2 = pool_->acquire(1024 * 1024, "second");
    EXPECT_EQ(h2.metal_buffer, first_ptr) << "Buffer should be reused from pool";

    pool_->release(h2);
}

TEST_F(ActivationPoolTest, PoolHitRate) {
    size_t size = 1024 * 1024;

    // First allocation is a miss
    auto h1 = pool_->acquire(size, "miss");
    EXPECT_EQ(pool_->miss_count(), 1u);
    EXPECT_EQ(pool_->hit_count(), 0u);
    pool_->release(h1);

    // Second allocation should be a hit
    auto h2 = pool_->acquire(size, "hit");
    EXPECT_EQ(pool_->hit_count(), 1u);
    pool_->release(h2);

    // Third allocation should also be a hit
    auto h3 = pool_->acquire(size, "hit2");
    EXPECT_EQ(pool_->hit_count(), 2u);
    pool_->release(h3);

    // Check hit rate
    float hit_rate = pool_->hit_rate();
    EXPECT_GT(hit_rate, 0.5f) << "Hit rate should be > 50%";
}

// -- Model configuration tests ------------------------------------------------

TEST_F(ActivationPoolTest, ConfigureForModel) {
    // GPT-2 small configuration
    size_t hidden_dim = 768;
    size_t num_heads = 12;
    size_t max_seq_len = 1024;
    size_t batch_size = 1;

    pool_->configure_for_model(hidden_dim, num_heads, max_seq_len, batch_size, 1);

    // Pool should have pre-allocated buffers
    EXPECT_GT(pool_->pool_size(), 0u) << "Pool should have pre-allocated buffers";
    EXPECT_GT(pool_->total_allocated(), 0u) << "Some memory should be allocated";
}

TEST_F(ActivationPoolTest, ModelSpecificSizes) {
    // Configure for a small model
    size_t hidden_dim = 512;
    size_t num_heads = 8;
    size_t max_seq_len = 512;
    size_t batch_size = 1;

    pool_->configure_for_model(hidden_dim, num_heads, max_seq_len, batch_size, 1);

    // Calculate expected QKV size
    size_t qkv_size = batch_size * max_seq_len * 3 * hidden_dim * sizeof(float);

    // First acquisition should hit the pre-allocated buffer
    auto h1 = pool_->acquire(qkv_size, "qkv_proj");
    size_t initial_hits = pool_->hit_count();

    // Should be a hit if pre-allocation worked
    EXPECT_GT(initial_hits, 0u) << "Pre-allocated buffer should be used";

    pool_->release(h1);
}

// -- Memory management tests --------------------------------------------------

TEST_F(ActivationPoolTest, ReleaseAll) {
    pool_->acquire(512 * 1024, "buf1");
    pool_->acquire(1024 * 1024, "buf2");
    pool_->acquire(2048 * 1024, "buf3");

    size_t in_use = pool_->total_in_use();
    EXPECT_GT(in_use, 0u) << "Some memory should be in use";

    // Release all at once
    pool_->release_all();

    EXPECT_EQ(pool_->total_in_use(), 0u) << "All memory should be released";
}

TEST_F(ActivationPoolTest, PurgeUnused) {
    // Allocate and release some buffers
    for (int i = 0; i < 5; ++i) {
        auto h = pool_->acquire(1024 * 1024, "temp");
        pool_->release(h);
    }

    size_t before_purge = pool_->total_allocated();
    EXPECT_GT(before_purge, 0u);

    // Purge should free unused buffers
    pool_->purge_unused();

    size_t after_purge = pool_->total_allocated();
    EXPECT_LT(after_purge, before_purge) << "Purge should reduce allocated memory";
}

TEST_F(ActivationPoolTest, StatsTracking) {
    // Initial stats
    EXPECT_EQ(pool_->total_allocated(), 0u);
    EXPECT_EQ(pool_->total_in_use(), 0u);
    EXPECT_EQ(pool_->hit_count(), 0u);
    EXPECT_EQ(pool_->miss_count(), 0u);

    // Acquire a buffer
    auto h = pool_->acquire(1024 * 1024, "test");

    EXPECT_GT(pool_->total_allocated(), 0u);
    EXPECT_GT(pool_->total_in_use(), 0u);
    EXPECT_EQ(pool_->miss_count(), 1u);

    pool_->release(h);

    EXPECT_EQ(pool_->total_in_use(), 0u) << "In-use should be zero after release";
    EXPECT_GT(pool_->total_allocated(), 0u) << "Allocated memory still in pool";
}

// -- Transformer inference simulation -----------------------------------------

TEST_F(ActivationPoolTest, SimulateTransformerLayer) {
    // Simulate a small transformer layer
    size_t hidden_dim = 512;
    size_t num_heads = 8;
    size_t seq_len = 256;
    size_t batch_size = 1;

    pool_->configure_for_model(hidden_dim, num_heads, seq_len, batch_size, 1);

    // Simulate one forward pass through a layer
    for (int layer = 0; layer < 3; ++layer) {
        // QKV projection
        size_t qkv_size = batch_size * seq_len * 3 * hidden_dim * sizeof(float);
        auto qkv = pool_->acquire(qkv_size, "qkv");

        // Attention scores
        size_t attn_size = batch_size * num_heads * seq_len * seq_len * sizeof(float);
        auto attn = pool_->acquire(attn_size, "attn");

        // FFN intermediate
        size_t ffn_size = batch_size * seq_len * 4 * hidden_dim * sizeof(float);
        auto ffn = pool_->acquire(ffn_size, "ffn");

        // Release in reverse order (simulating computation)
        pool_->release(ffn);
        pool_->release(attn);
        pool_->release(qkv);
    }

    // After 3 layers, hit rate should be high
    float hit_rate = pool_->hit_rate();
    EXPECT_GT(hit_rate, 0.6f) << "Hit rate should be > 60% for repeated sizes";
}

TEST_F(ActivationPoolTest, RepeatedInference) {
    // Configure for model
    pool_->configure_for_model(512, 8, 256, 1, 1);

    size_t iterations = 10;
    size_t buffer_size = 1024 * 1024;

    for (size_t i = 0; i < iterations; ++i) {
        auto h = pool_->acquire(buffer_size, "inference");
        pool_->release(h);
    }

    // Most allocations should be hits
    float hit_rate = pool_->hit_rate();
    EXPECT_GT(hit_rate, 0.85f) << "Hit rate should be > 85% for repeated inference";

    // Memory should be stable (no leaks)
    EXPECT_EQ(pool_->total_in_use(), 0u) << "No memory should be in use after release_all";
}

// -- Thread safety tests ------------------------------------------------------

TEST_F(ActivationPoolTest, ConcurrentAcquireRelease) {
    const int num_threads = 4;

    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this]() {
            for (int i = 0; i < 100; ++i) {
                auto h = pool_->acquire(512 * 1024, "concurrent");
                pool_->release(h);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // All buffers should be released
    EXPECT_EQ(pool_->total_in_use(), 0u);

    // Hit rate should be reasonable
    float hit_rate = pool_->hit_rate();
    EXPECT_GT(hit_rate, 0.5f) << "Concurrent access should still achieve decent hit rate";
}

// -- Error handling tests -----------------------------------------------------

TEST_F(ActivationPoolTest, DoubleRelease) {
    auto h = pool_->acquire(1024 * 1024, "test");
    pool_->release(h);

    // Double release should throw
    EXPECT_THROW(pool_->release(h), MNException);
}

TEST_F(ActivationPoolTest, InvalidHandle) {
    ActivationPool::BufferHandle invalid_handle{
        .metal_buffer = nullptr,
        .size = 0,
        .pool_index = 999999
    };

    // Releasing invalid handle should throw
    EXPECT_THROW(pool_->release(invalid_handle), MNException);
}
