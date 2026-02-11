/// @file test_kernel_cache.cpp
/// @brief Unit tests for KernelCache LRU cache.

#include <gtest/gtest.h>
#include "metal_native/kernels/kernel_cache.h"

#include <string>

using namespace metal_native;

// Test fixture for kernel cache tests
class KernelCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a small cache for testing LRU behavior
        cache_ = std::make_unique<KernelCache>(4);
    }

    std::unique_ptr<KernelCache> cache_;
};

// -- Basic operations tests ---------------------------------------------------

TEST_F(KernelCacheTest, InitialState) {
    EXPECT_EQ(cache_->size(), 0u);
    EXPECT_EQ(cache_->capacity(), 4u);
    EXPECT_EQ(cache_->hit_rate(), 0.0);
}

TEST_F(KernelCacheTest, GetMissingKeyReturnsNull) {
    void* pipeline = cache_->get("nonexistent_kernel");
    EXPECT_EQ(pipeline, nullptr);
}

TEST_F(KernelCacheTest, PutAndGet) {
    // Use a non-null pointer as a dummy pipeline state
    void* dummy_pipeline = reinterpret_cast<void*>(0x1234);

    cache_->put("test_kernel", dummy_pipeline);
    EXPECT_EQ(cache_->size(), 1u);

    void* retrieved = cache_->get("test_kernel");
    EXPECT_EQ(retrieved, dummy_pipeline);
}

TEST_F(KernelCacheTest, PutMultipleKeys) {
    void* pipeline1 = reinterpret_cast<void*>(0x1000);
    void* pipeline2 = reinterpret_cast<void*>(0x2000);
    void* pipeline3 = reinterpret_cast<void*>(0x3000);

    cache_->put("kernel1", pipeline1);
    cache_->put("kernel2", pipeline2);
    cache_->put("kernel3", pipeline3);

    EXPECT_EQ(cache_->size(), 3u);

    EXPECT_EQ(cache_->get("kernel1"), pipeline1);
    EXPECT_EQ(cache_->get("kernel2"), pipeline2);
    EXPECT_EQ(cache_->get("kernel3"), pipeline3);
}

// -- LRU eviction tests -------------------------------------------------------

TEST_F(KernelCacheTest, LRUEvictionWhenFull) {
    void* p1 = reinterpret_cast<void*>(0x1000);
    void* p2 = reinterpret_cast<void*>(0x2000);
    void* p3 = reinterpret_cast<void*>(0x3000);
    void* p4 = reinterpret_cast<void*>(0x4000);
    void* p5 = reinterpret_cast<void*>(0x5000);

    // Fill cache to capacity
    cache_->put("k1", p1);
    cache_->put("k2", p2);
    cache_->put("k3", p3);
    cache_->put("k4", p4);

    EXPECT_EQ(cache_->size(), 4u);

    // Insert one more - should evict oldest (k1)
    cache_->put("k5", p5);

    EXPECT_EQ(cache_->size(), 4u);
    EXPECT_EQ(cache_->get("k1"), nullptr);  // k1 was evicted
    EXPECT_EQ(cache_->get("k5"), p5);       // k5 is present
}

TEST_F(KernelCacheTest, LRUOrderPreservedByAccess) {
    void* p1 = reinterpret_cast<void*>(0x1000);
    void* p2 = reinterpret_cast<void*>(0x2000);
    void* p3 = reinterpret_cast<void*>(0x3000);
    void* p4 = reinterpret_cast<void*>(0x4000);
    void* p5 = reinterpret_cast<void*>(0x5000);

    // Fill cache
    cache_->put("k1", p1);
    cache_->put("k2", p2);
    cache_->put("k3", p3);
    cache_->put("k4", p4);

    // Access k1 to make it most recently used
    cache_->get("k1");

    // Insert k5 - should evict k2 (oldest) not k1
    cache_->put("k5", p5);

    EXPECT_NE(cache_->get("k1"), nullptr);  // k1 still present
    EXPECT_EQ(cache_->get("k2"), nullptr);  // k2 was evicted
}

TEST_F(KernelCacheTest, RepeatedAccessPromotesToMRU) {
    void* p1 = reinterpret_cast<void*>(0x1000);
    void* p2 = reinterpret_cast<void*>(0x2000);
    void* p3 = reinterpret_cast<void*>(0x3000);
    void* p4 = reinterpret_cast<void*>(0x4000);
    void* p5 = reinterpret_cast<void*>(0x5000);

    cache_->put("k1", p1);
    cache_->put("k2", p2);
    cache_->put("k3", p3);
    cache_->put("k4", p4);

    // Repeatedly access k1
    cache_->get("k1");
    cache_->get("k1");
    cache_->get("k1");

    // Insert k5 - k1 should remain because it's most recently used
    cache_->put("k5", p5);

    EXPECT_NE(cache_->get("k1"), nullptr);
}

// -- Overwrite tests ----------------------------------------------------------

TEST_F(KernelCacheTest, OverwriteExistingKey) {
    void* old_pipeline = reinterpret_cast<void*>(0x1000);
    void* new_pipeline = reinterpret_cast<void*>(0x2000);

    cache_->put("kernel", old_pipeline);
    EXPECT_EQ(cache_->size(), 1u);

    cache_->put("kernel", new_pipeline);
    EXPECT_EQ(cache_->size(), 1u);  // Size shouldn't change

    EXPECT_EQ(cache_->get("kernel"), new_pipeline);
}

// -- Clear tests --------------------------------------------------------------

TEST_F(KernelCacheTest, ClearEmptiesCache) {
    cache_->put("k1", reinterpret_cast<void*>(0x1000));
    cache_->put("k2", reinterpret_cast<void*>(0x2000));
    cache_->put("k3", reinterpret_cast<void*>(0x3000));

    EXPECT_EQ(cache_->size(), 3u);

    cache_->clear();

    EXPECT_EQ(cache_->size(), 0u);
    EXPECT_EQ(cache_->get("k1"), nullptr);
    EXPECT_EQ(cache_->get("k2"), nullptr);
    EXPECT_EQ(cache_->get("k3"), nullptr);
}

TEST_F(KernelCacheTest, ClearResetsStats) {
    cache_->put("k1", reinterpret_cast<void*>(0x1000));
    cache_->get("k1");  // Hit
    cache_->get("k2");  // Miss

    cache_->clear();

    // After clear, hit rate should be 0 (no lookups)
    EXPECT_EQ(cache_->hit_rate(), 0.0);
}

// -- Hit rate tests -----------------------------------------------------------

TEST_F(KernelCacheTest, HitRateAfterHit) {
    void* pipeline = reinterpret_cast<void*>(0x1000);

    cache_->put("kernel", pipeline);
    cache_->get("kernel");  // Hit

    EXPECT_GT(cache_->hit_rate(), 0.0);
}

TEST_F(KernelCacheTest, HitRateAfterMiss) {
    cache_->get("nonexistent");  // Miss

    EXPECT_EQ(cache_->hit_rate(), 0.0);
}

TEST_F(KernelCacheTest, HitRateCalculation) {
    void* pipeline = reinterpret_cast<void*>(0x1000);

    cache_->put("kernel", pipeline);

    cache_->get("kernel");      // Hit
    cache_->get("kernel");      // Hit
    cache_->get("missing");     // Miss

    // Hit rate should be 2/3 = 0.666...
    double hit_rate = cache_->hit_rate();
    EXPECT_NEAR(hit_rate, 2.0 / 3.0, 0.01);
}

// -- Capacity tests -----------------------------------------------------------

TEST_F(KernelCacheTest, CustomCapacity) {
    KernelCache small_cache(2);
    EXPECT_EQ(small_cache.capacity(), 2u);

    small_cache.put("k1", reinterpret_cast<void*>(0x1000));
    small_cache.put("k2", reinterpret_cast<void*>(0x2000));
    EXPECT_EQ(small_cache.size(), 2u);

    // Third insertion should evict
    small_cache.put("k3", reinterpret_cast<void*>(0x3000));
    EXPECT_EQ(small_cache.size(), 2u);
}

TEST_F(KernelCacheTest, LargeCapacity) {
    KernelCache large_cache(1024);
    EXPECT_EQ(large_cache.capacity(), 1024u);

    // Add many entries
    for (int i = 0; i < 100; ++i) {
        std::string key = "kernel_" + std::to_string(i);
        void* pipeline = reinterpret_cast<void*>(static_cast<uintptr_t>(i + 1));
        large_cache.put(key, pipeline);
    }

    EXPECT_EQ(large_cache.size(), 100u);
}

// -- Move semantics tests -----------------------------------------------------

TEST_F(KernelCacheTest, MoveConstructor) {
    cache_->put("kernel", reinterpret_cast<void*>(0x1000));

    KernelCache moved_cache(std::move(*cache_));

    EXPECT_EQ(moved_cache.size(), 1u);
    EXPECT_NE(moved_cache.get("kernel"), nullptr);
}

TEST_F(KernelCacheTest, MoveAssignment) {
    cache_->put("kernel", reinterpret_cast<void*>(0x1000));

    KernelCache other_cache(8);
    other_cache = std::move(*cache_);

    EXPECT_EQ(other_cache.size(), 1u);
    EXPECT_NE(other_cache.get("kernel"), nullptr);
}

// -- Key naming tests ---------------------------------------------------------

TEST_F(KernelCacheTest, LongKeyNames) {
    std::string long_key(1000, 'x');
    void* pipeline = reinterpret_cast<void*>(0x1000);

    cache_->put(long_key, pipeline);
    EXPECT_EQ(cache_->get(long_key), pipeline);
}

TEST_F(KernelCacheTest, SpecialCharactersInKeys) {
    std::string special_key = "kernel::add<float, 3>::v1.2.3";
    void* pipeline = reinterpret_cast<void*>(0x1000);

    cache_->put(special_key, pipeline);
    EXPECT_EQ(cache_->get(special_key), pipeline);
}
