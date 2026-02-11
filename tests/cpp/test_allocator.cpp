/// @file test_allocator.cpp
/// @brief Unit tests for MetalSmartAllocator memory management.

#include <gtest/gtest.h>
#include "metal_native/memory/allocator.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

using namespace metal_native;

// Test fixture for allocator tests
class AllocatorTest : public ::testing::Test {
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

        allocator_ = std::make_unique<MetalSmartAllocator>(*device_);
    }

    void TearDown() override {
        if (allocator_) {
            allocator_->empty_cache();
        }
    }

    MNDevice* device_ = nullptr;
    std::unique_ptr<MetalSmartAllocator> allocator_;
    bool has_metal_ = false;
};

// -- Basic allocation tests ---------------------------------------------------

TEST_F(AllocatorTest, AllocateSingleBlock) {
    AllocatedBlock block = allocator_->allocate(1024, StorageMode::Shared);

    EXPECT_GE(block.size, 1024u);
    EXPECT_EQ(block.requested, 1024u);
    EXPECT_EQ(block.mode, StorageMode::Shared);
    EXPECT_NE(block.buffer, nullptr);

    allocator_->deallocate(block);
}

TEST_F(AllocatorTest, AllocateMultipleBlocks) {
    AllocatedBlock block1 = allocator_->allocate(512, StorageMode::Shared);
    AllocatedBlock block2 = allocator_->allocate(1024, StorageMode::Shared);
    AllocatedBlock block3 = allocator_->allocate(2048, StorageMode::Shared);

    EXPECT_GE(block1.size, 512u);
    EXPECT_GE(block2.size, 1024u);
    EXPECT_GE(block3.size, 2048u);

    allocator_->deallocate(block1);
    allocator_->deallocate(block2);
    allocator_->deallocate(block3);
}

TEST_F(AllocatorTest, AllocatePrivateBuffer) {
    AllocatedBlock block = allocator_->allocate(1024, StorageMode::Private);

    EXPECT_GE(block.size, 1024u);
    EXPECT_EQ(block.mode, StorageMode::Private);
    EXPECT_NE(block.buffer, nullptr);

    allocator_->deallocate(block);
}

// -- Deallocation and caching tests -------------------------------------------

TEST_F(AllocatorTest, DeallocateBlock) {
    AllocatedBlock block = allocator_->allocate(1024, StorageMode::Shared);
    AllocatorStats stats_before = allocator_->stats();

    allocator_->deallocate(block);

    AllocatorStats stats_after = allocator_->stats();
    EXPECT_LT(stats_after.allocated_bytes, stats_before.allocated_bytes);
    EXPECT_GT(stats_after.cached_bytes, 0u);
}

TEST_F(AllocatorTest, CacheReuse) {
    // Allocate and deallocate to populate cache
    AllocatedBlock block1 = allocator_->allocate(1024, StorageMode::Shared);
    allocator_->deallocate(block1);

    size_t cache_hits_before = allocator_->stats().cache_hit_count;

    // Allocate again - should hit cache
    AllocatedBlock block2 = allocator_->allocate(1024, StorageMode::Shared);

    size_t cache_hits_after = allocator_->stats().cache_hit_count;
    EXPECT_GT(cache_hits_after, cache_hits_before);

    allocator_->deallocate(block2);
}

TEST_F(AllocatorTest, CacheMissForNewSize) {
    // Allocate and deallocate size 1024
    AllocatedBlock block1 = allocator_->allocate(1024, StorageMode::Shared);
    allocator_->deallocate(block1);

    size_t cache_misses_before = allocator_->stats().cache_miss_count;

    // Allocate different size - should miss cache
    AllocatedBlock block2 = allocator_->allocate(2048, StorageMode::Shared);

    size_t cache_misses_after = allocator_->stats().cache_miss_count;
    EXPECT_GT(cache_misses_after, cache_misses_before);

    allocator_->deallocate(block2);
}

// -- Cache management tests ---------------------------------------------------

TEST_F(AllocatorTest, EmptyCache) {
    // Allocate and deallocate to populate cache
    AllocatedBlock block = allocator_->allocate(1024, StorageMode::Shared);
    allocator_->deallocate(block);

    AllocatorStats stats_before = allocator_->stats();
    EXPECT_GT(stats_before.cached_bytes, 0u);

    allocator_->empty_cache();

    AllocatorStats stats_after = allocator_->stats();
    EXPECT_EQ(stats_after.cached_bytes, 0u);
}

TEST_F(AllocatorTest, EmptyCacheDoesNotAffectLiveAllocations) {
    AllocatedBlock block = allocator_->allocate(1024, StorageMode::Shared);
    size_t allocated_before = allocator_->stats().allocated_bytes;

    allocator_->empty_cache();

    size_t allocated_after = allocator_->stats().allocated_bytes;
    EXPECT_EQ(allocated_after, allocated_before);

    allocator_->deallocate(block);
}

// -- Make aliasable tests -----------------------------------------------------

TEST_F(AllocatorTest, MakeAliasable) {
    AllocatedBlock block = allocator_->allocate(1024, StorageMode::Shared);
    EXPECT_FALSE(block.aliasable);

    allocator_->make_aliasable(block);
    EXPECT_TRUE(block.aliasable);

    allocator_->deallocate(block);
}

// -- Statistics tests ---------------------------------------------------------

TEST_F(AllocatorTest, StatsTrackAllocation) {
    AllocatorStats stats_before = allocator_->stats();

    AllocatedBlock block = allocator_->allocate(1024, StorageMode::Shared);

    AllocatorStats stats_after = allocator_->stats();
    EXPECT_GT(stats_after.allocated_bytes, stats_before.allocated_bytes);
    EXPECT_GT(stats_after.allocation_count, stats_before.allocation_count);

    allocator_->deallocate(block);
}

TEST_F(AllocatorTest, StatsTrackDeallocation) {
    AllocatedBlock block = allocator_->allocate(1024, StorageMode::Shared);
    AllocatorStats stats_before = allocator_->stats();

    allocator_->deallocate(block);

    AllocatorStats stats_after = allocator_->stats();
    EXPECT_LT(stats_after.allocated_bytes, stats_before.allocated_bytes);
    EXPECT_LT(stats_after.allocation_count, stats_before.allocation_count);
}

TEST_F(AllocatorTest, StatsPeakBytes) {
    AllocatedBlock block1 = allocator_->allocate(1024, StorageMode::Shared);
    AllocatedBlock block2 = allocator_->allocate(2048, StorageMode::Shared);

    AllocatorStats stats = allocator_->stats();
    size_t peak = stats.peak_bytes;
    EXPECT_GE(peak, 3072u);  // At least 1024 + 2048

    allocator_->deallocate(block1);
    allocator_->deallocate(block2);

    // Peak should remain even after deallocation
    stats = allocator_->stats();
    EXPECT_EQ(stats.peak_bytes, peak);
}

TEST_F(AllocatorTest, StatsCacheHitRate) {
    // Initial state
    AllocatorStats initial = allocator_->stats();

    // Allocate and deallocate to populate cache
    AllocatedBlock block1 = allocator_->allocate(1024, StorageMode::Shared);
    allocator_->deallocate(block1);

    // Allocate again - should hit cache
    AllocatedBlock block2 = allocator_->allocate(1024, StorageMode::Shared);
    allocator_->deallocate(block2);

    AllocatorStats final = allocator_->stats();
    EXPECT_GT(final.cache_hit_count, initial.cache_hit_count);
}

// -- Size class tests ---------------------------------------------------------

TEST_F(AllocatorTest, SizeClassRounding) {
    AllocatedBlock block = allocator_->allocate(100, StorageMode::Shared);

    // Size should be rounded up to power-of-2 class
    EXPECT_GE(block.size, 100u);
    EXPECT_GT(block.size_class, 0u);

    allocator_->deallocate(block);
}

TEST_F(AllocatorTest, MultipleSizeClasses) {
    AllocatedBlock small = allocator_->allocate(64, StorageMode::Shared);
    AllocatedBlock medium = allocator_->allocate(1024, StorageMode::Shared);
    AllocatedBlock large = allocator_->allocate(65536, StorageMode::Shared);

    // Different sizes should have different size classes
    EXPECT_NE(small.size_class, medium.size_class);
    EXPECT_NE(medium.size_class, large.size_class);

    allocator_->deallocate(small);
    allocator_->deallocate(medium);
    allocator_->deallocate(large);
}
