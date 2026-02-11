/// @file test_buffer.cpp
/// @brief Unit tests for MNBuffer GPU buffer abstraction.

#include <gtest/gtest.h>
#include "metal_native/core/buffer.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

using namespace metal_native;

// Test fixture that checks for Metal device availability
class BufferTest : public ::testing::Test {
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
    }

    MNDevice* device_ = nullptr;
    bool has_metal_ = false;
};

// -- Buffer creation tests ----------------------------------------------------

TEST_F(BufferTest, CreateSharedBuffer) {
    MNBuffer buffer(*device_, 1024, StorageMode::Shared);
    EXPECT_EQ(buffer.size(), 1024u);
    EXPECT_EQ(buffer.storage_mode(), StorageMode::Shared);
}

TEST_F(BufferTest, CreatePrivateBuffer) {
    MNBuffer buffer(*device_, 2048, StorageMode::Private);
    EXPECT_EQ(buffer.size(), 2048u);
    EXPECT_EQ(buffer.storage_mode(), StorageMode::Private);
}

TEST_F(BufferTest, CreateSmallBuffer) {
    // Test small allocation (1 byte)
    MNBuffer buffer(*device_, 1, StorageMode::Shared);
    EXPECT_EQ(buffer.size(), 1u);
}

TEST_F(BufferTest, CreateLargeBuffer) {
    // Test larger allocation (1 MB)
    size_t size = 1024 * 1024;
    MNBuffer buffer(*device_, size, StorageMode::Shared);
    EXPECT_EQ(buffer.size(), size);
}

// -- Data pointer tests -------------------------------------------------------

TEST_F(BufferTest, SharedBufferHasValidDataPointer) {
    MNBuffer buffer(*device_, 1024, StorageMode::Shared);
    void* ptr = buffer.data();
    EXPECT_NE(ptr, nullptr);
}

TEST_F(BufferTest, PrivateBufferDataPointerIsNull) {
    MNBuffer buffer(*device_, 1024, StorageMode::Private);
    void* ptr = buffer.data();
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(BufferTest, SharedBufferReadWrite) {
    MNBuffer buffer(*device_, 256, StorageMode::Shared);
    void* ptr = buffer.data();
    ASSERT_NE(ptr, nullptr);

    // Write test pattern
    uint8_t* bytes = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < 256; ++i) {
        bytes[i] = static_cast<uint8_t>(i);
    }

    // Read back and verify
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_EQ(bytes[i], static_cast<uint8_t>(i));
    }
}

// -- GPU address tests --------------------------------------------------------

TEST_F(BufferTest, GPUAddressIsNonZero) {
    MNBuffer buffer(*device_, 1024, StorageMode::Shared);
    uint64_t gpu_addr = buffer.gpu_address();
    // GPU address should be non-zero for a valid buffer
    EXPECT_NE(gpu_addr, 0u);
}

TEST_F(BufferTest, DifferentBuffersHaveDifferentGPUAddresses) {
    MNBuffer buffer1(*device_, 1024, StorageMode::Shared);
    MNBuffer buffer2(*device_, 1024, StorageMode::Shared);

    uint64_t addr1 = buffer1.gpu_address();
    uint64_t addr2 = buffer2.gpu_address();

    EXPECT_NE(addr1, addr2);
}

// -- Metal buffer accessor tests ----------------------------------------------

TEST_F(BufferTest, MetalBufferIsValid) {
    MNBuffer buffer(*device_, 1024, StorageMode::Shared);
    void* metal_buf = buffer.metal_buffer();
    EXPECT_NE(metal_buf, nullptr);
}

// -- Move semantics tests -----------------------------------------------------

TEST_F(BufferTest, MoveConstructor) {
    MNBuffer buffer1(*device_, 1024, StorageMode::Shared);
    size_t original_size = buffer1.size();
    uint64_t original_addr = buffer1.gpu_address();

    MNBuffer buffer2(std::move(buffer1));

    EXPECT_EQ(buffer2.size(), original_size);
    EXPECT_EQ(buffer2.gpu_address(), original_addr);
}

TEST_F(BufferTest, MoveAssignment) {
    MNBuffer buffer1(*device_, 1024, StorageMode::Shared);
    size_t original_size = buffer1.size();
    uint64_t original_addr = buffer1.gpu_address();

    MNBuffer buffer2(*device_, 512, StorageMode::Shared);
    buffer2 = std::move(buffer1);

    EXPECT_EQ(buffer2.size(), original_size);
    EXPECT_EQ(buffer2.gpu_address(), original_addr);
}

// -- Storage mode tests -------------------------------------------------------

TEST_F(BufferTest, DefaultStorageModeIsShared) {
    MNBuffer buffer(*device_, 1024);
    EXPECT_EQ(buffer.storage_mode(), StorageMode::Shared);
}

TEST_F(BufferTest, StorageModePreserved) {
    MNBuffer shared_buffer(*device_, 1024, StorageMode::Shared);
    EXPECT_EQ(shared_buffer.storage_mode(), StorageMode::Shared);

    MNBuffer private_buffer(*device_, 1024, StorageMode::Private);
    EXPECT_EQ(private_buffer.storage_mode(), StorageMode::Private);
}

// -- Size tests ---------------------------------------------------------------

TEST_F(BufferTest, SizeMatchesAllocation) {
    size_t sizes[] = {16, 64, 256, 1024, 4096, 16384};

    for (size_t size : sizes) {
        MNBuffer buffer(*device_, size, StorageMode::Shared);
        EXPECT_EQ(buffer.size(), size);
    }
}
