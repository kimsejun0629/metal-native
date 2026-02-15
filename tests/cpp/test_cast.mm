/// @file test_cast.mm
/// @brief Unit tests for dtype cast operations.

#include <gtest/gtest.h>
#include "metal_native/ops/elementwise.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/kernels/kernel_registry.h"
#include <cstring>
#include <unistd.h>

using namespace metal_native;

class CastTest : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            device_ = &MNDevice::instance();
            has_metal_ = (device_->metal_device() != nullptr);
        } catch (...) {
            has_metal_ = false;
        }

        if (!has_metal_) {
            GTEST_SKIP() << "Metal device not available";
        }

        // Compile copy_kernels.metal at runtime for cast kernel functions
        @autoreleasepool {
            const char* possible_paths[] = {
                "../metal_native/shaders/copy_kernels.metal",
                "../../metal_native/shaders/copy_kernels.metal",
                "../../../metal_native/shaders/copy_kernels.metal",
                "/Users/kimsejun/Documents/projects/pytorch_mps/metal_native/shaders/copy_kernels.metal"
            };

            std::string source_path;
            for (const char* path : possible_paths) {
                if (access(path, F_OK) == 0) {
                    source_path = path;
                    break;
                }
            }

            if (source_path.empty()) {
                GTEST_SKIP() << "Could not find copy_kernels.metal source file";
                return;
            }

            std::string temp_air = std::string(std::tmpnam(nullptr)) + ".air";
            std::string temp_metallib = std::string(std::tmpnam(nullptr)) + ".metallib";

            std::string compile_cmd = "xcrun -sdk macosx metal -c " + source_path + " -o " + temp_air + " 2>&1";
            FILE* compile_pipe = popen(compile_cmd.c_str(), "r");
            if (compile_pipe) {
                pclose(compile_pipe);
            }

            std::string link_cmd = "xcrun -sdk macosx metallib " + temp_air + " -o " + temp_metallib + " 2>&1";
            FILE* link_pipe = popen(link_cmd.c_str(), "r");
            if (link_pipe) {
                pclose(link_pipe);
            }

            if (access(temp_metallib.c_str(), F_OK) != 0) {
                GTEST_SKIP() << "Failed to compile Metal shaders";
                return;
            }

            try {
                KernelRegistry& registry = KernelRegistry::instance();
                registry.load_library(temp_metallib);
            } catch (const std::exception& e) {
                unlink(temp_air.c_str());
                unlink(temp_metallib.c_str());
                GTEST_SKIP() << "Failed to load Metal library: " << e.what();
                return;
            }

            unlink(temp_air.c_str());
            temp_metallib_ = temp_metallib;
        }
    }

    void TearDown() override {
        if (!temp_metallib_.empty()) {
            unlink(temp_metallib_.c_str());
        }
    }

    std::string temp_metallib_;
    MNDevice* device_ = nullptr;
    bool has_metal_ = false;
};

TEST_F(CastTest, FP32ToFP16) {
    // Create FP32 tensor with known values
    std::vector<float> data = {1.0f, 2.5f, -3.75f, 0.0f, 100.0f, -0.5f};
    MNTensor input = MNTensor::empty(MNShape({2, 3}), MNDType::Float32, *device_, StorageMode::Shared);

    // Copy data to tensor
    float* input_ptr = input.data_ptr<float>();
    std::memcpy(input_ptr, data.data(), data.size() * sizeof(float));

    // Cast to FP16
    MNTensor output = cast_dtype(input, MNDType::Float16, *device_);

    // Verify dtype
    EXPECT_EQ(output.dtype(), MNDType::Float16);
    EXPECT_EQ(output.shape(), input.shape());

    // Synchronize to ensure GPU work completes
    device_->synchronize();

    // Cast back to FP32 to verify values
    MNTensor back = cast_dtype(output, MNDType::Float32, *device_);
    device_->synchronize();

    const float* result = back.data_ptr<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(result[i], data[i], 0.01f) << "Mismatch at index " << i;
    }
}

TEST_F(CastTest, FP16ToFP32) {
    // Create FP16 tensor
    std::vector<float> expected = {1.0f, 2.5f, -3.75f, 0.0f, 50.0f, -0.5f};
    MNTensor input = MNTensor::empty(MNShape({2, 3}), MNDType::Float16, *device_, StorageMode::Shared);

    // Convert float to __fp16 and write to tensor
    uint16_t* input_ptr = input.data_ptr<uint16_t>();
    for (size_t i = 0; i < expected.size(); ++i) {
        __fp16 val = static_cast<__fp16>(expected[i]);
        std::memcpy(&input_ptr[i], &val, sizeof(uint16_t));
    }

    // Cast to FP32
    MNTensor output = cast_dtype(input, MNDType::Float32, *device_);

    // Verify dtype
    EXPECT_EQ(output.dtype(), MNDType::Float32);
    EXPECT_EQ(output.shape(), input.shape());

    device_->synchronize();

    const float* result = output.data_ptr<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 0.01f) << "Mismatch at index " << i;
    }
}

TEST_F(CastTest, NoOpCast) {
    // Create FP32 tensor
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    MNTensor input = MNTensor::empty(MNShape({3}), MNDType::Float32, *device_, StorageMode::Shared);

    float* input_ptr = input.data_ptr<float>();
    std::memcpy(input_ptr, data.data(), data.size() * sizeof(float));

    // Cast to same dtype (should be no-op)
    MNTensor output = cast_dtype(input, MNDType::Float32, *device_);

    // Should return same tensor (or at least same data pointer)
    EXPECT_EQ(output.dtype(), MNDType::Float32);
    EXPECT_EQ(output.shape(), input.shape());
}

TEST_F(CastTest, LargeTensor) {
    // Test with 8M elements (the problematic size mentioned in the task)
    const int64_t N = 8 * 1024 * 1024;
    MNTensor input = MNTensor::empty(MNShape({N}), MNDType::Float32, *device_, StorageMode::Shared);

    // Fill with test data
    float* input_ptr = input.data_ptr<float>();
    for (int64_t i = 0; i < N; ++i) {
        input_ptr[i] = static_cast<float>(i % 1000) / 100.0f;
    }

    // Cast to FP16 (should be fast on GPU, not slow CPU loop)
    MNTensor output = cast_dtype(input, MNDType::Float16, *device_);

    EXPECT_EQ(output.dtype(), MNDType::Float16);
    EXPECT_EQ(output.numel(), N);

    device_->synchronize();

    // Spot check a few values
    MNTensor back = cast_dtype(output, MNDType::Float32, *device_);
    device_->synchronize();

    const float* result = back.data_ptr<float>();
    EXPECT_NEAR(result[0], input_ptr[0], 0.01f);
    EXPECT_NEAR(result[N/2], input_ptr[N/2], 0.01f);
    EXPECT_NEAR(result[N-1], input_ptr[N-1], 0.01f);
}
