/// @file test_dequant_matmul.mm
/// @brief Unit tests for INT4/INT8 dequantization + matmul operations.

#include <gtest/gtest.h>
#include "metal_native/core/device.h"
#include "metal_native/core/tensor.h"
#include "metal_native/ops/matmul.h"
#include "metal_native/dispatch/command_pipeline.h"
#include "metal_native/kernels/kernel_registry.h"
#include <cmath>
#include <vector>

using namespace metal_native;

class DequantMatmulTest : public ::testing::Test {
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

        // Compile Metal shaders at runtime using system commands
        // This is the most reliable way for tests
        @autoreleasepool {
            // Try multiple paths to find the shader source
            const char* possible_paths[] = {
                "../metal_native/shaders/matmul_dequant_kernel.metal",
                "../../metal_native/shaders/matmul_dequant_kernel.metal",
                "../../../metal_native/shaders/matmul_dequant_kernel.metal",
                "/Users/kimsejun/Documents/projects/pytorch_mps/metal_native/shaders/matmul_dequant_kernel.metal"
            };

            std::string source_path;
            for (const char* path : possible_paths) {
                if (access(path, F_OK) == 0) {
                    source_path = path;
                    break;
                }
            }

            if (source_path.empty()) {
                GTEST_SKIP() << "Could not find matmul_dequant_kernel.metal source file";
                return;
            }

            // Create temporary files for compilation
            std::string temp_air = std::string(std::tmpnam(nullptr)) + ".air";
            std::string temp_metallib = std::string(std::tmpnam(nullptr)) + ".metallib";

            // Compile .metal -> .air
            std::string compile_cmd = "xcrun -sdk macosx metal -c " + source_path + " -o " + temp_air + " 2>&1";
            FILE* compile_pipe = popen(compile_cmd.c_str(), "r");
            if (compile_pipe) {
                pclose(compile_pipe);
            }

            // Link .air -> .metallib
            std::string link_cmd = "xcrun -sdk macosx metallib " + temp_air + " -o " + temp_metallib + " 2>&1";
            FILE* link_pipe = popen(link_cmd.c_str(), "r");
            if (link_pipe) {
                pclose(link_pipe);
            }

            // Check if metallib was created
            if (access(temp_metallib.c_str(), F_OK) != 0) {
                GTEST_SKIP() << "Failed to compile Metal shaders";
                return;
            }

            // Load the compiled library into KernelRegistry
            try {
                KernelRegistry& registry = KernelRegistry::instance();
                registry.load_library(temp_metallib);
            } catch (const std::exception& e) {
                // Clean up temp files
                unlink(temp_air.c_str());
                unlink(temp_metallib.c_str());
                GTEST_SKIP() << "Failed to load Metal library: " << e.what();
                return;
            }

            // Clean up temp files
            unlink(temp_air.c_str());
            temp_metallib_ = temp_metallib; // Keep for cleanup in TearDown
        }
    }

    void TearDown() override {
        if (!temp_metallib_.empty()) {
            unlink(temp_metallib_.c_str());
        }
    }

    std::string temp_metallib_;

    // Helper: Pack pairs of INT4 values into bytes (low nibble first, then high nibble)
    std::vector<uint8_t> pack_int4(const std::vector<int8_t>& values) {
        std::vector<uint8_t> packed;
        packed.reserve((values.size() + 1) / 2);

        for (size_t i = 0; i < values.size(); i += 2) {
            uint8_t low = static_cast<uint8_t>(values[i] & 0x0F);
            uint8_t high = (i + 1 < values.size()) ?
                static_cast<uint8_t>(values[i + 1] & 0x0F) : 0;
            packed.push_back(low | (high << 4));
        }

        return packed;
    }

    // Helper: CPU reference INT4 dequantization
    std::vector<float> dequant_reference_int4(
        const std::vector<uint8_t>& packed,
        const std::vector<float>& scales,
        const std::vector<float>& zeros,
        uint32_t K,
        uint32_t N,
        uint32_t group_size)
    {
        std::vector<float> dequantized(N * K);
        uint32_t num_groups_per_k = (K + group_size - 1) / group_size;

        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t k = 0; k < K; ++k) {
                uint32_t packed_idx = n * ((K + 1) / 2) + k / 2;
                uint8_t packed_byte = packed[packed_idx];

                // Extract nibble (low or high based on k % 2)
                int8_t int4_val;
                if (k % 2 == 0) {
                    int4_val = static_cast<int8_t>(packed_byte & 0x0F);
                } else {
                    int4_val = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                }

                // Convert to signed: subtract 8 for zero-centered [-8, 7]
                uint32_t group_idx = k / group_size;
                float scale = scales[n * num_groups_per_k + group_idx];
                float zero = zeros[n * num_groups_per_k + group_idx];

                // Dequantize: (int4_val - 8 - zero) * scale
                float dequant_val = (static_cast<float>(int4_val) - 8.0f) * scale - zero;

                // Store in [N, K] layout
                dequantized[n * K + k] = dequant_val;
            }
        }

        return dequantized;
    }

    // Helper: CPU reference INT8 dequantization
    std::vector<float> dequant_reference_int8(
        const std::vector<int8_t>& data,
        const std::vector<float>& scales,
        const std::vector<float>& zeros,
        uint32_t K,
        uint32_t N,
        uint32_t group_size)
    {
        std::vector<float> dequantized(N * K);
        uint32_t num_groups_per_k = (K + group_size - 1) / group_size;

        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t k = 0; k < K; ++k) {
                int8_t int8_val = data[n * K + k];

                uint32_t group_idx = k / group_size;
                float scale = scales[n * num_groups_per_k + group_idx];
                float zero = zeros[n * num_groups_per_k + group_idx];

                // Dequantize: int8_val * scale - zero
                float dequant_val = static_cast<float>(int8_val) * scale - zero;

                // Store in [N, K] layout
                dequantized[n * K + k] = dequant_val;
            }
        }

        return dequantized;
    }

    // Helper: CPU reference matmul (A: [M, K], B: [N, K] transposed layout, output: [M, N])
    std::vector<float> matmul_reference(
        const std::vector<float>& A,
        const std::vector<float>& B,
        uint32_t M,
        uint32_t N,
        uint32_t K)
    {
        std::vector<float> C(M * N, 0.0f);

        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < K; ++k) {
                    // A is row-major [M, K]
                    // B is row-major [N, K] (transposed form for dot product)
                    sum += A[m * K + k] * B[n * K + k];
                }
                C[m * N + n] = sum;
            }
        }

        return C;
    }

    // Helper: Convert FP16 to FP32
    static float fp16_to_fp32(uint16_t h) {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exponent = (h & 0x7C00) >> 10;
        uint32_t mantissa = (h & 0x03FF);

        if (exponent == 0) {
            if (mantissa == 0) {
                float zero = 0.0f;
                return sign ? -zero : zero;
            }
            // Subnormal
            return std::ldexp((float)mantissa / 1024.0f, -14) * (sign ? -1.0f : 1.0f);
        } else if (exponent == 31) {
            return (mantissa == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
        }

        uint32_t f32_exp = (exponent - 15 + 127) << 23;
        uint32_t f32_mantissa = mantissa << 13;
        uint32_t f32_bits = sign | f32_exp | f32_mantissa;
        float result;
        std::memcpy(&result, &f32_bits, sizeof(float));
        return result;
    }

    // Helper: Convert FP32 to FP16
    static uint16_t fp32_to_fp16(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(uint32_t));
        uint32_t sign = (bits & 0x80000000) >> 16;
        int32_t exponent = ((bits & 0x7F800000) >> 23) - 127 + 15;
        uint32_t mantissa = bits & 0x007FFFFF;

        if (exponent <= 0) {
            if (exponent < -10) return sign;  // Too small, flush to zero
            // Subnormal
            mantissa |= 0x00800000;
            uint16_t frac = mantissa >> (14 - exponent);
            return sign | frac;
        } else if (exponent >= 31) {
            return sign | 0x7C00;  // Infinity
        }

        return sign | (exponent << 10) | (mantissa >> 13);
    }

    MNDevice* device_ = nullptr;
    bool has_metal_ = false;
};

// ============================================================================
// INT4 Test Cases
// ============================================================================

TEST_F(DequantMatmulTest, INT4_SmallKnownAnswer) {
    // Simple case: M=1, N=2, K=4, group_size=32
    // Hand-compute expected values for verification

    const uint32_t M = 1;
    const uint32_t N = 2;
    const uint32_t K = 4;
    const uint32_t group_size = 32;

    // Activation: [1, 4] = [1.0, 2.0, 3.0, 4.0]
    std::vector<float> A_data = {1.0f, 2.0f, 3.0f, 4.0f};

    // Weights INT4: [2, 4] (N=2, K=4)
    // Row 0: [3, 5, 2, 7] -> after dequant with scale=0.1, zero=0: [0.3, 0.5, 0.2, 0.7]
    // Row 1: [1, 4, 6, 2] -> after dequant with scale=0.2, zero=0: [0.2, 0.8, 1.2, 0.4]
    std::vector<int8_t> B_int4 = {3, 5, 2, 7, 1, 4, 6, 2};
    std::vector<uint8_t> B_packed = pack_int4(B_int4);

    // Scales and zeros: [2, 1] (one group per row)
    std::vector<float> scales_data = {0.1f, 0.2f};
    std::vector<float> zeros_data = {0.0f, 0.0f};

    // Create tensors with proper FP16 conversion
    MNTensor A = MNTensor::empty(MNShape({M, K}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < A_data.size(); ++i) {
        A.data_ptr<uint16_t>()[i] = fp32_to_fp16(A_data[i]);
    }

    MNTensor B = MNTensor::empty(MNShape({N, K/2}), MNDType::UInt8, *device_, StorageMode::Shared);
    std::memcpy(B.data_ptr<uint8_t>(), B_packed.data(), B_packed.size());

    MNTensor scales = MNTensor::empty(MNShape({N, 1}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < scales_data.size(); ++i) {
        scales.data_ptr<uint16_t>()[i] = fp32_to_fp16(scales_data[i]);
    }

    MNTensor zeros = MNTensor::empty(MNShape({N, 1}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < zeros_data.size(); ++i) {
        zeros.data_ptr<uint16_t>()[i] = fp32_to_fp16(zeros_data[i]);
    }

    // Execute
    MNTensor result = dequant_matmul(A, B, scales, zeros, group_size, QuantType::INT4);

    // Expected (dequant formula: (int4_val - 8) * scale - zero):
    // Row 0: [3,5,2,7] with scale=0.1 -> [(3-8)*0.1, (5-8)*0.1, (2-8)*0.1, (7-8)*0.1] = [-0.5, -0.3, -0.6, -0.1]
    // Row 1: [1,4,6,2] with scale=0.2 -> [(1-8)*0.2, (4-8)*0.2, (6-8)*0.2, (2-8)*0.2] = [-1.4, -0.8, -0.4, -1.2]
    // C[0,0] = 1.0*(-0.5) + 2.0*(-0.3) + 3.0*(-0.6) + 4.0*(-0.1) = -0.5 - 0.6 - 1.8 - 0.4 = -3.3
    // C[0,1] = 1.0*(-1.4) + 2.0*(-0.8) + 3.0*(-0.4) + 4.0*(-1.2) = -1.4 - 1.6 - 1.2 - 4.8 = -9.0

    // Wait for GPU
    device_->command_pipeline().synchronize();

    // Convert FP16 results to float for comparison
    float result_0 = fp16_to_fp32(result.data_ptr<uint16_t>()[0]);
    float result_1 = fp16_to_fp32(result.data_ptr<uint16_t>()[1]);

    // Debug: print actual results
    std::cout << "Result[0,0] = " << result_0 << " (expected ~-3.3)" << std::endl;
    std::cout << "Result[0,1] = " << result_1 << " (expected ~-9.0)" << std::endl;

    // Check that we got non-zero results
    EXPECT_NE(result_0, 0.0f) << "Result should not be zero - kernel may not have executed";
    EXPECT_NE(result_1, 0.0f) << "Result should not be zero - kernel may not have executed";

    EXPECT_NEAR(result_0, -3.3f, 0.2f);
    EXPECT_NEAR(result_1, -9.0f, 0.2f);
}

TEST_F(DequantMatmulTest, INT4_SquareMatrix) {
    // M=16, N=16, K=32, group_size=32
    const uint32_t M = 16;
    const uint32_t N = 16;
    const uint32_t K = 32;
    const uint32_t group_size = 32;

    // Random-ish activations
    std::vector<float> A_data(M * K);
    for (size_t i = 0; i < A_data.size(); ++i) {
        A_data[i] = (static_cast<float>(i % 7) - 3.0f) * 0.5f;
    }
    // Random INT4 weights
    std::vector<int8_t> B_int4(N * K);
    for (size_t i = 0; i < B_int4.size(); ++i) {
        B_int4[i] = static_cast<int8_t>((i % 16));
    }
    std::vector<uint8_t> B_packed = pack_int4(B_int4);

    // Scales and zeros
    uint32_t num_groups = (K + group_size - 1) / group_size;
    std::vector<float> scales_data(N * num_groups, 0.05f);
    std::vector<float> zeros_data(N * num_groups, 0.0f);

    // Create tensors
    MNTensor A = MNTensor::empty(MNShape({M, K}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < A_data.size(); ++i) {
        A.data_ptr<uint16_t>()[i] = fp32_to_fp16(A_data[i]);
    }

    MNTensor B = MNTensor::empty(MNShape({N, K/2}), MNDType::UInt8, *device_, StorageMode::Shared);
    std::memcpy(B.data_ptr<uint8_t>(), B_packed.data(), B_packed.size());

    MNTensor scales = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < scales_data.size(); ++i) {
        scales.data_ptr<uint16_t>()[i] = fp32_to_fp16(scales_data[i]);
    }

    MNTensor zeros = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < zeros_data.size(); ++i) {
        zeros.data_ptr<uint16_t>()[i] = fp32_to_fp16(zeros_data[i]);
    }

    // Execute
    MNTensor result = dequant_matmul(A, B, scales, zeros, group_size, QuantType::INT4);
    device_->command_pipeline().synchronize();

    // CPU reference
    std::vector<float> B_dequant = dequant_reference_int4(B_packed, scales_data, zeros_data, K, N, group_size);
    std::vector<float> expected = matmul_reference(A_data, B_dequant, M, N, K);

    // Compare
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float gpu_result = fp16_to_fp32(result.data_ptr<uint16_t>()[m * N + n]);
            float cpu_result = expected[m * N + n];
            EXPECT_NEAR(gpu_result, cpu_result, 5e-2f)
                << "Mismatch at [" << m << "," << n << "]";
        }
    }
}

TEST_F(DequantMatmulTest, INT4_NonAlignedK) {
    // M=4, N=8, K=48 (not divisible by TILE_K=32)
    const uint32_t M = 4;
    const uint32_t N = 8;
    const uint32_t K = 48;
    const uint32_t group_size = 32;

    std::vector<float> A_data(M * K, 1.0f);
    std::vector<int8_t> B_int4(N * K, 5);
    std::vector<uint8_t> B_packed = pack_int4(B_int4);

    uint32_t num_groups = (K + group_size - 1) / group_size;
    std::vector<float> scales_data(N * num_groups, 0.1f);
    std::vector<float> zeros_data(N * num_groups, 0.0f);

    MNTensor A = MNTensor::empty(MNShape({M, K}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < A_data.size(); ++i) {
        A.data_ptr<uint16_t>()[i] = fp32_to_fp16(A_data[i]);
    }

    MNTensor B = MNTensor::empty(MNShape({N, K/2}), MNDType::UInt8, *device_, StorageMode::Shared);
    std::memcpy(B.data_ptr<uint8_t>(), B_packed.data(), B_packed.size());

    MNTensor scales = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < scales_data.size(); ++i) {
        scales.data_ptr<uint16_t>()[i] = fp32_to_fp16(scales_data[i]);
    }

    MNTensor zeros = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < zeros_data.size(); ++i) {
        zeros.data_ptr<uint16_t>()[i] = fp32_to_fp16(zeros_data[i]);
    }

    MNTensor result = dequant_matmul(A, B, scales, zeros, group_size, QuantType::INT4);
    device_->command_pipeline().synchronize();

    // Verify result is computed
    EXPECT_EQ(result.shape()[0], M);
    EXPECT_EQ(result.shape()[1], N);
}

TEST_F(DequantMatmulTest, INT4_NonAlignedN) {
    // M=4, N=48, K=32 (N not divisible by TILE_N=64)
    const uint32_t M = 4;
    const uint32_t N = 48;
    const uint32_t K = 32;
    const uint32_t group_size = 32;

    std::vector<float> A_data(M * K, 1.0f);
    std::vector<int8_t> B_int4(N * K, 3);
    std::vector<uint8_t> B_packed = pack_int4(B_int4);

    uint32_t num_groups = (K + group_size - 1) / group_size;
    std::vector<float> scales_data(N * num_groups, 0.1f);
    std::vector<float> zeros_data(N * num_groups, 0.0f);

    MNTensor A = MNTensor::empty(MNShape({M, K}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < A_data.size(); ++i) {
        A.data_ptr<uint16_t>()[i] = fp32_to_fp16(A_data[i]);
    }

    MNTensor B = MNTensor::empty(MNShape({N, K/2}), MNDType::UInt8, *device_, StorageMode::Shared);
    std::memcpy(B.data_ptr<uint8_t>(), B_packed.data(), B_packed.size());

    MNTensor scales = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < scales_data.size(); ++i) {
        scales.data_ptr<uint16_t>()[i] = fp32_to_fp16(scales_data[i]);
    }

    MNTensor zeros = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < zeros_data.size(); ++i) {
        zeros.data_ptr<uint16_t>()[i] = fp32_to_fp16(zeros_data[i]);
    }

    MNTensor result = dequant_matmul(A, B, scales, zeros, group_size, QuantType::INT4);
    device_->command_pipeline().synchronize();

    EXPECT_EQ(result.shape()[0], M);
    EXPECT_EQ(result.shape()[1], N);
}

TEST_F(DequantMatmulTest, INT4_LargeMatrix) {
    // M=64, N=128, K=256, group_size=64
    const uint32_t M = 64;
    const uint32_t N = 128;
    const uint32_t K = 256;
    const uint32_t group_size = 64;

    std::vector<float> A_data(M * K);
    for (size_t i = 0; i < A_data.size(); ++i) {
        A_data[i] = (static_cast<float>(i % 11) - 5.0f) * 0.1f;
    }

    std::vector<int8_t> B_int4(N * K);
    for (size_t i = 0; i < B_int4.size(); ++i) {
        B_int4[i] = static_cast<int8_t>(i % 16);
    }
    std::vector<uint8_t> B_packed = pack_int4(B_int4);

    uint32_t num_groups = (K + group_size - 1) / group_size;
    std::vector<float> scales_data(N * num_groups, 0.05f);
    std::vector<float> zeros_data(N * num_groups, 0.0f);

    MNTensor A = MNTensor::empty(MNShape({M, K}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < A_data.size(); ++i) {
        A.data_ptr<uint16_t>()[i] = fp32_to_fp16(A_data[i]);
    }

    MNTensor B = MNTensor::empty(MNShape({N, K/2}), MNDType::UInt8, *device_, StorageMode::Shared);
    std::memcpy(B.data_ptr<uint8_t>(), B_packed.data(), B_packed.size());

    MNTensor scales = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < scales_data.size(); ++i) {
        scales.data_ptr<uint16_t>()[i] = fp32_to_fp16(scales_data[i]);
    }

    MNTensor zeros = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < zeros_data.size(); ++i) {
        zeros.data_ptr<uint16_t>()[i] = fp32_to_fp16(zeros_data[i]);
    }

    MNTensor result = dequant_matmul(A, B, scales, zeros, group_size, QuantType::INT4);
    device_->command_pipeline().synchronize();

    EXPECT_EQ(result.shape()[0], M);
    EXPECT_EQ(result.shape()[1], N);
}

TEST_F(DequantMatmulTest, INT4_VecMat) {
    // M=1, N=64, K=128 (tests the vecmat variant)
    const uint32_t M = 1;
    const uint32_t N = 64;
    const uint32_t K = 128;
    const uint32_t group_size = 32;

    std::vector<float> A_data(M * K, 0.5f);
    std::vector<int8_t> B_int4(N * K, 4);
    std::vector<uint8_t> B_packed = pack_int4(B_int4);

    uint32_t num_groups = (K + group_size - 1) / group_size;
    std::vector<float> scales_data(N * num_groups, 0.1f);
    std::vector<float> zeros_data(N * num_groups, 0.0f);

    MNTensor A = MNTensor::empty(MNShape({M, K}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < A_data.size(); ++i) {
        A.data_ptr<uint16_t>()[i] = fp32_to_fp16(A_data[i]);
    }

    MNTensor B = MNTensor::empty(MNShape({N, K/2}), MNDType::UInt8, *device_, StorageMode::Shared);
    std::memcpy(B.data_ptr<uint8_t>(), B_packed.data(), B_packed.size());

    MNTensor scales = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < scales_data.size(); ++i) {
        scales.data_ptr<uint16_t>()[i] = fp32_to_fp16(scales_data[i]);
    }

    MNTensor zeros = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < zeros_data.size(); ++i) {
        zeros.data_ptr<uint16_t>()[i] = fp32_to_fp16(zeros_data[i]);
    }

    MNTensor result = dequant_matmul(A, B, scales, zeros, group_size, QuantType::INT4);
    device_->command_pipeline().synchronize();

    EXPECT_EQ(result.shape()[0], M);
    EXPECT_EQ(result.shape()[1], N);
}

TEST_F(DequantMatmulTest, INT4_VecMatBatch) {
    // M=4, N=64, K=128 (batch vecmat variant)
    const uint32_t M = 4;
    const uint32_t N = 64;
    const uint32_t K = 128;
    const uint32_t group_size = 32;

    std::vector<float> A_data(M * K, 0.5f);
    std::vector<int8_t> B_int4(N * K, 4);
    std::vector<uint8_t> B_packed = pack_int4(B_int4);

    uint32_t num_groups = (K + group_size - 1) / group_size;
    std::vector<float> scales_data(N * num_groups, 0.1f);
    std::vector<float> zeros_data(N * num_groups, 0.0f);

    MNTensor A = MNTensor::empty(MNShape({M, K}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < A_data.size(); ++i) {
        A.data_ptr<uint16_t>()[i] = fp32_to_fp16(A_data[i]);
    }

    MNTensor B = MNTensor::empty(MNShape({N, K/2}), MNDType::UInt8, *device_, StorageMode::Shared);
    std::memcpy(B.data_ptr<uint8_t>(), B_packed.data(), B_packed.size());

    MNTensor scales = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < scales_data.size(); ++i) {
        scales.data_ptr<uint16_t>()[i] = fp32_to_fp16(scales_data[i]);
    }

    MNTensor zeros = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < zeros_data.size(); ++i) {
        zeros.data_ptr<uint16_t>()[i] = fp32_to_fp16(zeros_data[i]);
    }

    MNTensor result = dequant_matmul(A, B, scales, zeros, group_size, QuantType::INT4);
    device_->command_pipeline().synchronize();

    EXPECT_EQ(result.shape()[0], M);
    EXPECT_EQ(result.shape()[1], N);
}

// ============================================================================
// INT8 Test Cases
// ============================================================================

TEST_F(DequantMatmulTest, INT8_SmallKnownAnswer) {
    const uint32_t M = 1;
    const uint32_t N = 2;
    const uint32_t K = 4;
    const uint32_t group_size = 32;

    std::vector<float> A_data = {1.0f, 2.0f, 3.0f, 4.0f};

    // INT8 weights: [2, 4]
    std::vector<int8_t> B_int8 = {10, 20, 30, 40, 5, 15, 25, 35};

    std::vector<float> scales_data = {0.1f, 0.1f};
    std::vector<float> zeros_data = {0.0f, 0.0f};

    MNTensor A = MNTensor::empty(MNShape({M, K}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < A_data.size(); ++i) {
        A.data_ptr<uint16_t>()[i] = fp32_to_fp16(A_data[i]);
    }

    MNTensor B = MNTensor::empty(MNShape({N, K}), MNDType::Int8, *device_, StorageMode::Shared);
    std::memcpy(B.data_ptr<int8_t>(), B_int8.data(), B_int8.size());

    MNTensor scales = MNTensor::empty(MNShape({N, 1}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < scales_data.size(); ++i) {
        scales.data_ptr<uint16_t>()[i] = fp32_to_fp16(scales_data[i]);
    }

    MNTensor zeros = MNTensor::empty(MNShape({N, 1}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < zeros_data.size(); ++i) {
        zeros.data_ptr<uint16_t>()[i] = fp32_to_fp16(zeros_data[i]);
    }

    MNTensor result = dequant_matmul(A, B, scales, zeros, group_size, QuantType::INT8);
    device_->command_pipeline().synchronize();

    EXPECT_EQ(result.shape()[0], M);
    EXPECT_EQ(result.shape()[1], N);
}

TEST_F(DequantMatmulTest, INT8_SquareMatrix) {
    const uint32_t M = 16;
    const uint32_t N = 16;
    const uint32_t K = 32;
    const uint32_t group_size = 32;

    std::vector<float> A_data(M * K);
    for (size_t i = 0; i < A_data.size(); ++i) {
        A_data[i] = (static_cast<float>(i % 7) - 3.0f) * 0.5f;
    }

    std::vector<int8_t> B_int8(N * K);
    for (size_t i = 0; i < B_int8.size(); ++i) {
        B_int8[i] = static_cast<int8_t>(static_cast<int>(i % 50) - 25);
    }

    uint32_t num_groups = (K + group_size - 1) / group_size;
    std::vector<float> scales_data(N * num_groups, 0.05f);
    std::vector<float> zeros_data(N * num_groups, 0.0f);

    MNTensor A = MNTensor::empty(MNShape({M, K}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < A_data.size(); ++i) {
        A.data_ptr<uint16_t>()[i] = fp32_to_fp16(A_data[i]);
    }

    MNTensor B = MNTensor::empty(MNShape({N, K}), MNDType::Int8, *device_, StorageMode::Shared);
    std::memcpy(B.data_ptr<int8_t>(), B_int8.data(), B_int8.size());

    MNTensor scales = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < scales_data.size(); ++i) {
        scales.data_ptr<uint16_t>()[i] = fp32_to_fp16(scales_data[i]);
    }

    MNTensor zeros = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
    for (size_t i = 0; i < zeros_data.size(); ++i) {
        zeros.data_ptr<uint16_t>()[i] = fp32_to_fp16(zeros_data[i]);
    }

    MNTensor result = dequant_matmul(A, B, scales, zeros, group_size, QuantType::INT8);
    device_->command_pipeline().synchronize();

    // CPU reference
    std::vector<float> B_dequant = dequant_reference_int8(B_int8, scales_data, zeros_data, K, N, group_size);
    std::vector<float> expected = matmul_reference(A_data, B_dequant, M, N, K);

    // Compare with looser tolerance for INT8
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float gpu_result = fp16_to_fp32(result.data_ptr<uint16_t>()[m * N + n]);
            float cpu_result = expected[m * N + n];
            EXPECT_NEAR(gpu_result, cpu_result, 1e-2f)
                << "Mismatch at [" << m << "," << n << "]";
        }
    }
}

TEST_F(DequantMatmulTest, GroupSize_Variants) {
    // Test group_size=32, 64, 128 with same M=16,N=32,K=128
    const uint32_t M = 16;
    const uint32_t N = 32;
    const uint32_t K = 128;

    std::vector<float> A_data(M * K, 1.0f);
    std::vector<int8_t> B_int4(N * K, 5);
    std::vector<uint8_t> B_packed = pack_int4(B_int4);

    for (uint32_t group_size : {32u, 64u, 128u}) {
        uint32_t num_groups = (K + group_size - 1) / group_size;
        std::vector<float> scales_data(N * num_groups, 0.1f);
        std::vector<float> zeros_data(N * num_groups, 0.0f);

        MNTensor A = MNTensor::empty(MNShape({M, K}), MNDType::Float16, *device_, StorageMode::Shared);
        for (size_t i = 0; i < A_data.size(); ++i) {
            A.data_ptr<uint16_t>()[i] = fp32_to_fp16(A_data[i]);
        }

        MNTensor B = MNTensor::empty(MNShape({N, K/2}), MNDType::UInt8, *device_, StorageMode::Shared);
        std::memcpy(B.data_ptr<uint8_t>(), B_packed.data(), B_packed.size());

        MNTensor scales = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
        for (size_t i = 0; i < scales_data.size(); ++i) {
            scales.data_ptr<uint16_t>()[i] = fp32_to_fp16(scales_data[i]);
        }

        MNTensor zeros = MNTensor::empty(MNShape({N, static_cast<int64_t>(num_groups)}), MNDType::Float16, *device_, StorageMode::Shared);
        for (size_t i = 0; i < zeros_data.size(); ++i) {
            zeros.data_ptr<uint16_t>()[i] = fp32_to_fp16(zeros_data[i]);
        }

        MNTensor result = dequant_matmul(A, B, scales, zeros, group_size, QuantType::INT4);
        device_->command_pipeline().synchronize();

        EXPECT_EQ(result.shape()[0], M) << "group_size=" << group_size;
        EXPECT_EQ(result.shape()[1], N) << "group_size=" << group_size;
    }
}
