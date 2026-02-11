/// @file test_kernel_registry.cpp
/// @brief Unit tests for KernelRegistry precompilation.

#include <gtest/gtest.h>
#include "metal_native/kernels/kernel_registry.h"
#include "metal_native/core/device.h"

#include <thread>
#include <chrono>

using namespace metal_native;

// Test fixture for kernel registry tests
class KernelRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure device is initialized
        MNDevice::instance();
    }
};

// -- Basic functionality tests ------------------------------------------------

TEST_F(KernelRegistryTest, SingletonAccess) {
    KernelRegistry& reg1 = KernelRegistry::instance();
    KernelRegistry& reg2 = KernelRegistry::instance();

    // Should return the same instance
    EXPECT_EQ(&reg1, &reg2);
}

TEST_F(KernelRegistryTest, RegisterKernel) {
    KernelRegistry& registry = KernelRegistry::instance();

    size_t initial_size = registry.size();
    registry.register_kernel("test_kernel_unique_name", "test_function");

    EXPECT_EQ(registry.size(), initial_size + 1);
}

// -- Precompilation tests -----------------------------------------------------

TEST_F(KernelRegistryTest, PrecompileWithoutLibrary) {
    KernelRegistry& registry = KernelRegistry::instance();

    // Should not crash when no library is loaded
    EXPECT_NO_THROW(registry.precompile_pipelines());
}

TEST_F(KernelRegistryTest, PrecompileAfterRegistration) {
    KernelRegistry& registry = KernelRegistry::instance();

    // Register a few test kernels
    registry.register_kernel("precompile_test_1", "elementwise_add_float");
    registry.register_kernel("precompile_test_2", "elementwise_mul_float");

    // Precompile should not throw even if some kernels don't exist
    EXPECT_NO_THROW(registry.precompile_pipelines());
}

// Note: Testing actual pipeline compilation requires:
// 1. A valid Metal library to be loaded
// 2. Valid Metal shader functions
// 3. Proper Metal device initialization
// These are integration tests better suited for runtime verification.
