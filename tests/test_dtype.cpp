/// @file test_dtype.cpp
/// @brief Unit tests for MNDType enum and utilities.

#include <gtest/gtest.h>
#include "metal_native/core/dtype.h"

using namespace metal_native;

// -- dtype_size tests ---------------------------------------------------------

TEST(MNDTypeTest, Float32Size) {
    EXPECT_EQ(dtype_size(MNDType::Float32), 4u);
}

TEST(MNDTypeTest, Float16Size) {
    EXPECT_EQ(dtype_size(MNDType::Float16), 2u);
}

TEST(MNDTypeTest, BFloat16Size) {
    EXPECT_EQ(dtype_size(MNDType::BFloat16), 2u);
}

TEST(MNDTypeTest, Int64Size) {
    EXPECT_EQ(dtype_size(MNDType::Int64), 8u);
}

TEST(MNDTypeTest, Int32Size) {
    EXPECT_EQ(dtype_size(MNDType::Int32), 4u);
}

TEST(MNDTypeTest, Int16Size) {
    EXPECT_EQ(dtype_size(MNDType::Int16), 2u);
}

TEST(MNDTypeTest, Int8Size) {
    EXPECT_EQ(dtype_size(MNDType::Int8), 1u);
}

TEST(MNDTypeTest, UInt8Size) {
    EXPECT_EQ(dtype_size(MNDType::UInt8), 1u);
}

TEST(MNDTypeTest, BoolSize) {
    EXPECT_EQ(dtype_size(MNDType::Bool), 1u);
}

// -- dtype_name tests ---------------------------------------------------------

TEST(MNDTypeTest, Float32Name) {
    EXPECT_STREQ(dtype_name(MNDType::Float32), "float32");
}

TEST(MNDTypeTest, Float16Name) {
    EXPECT_STREQ(dtype_name(MNDType::Float16), "float16");
}

TEST(MNDTypeTest, BFloat16Name) {
    EXPECT_STREQ(dtype_name(MNDType::BFloat16), "bfloat16");
}

TEST(MNDTypeTest, Int64Name) {
    EXPECT_STREQ(dtype_name(MNDType::Int64), "int64");
}

TEST(MNDTypeTest, Int32Name) {
    EXPECT_STREQ(dtype_name(MNDType::Int32), "int32");
}

TEST(MNDTypeTest, Int16Name) {
    EXPECT_STREQ(dtype_name(MNDType::Int16), "int16");
}

TEST(MNDTypeTest, Int8Name) {
    EXPECT_STREQ(dtype_name(MNDType::Int8), "int8");
}

TEST(MNDTypeTest, UInt8Name) {
    EXPECT_STREQ(dtype_name(MNDType::UInt8), "uint8");
}

TEST(MNDTypeTest, BoolName) {
    EXPECT_STREQ(dtype_name(MNDType::Bool), "bool");
}

// -- dtype_is_floating_point tests --------------------------------------------

TEST(MNDTypeTest, FloatingPointTypes) {
    EXPECT_TRUE(dtype_is_floating_point(MNDType::Float32));
    EXPECT_TRUE(dtype_is_floating_point(MNDType::Float16));
    EXPECT_TRUE(dtype_is_floating_point(MNDType::BFloat16));
}

TEST(MNDTypeTest, NonFloatingPointTypes) {
    EXPECT_FALSE(dtype_is_floating_point(MNDType::Int64));
    EXPECT_FALSE(dtype_is_floating_point(MNDType::Int32));
    EXPECT_FALSE(dtype_is_floating_point(MNDType::Int16));
    EXPECT_FALSE(dtype_is_floating_point(MNDType::Int8));
    EXPECT_FALSE(dtype_is_floating_point(MNDType::UInt8));
    EXPECT_FALSE(dtype_is_floating_point(MNDType::Bool));
}

// -- dtype_is_integer tests ---------------------------------------------------

TEST(MNDTypeTest, IntegerTypes) {
    EXPECT_TRUE(dtype_is_integer(MNDType::Int64));
    EXPECT_TRUE(dtype_is_integer(MNDType::Int32));
    EXPECT_TRUE(dtype_is_integer(MNDType::Int16));
    EXPECT_TRUE(dtype_is_integer(MNDType::Int8));
    EXPECT_TRUE(dtype_is_integer(MNDType::UInt8));
}

TEST(MNDTypeTest, NonIntegerTypes) {
    EXPECT_FALSE(dtype_is_integer(MNDType::Float32));
    EXPECT_FALSE(dtype_is_integer(MNDType::Float16));
    EXPECT_FALSE(dtype_is_integer(MNDType::BFloat16));
    EXPECT_FALSE(dtype_is_integer(MNDType::Bool));
}

// -- dtype_is_signed tests ----------------------------------------------------

TEST(MNDTypeTest, SignedTypes) {
    EXPECT_TRUE(dtype_is_signed(MNDType::Float32));
    EXPECT_TRUE(dtype_is_signed(MNDType::Float16));
    EXPECT_TRUE(dtype_is_signed(MNDType::BFloat16));
    EXPECT_TRUE(dtype_is_signed(MNDType::Int64));
    EXPECT_TRUE(dtype_is_signed(MNDType::Int32));
    EXPECT_TRUE(dtype_is_signed(MNDType::Int16));
    EXPECT_TRUE(dtype_is_signed(MNDType::Int8));
}

TEST(MNDTypeTest, UnsignedTypes) {
    EXPECT_FALSE(dtype_is_signed(MNDType::UInt8));
    EXPECT_FALSE(dtype_is_signed(MNDType::Bool));
}

// -- Metal format tests -------------------------------------------------------

TEST(MNDTypeTest, MetalVertexFormatFloat32) {
    uint32_t format = dtype_to_mtl_vertex_format(MNDType::Float32);
    // Should return a valid format (non-zero for float32)
    EXPECT_NE(format, 0u);
}

TEST(MNDTypeTest, MetalVertexFormatBool) {
    uint32_t format = dtype_to_mtl_vertex_format(MNDType::Bool);
    // Bool may not have a vertex format mapping
    // Just ensure it doesn't crash
    (void)format;
}
