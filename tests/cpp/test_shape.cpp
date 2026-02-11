/// @file test_shape.cpp
/// @brief Unit tests for MNShape class.

#include <gtest/gtest.h>
#include "metal_native/core/shape.h"
#include "metal_native/core/error.h"

using namespace metal_native;

// -- Construction tests -------------------------------------------------------

TEST(MNShapeTest, DefaultConstructorCreatesScalar) {
    MNShape shape;
    EXPECT_EQ(shape.ndim(), 0u);
    EXPECT_EQ(shape.numel(), 1);
}

TEST(MNShapeTest, ConstructFromInitializerList) {
    MNShape shape{2, 3, 4};
    EXPECT_EQ(shape.ndim(), 3u);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
}

TEST(MNShapeTest, ConstructFromVector) {
    std::vector<int64_t> dims = {5, 10};
    MNShape shape(dims);
    EXPECT_EQ(shape.ndim(), 2u);
    EXPECT_EQ(shape[0], 5);
    EXPECT_EQ(shape[1], 10);
}

TEST(MNShapeTest, ConstructFromMoveVector) {
    std::vector<int64_t> dims = {7, 8, 9};
    MNShape shape(std::move(dims));
    EXPECT_EQ(shape.ndim(), 3u);
    EXPECT_EQ(shape[0], 7);
    EXPECT_EQ(shape[1], 8);
    EXPECT_EQ(shape[2], 9);
}

TEST(MNShapeTest, SingleDimension) {
    MNShape shape{42};
    EXPECT_EQ(shape.ndim(), 1u);
    EXPECT_EQ(shape[0], 42);
    EXPECT_EQ(shape.numel(), 42);
}

// -- Numel tests --------------------------------------------------------------

TEST(MNShapeTest, NumelProduct) {
    MNShape shape{2, 3, 4};
    EXPECT_EQ(shape.numel(), 24);
}

TEST(MNShapeTest, NumelWithOneDim) {
    MNShape shape{1, 3, 1};
    EXPECT_EQ(shape.numel(), 3);
}

TEST(MNShapeTest, NumelScalar) {
    MNShape shape;
    EXPECT_EQ(shape.numel(), 1);
}

// -- Indexing tests -----------------------------------------------------------

TEST(MNShapeTest, NegativeIndexing) {
    MNShape shape{2, 3, 4};
    EXPECT_EQ(shape[-1], 4);
    EXPECT_EQ(shape[-2], 3);
    EXPECT_EQ(shape[-3], 2);
}

// -- Comparison tests ---------------------------------------------------------

TEST(MNShapeTest, EqualShapes) {
    MNShape a{2, 3, 4};
    MNShape b{2, 3, 4};
    EXPECT_EQ(a, b);
}

TEST(MNShapeTest, UnequalShapes) {
    MNShape a{2, 3, 4};
    MNShape b{2, 3, 5};
    EXPECT_NE(a, b);
}

TEST(MNShapeTest, DifferentRankNotEqual) {
    MNShape a{2, 3};
    MNShape b{2, 3, 1};
    EXPECT_NE(a, b);
}

// -- Copy/Move tests ----------------------------------------------------------

TEST(MNShapeTest, CopyConstructor) {
    MNShape original{2, 3, 4};
    MNShape copy(original);
    EXPECT_EQ(copy, original);
    EXPECT_EQ(copy.ndim(), 3u);
}

TEST(MNShapeTest, CopyAssignment) {
    MNShape original{2, 3, 4};
    MNShape copy;
    copy = original;
    EXPECT_EQ(copy, original);
}

TEST(MNShapeTest, MoveConstructor) {
    MNShape original{2, 3, 4};
    MNShape moved(std::move(original));
    EXPECT_EQ(moved.ndim(), 3u);
    EXPECT_EQ(moved[0], 2);
}

// -- Stride tests -------------------------------------------------------------

TEST(MNShapeTest, ContiguousStrides) {
    MNShape shape{2, 3, 4};
    auto strides = shape.contiguous_strides();
    ASSERT_EQ(strides.size(), 3u);
    EXPECT_EQ(strides[0], 12);  // 3 * 4
    EXPECT_EQ(strides[1], 4);   // 4
    EXPECT_EQ(strides[2], 1);   // 1
}

TEST(MNShapeTest, ContiguousStridesScalar) {
    MNShape shape;
    auto strides = shape.contiguous_strides();
    EXPECT_EQ(strides.size(), 0u);
}

TEST(MNShapeTest, IsContiguous) {
    MNShape shape{2, 3, 4};
    auto strides = shape.contiguous_strides();
    EXPECT_TRUE(shape.is_contiguous(strides));
}

TEST(MNShapeTest, NonContiguousStrides) {
    MNShape shape{2, 3, 4};
    std::vector<int64_t> bad_strides = {100, 4, 1};
    EXPECT_FALSE(shape.is_contiguous(bad_strides));
}

// -- Broadcast tests ----------------------------------------------------------

TEST(MNShapeTest, BroadcastSameShape) {
    MNShape a{2, 3, 4};
    MNShape b{2, 3, 4};
    MNShape result = a.broadcast_with(b);
    EXPECT_EQ(result, MNShape({2, 3, 4}));
}

TEST(MNShapeTest, BroadcastWithOne) {
    MNShape a{2, 3, 4};
    MNShape b{1, 1, 4};
    MNShape result = a.broadcast_with(b);
    EXPECT_EQ(result, MNShape({2, 3, 4}));
}

TEST(MNShapeTest, BroadcastDifferentRank) {
    MNShape a{2, 3, 4};
    MNShape b{4};
    MNShape result = a.broadcast_with(b);
    EXPECT_EQ(result, MNShape({2, 3, 4}));
}

TEST(MNShapeTest, BroadcastIncompatibleThrows) {
    MNShape a{2, 3, 4};
    MNShape b{2, 5, 4};
    EXPECT_THROW(a.broadcast_with(b), MNException);
}

// -- ToString test ------------------------------------------------------------

TEST(MNShapeTest, ToString) {
    MNShape shape{2, 3, 4};
    std::string str = shape.to_string();
    EXPECT_FALSE(str.empty());
    // Should contain the dimensions
    EXPECT_NE(str.find("2"), std::string::npos);
    EXPECT_NE(str.find("3"), std::string::npos);
    EXPECT_NE(str.find("4"), std::string::npos);
}

// -- Data access test ---------------------------------------------------------

TEST(MNShapeTest, DataPointer) {
    MNShape shape{2, 3, 4};
    const int64_t* data = shape.data();
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data[0], 2);
    EXPECT_EQ(data[1], 3);
    EXPECT_EQ(data[2], 4);
}

TEST(MNShapeTest, DimsReference) {
    MNShape shape{2, 3, 4};
    const auto& dims = shape.dims();
    ASSERT_EQ(dims.size(), 3u);
    EXPECT_EQ(dims[0], 2);
}
