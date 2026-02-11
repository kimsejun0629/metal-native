/// @file shape_bucketing.mm
/// @brief Objective-C++ implementation of shape bucketing with MPSGraph support.

#include "metal_native/graph/shape_bucketing.h"
#include "metal_native/core/error.h"

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>

namespace metal_native {

// ---------------------------------------------------------------------------
// Bucket values
// ---------------------------------------------------------------------------

/// Predefined bucket sizes (powers of 2 with 1.5x intermediate steps).
static constexpr std::array<size_t, 16> kBucketSizes = {
    32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384
};

size_t bucket_size(size_t size) noexcept {
    // If size is already at or above the largest bucket, return it unchanged.
    if (size >= kBucketSizes.back()) {
        return size;
    }

    // Binary search for the smallest bucket >= size.
    auto it = std::lower_bound(kBucketSizes.begin(), kBucketSizes.end(), size);
    if (it != kBucketSizes.end()) {
        return *it;
    }

    // Should never reach here due to the check above, but fallback to size.
    return size;
}

std::vector<size_t> bucket_values() noexcept {
    return std::vector<size_t>(kBucketSizes.begin(), kBucketSizes.end());
}

// ---------------------------------------------------------------------------
// ShapeBucketer
// ---------------------------------------------------------------------------

MNShape ShapeBucketer::bucket_shape(const MNShape& shape) const noexcept {
    std::vector<int64_t> bucketed_dims;
    bucketed_dims.reserve(shape.ndim());

    for (size_t i = 0; i < shape.ndim(); ++i) {
        int64_t dim = shape[static_cast<int64_t>(i)];
        if (dim <= 0) {
            // Preserve negative/zero dimensions (e.g., dynamic axes).
            bucketed_dims.push_back(dim);
        } else {
            bucketed_dims.push_back(static_cast<int64_t>(
                bucket_size(static_cast<size_t>(dim))
            ));
        }
    }

    return MNShape(std::move(bucketed_dims));
}

bool ShapeBucketer::needs_bucketing(const MNShape& shape) const noexcept {
    for (size_t i = 0; i < shape.ndim(); ++i) {
        int64_t dim = shape[static_cast<int64_t>(i)];
        if (dim > 0) {
            size_t udim = static_cast<size_t>(dim);
            if (bucket_size(udim) != udim) {
                return true;
            }
        }
    }
    return false;
}

size_t ShapeBucketer::bucket_count() const noexcept {
    return kBucketSizes.size();
}

size_t ShapeBucketer::max_bucket_size() const noexcept {
    return kBucketSizes.back();
}

// ---------------------------------------------------------------------------
// Tensor padding/slicing stubs
// ---------------------------------------------------------------------------

void* pad_tensor(void* tensor,
                 const MNShape& actual_shape,
                 const MNShape& bucketed_shape) {
    MN_CHECK(actual_shape.ndim() == bucketed_shape.ndim(),
             MetalNativeError::InvalidArgument,
             "pad_tensor: shape rank mismatch");

    for (size_t i = 0; i < actual_shape.ndim(); ++i) {
        int64_t actual_dim = actual_shape[static_cast<int64_t>(i)];
        int64_t bucketed_dim = bucketed_shape[static_cast<int64_t>(i)];
        MN_CHECK(bucketed_dim >= actual_dim,
                 MetalNativeError::InvalidArgument,
                 "pad_tensor: bucketed dimension must be >= actual dimension");
    }

    // Check if padding is needed
    bool needs_pad = false;
    for (size_t i = 0; i < actual_shape.ndim(); ++i) {
        if (actual_shape[static_cast<int64_t>(i)] != bucketed_shape[static_cast<int64_t>(i)]) {
            needs_pad = true;
            break;
        }
    }
    if (!needs_pad) {
        return tensor;  // No padding needed
    }

    @autoreleasepool {
        MPSGraphTensor* input_tensor = (__bridge MPSGraphTensor*)(tensor);
        MPSGraph* graph = input_tensor.operation.graph;

        // Build left and right padding arrays
        NSMutableArray<NSNumber*>* left_padding = [NSMutableArray arrayWithCapacity:actual_shape.ndim()];
        NSMutableArray<NSNumber*>* right_padding = [NSMutableArray arrayWithCapacity:actual_shape.ndim()];

        for (size_t i = 0; i < actual_shape.ndim(); ++i) {
            int64_t actual_dim = actual_shape[static_cast<int64_t>(i)];
            int64_t bucketed_dim = bucketed_shape[static_cast<int64_t>(i)];
            [left_padding addObject:@(0)];
            [right_padding addObject:@(bucketed_dim - actual_dim)];
        }

        MPSGraphTensor* padded = [graph padTensor:input_tensor
                                  withPaddingMode:MPSGraphPaddingModeZero
                                      leftPadding:left_padding
                                     rightPadding:right_padding
                                    constantValue:0.0
                                             name:nil];

        return (__bridge void*)(padded);
    }
}

void* slice_tensor(void* tensor,
                   const MNShape& bucketed_shape,
                   const MNShape& actual_shape) {
    MN_CHECK(actual_shape.ndim() == bucketed_shape.ndim(),
             MetalNativeError::InvalidArgument,
             "slice_tensor: shape rank mismatch");

    for (size_t i = 0; i < actual_shape.ndim(); ++i) {
        int64_t actual_dim = actual_shape[static_cast<int64_t>(i)];
        int64_t bucketed_dim = bucketed_shape[static_cast<int64_t>(i)];
        MN_CHECK(bucketed_dim >= actual_dim,
                 MetalNativeError::InvalidArgument,
                 "slice_tensor: bucketed dimension must be >= actual dimension");
    }

    // Check if slicing is needed
    bool needs_slice = false;
    for (size_t i = 0; i < actual_shape.ndim(); ++i) {
        if (actual_shape[static_cast<int64_t>(i)] != bucketed_shape[static_cast<int64_t>(i)]) {
            needs_slice = true;
            break;
        }
    }
    if (!needs_slice) {
        return tensor;  // No slicing needed
    }

    @autoreleasepool {
        MPSGraphTensor* input_tensor = (__bridge MPSGraphTensor*)(tensor);
        MPSGraph* graph = input_tensor.operation.graph;

        // Build start, end, stride arrays for slicing
        NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:actual_shape.ndim()];
        NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:actual_shape.ndim()];
        NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:actual_shape.ndim()];

        for (size_t i = 0; i < actual_shape.ndim(); ++i) {
            [starts addObject:@(0)];
            [ends addObject:@(actual_shape[static_cast<int64_t>(i)])];
            [strides addObject:@(1)];
        }

        MPSGraphTensor* sliced = [graph sliceTensor:input_tensor
                                          starts:starts
                                            ends:ends
                                         strides:strides
                                            name:nil];

        return (__bridge void*)(sliced);
    }
}

} // namespace metal_native
