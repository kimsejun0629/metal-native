#pragma once

/// @file shape_bucketing.h
/// @brief Shape bucketing for reducing MPSGraph recompilation overhead.
///
/// Dynamic shapes cause MPSGraph to recompile for every unique shape tuple,
/// which can be slow for models with variable batch sizes or sequence lengths.
/// Shape bucketing reduces recompilation by rounding shapes up to a small
/// set of predefined bucket sizes, allowing the same compiled graph to be
/// reused for multiple similar shapes.
///
/// Bucket values: 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536,
///                2048, 4096, 8192, 16384.
///
/// Pure C++ implementation (no Metal dependencies).

#include <cstddef>
#include <cstdint>
#include <vector>

#include "metal_native/core/shape.h"

namespace metal_native {

// ---------------------------------------------------------------------------
// Bucket size helpers
// ---------------------------------------------------------------------------

/// Round a size up to the nearest bucket value.
///
/// If the size is already a bucket value, it is returned unchanged.
/// If the size exceeds the largest bucket (16384), it is returned unchanged.
///
/// @param size  The actual size to bucket.
/// @return      The bucketed size (>= size).
size_t bucket_size(size_t size) noexcept;

/// Return the predefined bucket values as a vector.
///
/// The returned vector is sorted in ascending order.
std::vector<size_t> bucket_values() noexcept;

// ---------------------------------------------------------------------------
// Shape bucketing
// ---------------------------------------------------------------------------

/// Shape bucketing utility for reducing graph recompilation.
class ShapeBucketer {
public:
    ShapeBucketer() = default;
    ~ShapeBucketer() = default;

    // -- Bucketing -----------------------------------------------------------

    /// Bucket an entire shape by rounding each dimension up to the nearest
    /// bucket value.
    ///
    /// @param shape  The original shape.
    /// @return       A bucketed shape where each dimension is >= the original.
    MNShape bucket_shape(const MNShape& shape) const noexcept;

    /// Check whether the given shape requires bucketing (i.e., whether any
    /// dimension is not already a bucket value).
    ///
    /// @param shape  The shape to check.
    /// @return       True if bucketing would change the shape.
    bool needs_bucketing(const MNShape& shape) const noexcept;

    // -- Statistics ----------------------------------------------------------

    /// Return the number of unique bucket sizes.
    size_t bucket_count() const noexcept;

    /// Return the largest bucket value.
    size_t max_bucket_size() const noexcept;
};

// ---------------------------------------------------------------------------
// Tensor padding/slicing helpers
// ---------------------------------------------------------------------------

/// Pad a tensor from its actual shape to a bucketed shape.
///
/// This is a stub for future implementation. The actual padding operation
/// would be performed by MPSGraph operations (e.g., padWithConstant).
///
/// @param actual_shape    The original tensor shape.
/// @param bucketed_shape  The target bucketed shape.
/// @return                Opaque handle to the padded tensor (MPSGraphTensor*).
void* pad_tensor(void* tensor,
                 const MNShape& actual_shape,
                 const MNShape& bucketed_shape);

/// Slice a tensor from a bucketed shape back to the actual shape.
///
/// This is a stub for future implementation. The actual slicing operation
/// would be performed by MPSGraph slice operations.
///
/// @param bucketed_shape  The bucketed tensor shape.
/// @param actual_shape    The target actual shape.
/// @return                Opaque handle to the sliced tensor (MPSGraphTensor*).
void* slice_tensor(void* tensor,
                   const MNShape& bucketed_shape,
                   const MNShape& actual_shape);

} // namespace metal_native
