#pragma once

/// @file tensor.h
/// @brief Multi-dimensional tensor backed by an MNBuffer.
///
/// MNTensor is the primary user-facing data container.  It carries a shared
/// reference to an MNBuffer (allowing views to alias the same storage), plus
/// shape, strides, dtype, and a byte offset into the buffer.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "metal_native/core/buffer.h"
#include "metal_native/core/dtype.h"
#include "metal_native/core/shape.h"

namespace metal_native {

class MNDevice;

class MNTensor {
public:
    // -- Construction (low-level) --------------------------------------------

    /// Construct a tensor from explicit components.
    ///
    /// @param buffer   Shared ownership of the backing GPU buffer.
    /// @param shape    Logical shape of the tensor.
    /// @param strides  Element strides (not byte strides).
    /// @param dtype    Scalar data type.
    /// @param offset   Byte offset into @p buffer where this tensor's data
    ///                 begins (used by views / slices).
    MNTensor(std::shared_ptr<MNBuffer> buffer,
             MNShape shape,
             std::vector<int64_t> strides,
             MNDType dtype,
             size_t offset = 0);

    // -- Static factories ----------------------------------------------------

    /// Allocate an uninitialised tensor.
    /// @param mode Storage mode (default: Shared for CPU-accessible tensors).
    ///             Use Private for GPU-only intermediate results.
    static MNTensor empty(const MNShape& shape, MNDType dtype,
                          MNDevice& device,
                          StorageMode mode = StorageMode::Shared);

    /// Allocate a zero-filled tensor.
    static MNTensor zeros(const MNShape& shape, MNDType dtype,
                          MNDevice& device);

    /// Allocate a tensor filled with ones.
    static MNTensor ones(const MNShape& shape, MNDType dtype,
                         MNDevice& device);

    // -- Element access ------------------------------------------------------

    /// Typed pointer to the first element of this tensor (accounts for offset).
    template <typename T>
    T* data_ptr() {
        return reinterpret_cast<T*>(
            static_cast<uint8_t*>(buffer_->data()) + offset_);
    }

    template <typename T>
    const T* data_ptr() const {
        return reinterpret_cast<const T*>(
            static_cast<const uint8_t*>(buffer_->data()) + offset_);
    }

    /// Raw (void*) pointer to the first element, adjusted for offset.
    void* raw_data() {
        return static_cast<uint8_t*>(buffer_->data()) + offset_;
    }

    const void* raw_data() const {
        return static_cast<const uint8_t*>(buffer_->data()) + offset_;
    }

    // -- Shape / layout queries ----------------------------------------------

    /// Logical shape.
    const MNShape& shape() const noexcept { return shape_; }

    /// Element strides (not byte strides).
    const std::vector<int64_t>& strides() const noexcept { return strides_; }

    /// Scalar data type.
    MNDType dtype() const noexcept { return dtype_; }

    /// Byte offset within the backing buffer.
    size_t offset() const noexcept { return offset_; }

    /// Total number of elements.
    int64_t numel() const noexcept { return shape_.numel(); }

    /// Number of dimensions.
    size_t ndim() const noexcept { return shape_.ndim(); }

    /// Total byte size of the data region (numel * element size).
    size_t nbytes() const noexcept;

    /// True when elements are laid out contiguously in memory (row-major).
    bool is_contiguous() const noexcept;

    // -- View operations (no data copy) --------------------------------------

    /// Return a tensor with the given shape that shares storage.
    /// The tensor must be contiguous.  One dimension may be -1 (inferred).
    /// Throws MNException(InvalidArgument) on failure.
    MNTensor reshape(const MNShape& new_shape) const;

    /// Return a view into a contiguous sub-range along dimension @p dim.
    /// @param dim    Dimension to slice.
    /// @param start  Start index (inclusive).
    /// @param end    End index (exclusive, -1 = extent of dim).
    MNTensor slice(int64_t dim, int64_t start, int64_t end) const;

    // -- Data operations -----------------------------------------------------

    /// Deep-copy this tensor into a new contiguous allocation.
    MNTensor clone() const;

    /// Fill every element with @p value (interpreted through the tensor's dtype).
    void fill_(double value);

    // -- Buffer access -------------------------------------------------------

    /// Shared pointer to the backing buffer (for alias tracking).
    const std::shared_ptr<MNBuffer>& buffer() const noexcept { return buffer_; }

    // -- Debug ---------------------------------------------------------------

    /// Human-readable summary, e.g. "MNTensor(shape=[2,3], dtype=float32)".
    std::string to_string() const;

private:
    std::shared_ptr<MNBuffer> buffer_;
    MNShape shape_;
    std::vector<int64_t> strides_;
    MNDType dtype_;
    size_t offset_;
};

} // namespace metal_native
