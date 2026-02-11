#pragma once

/// @file shape.h
/// @brief Tensor shape and stride utilities.
///
/// MNShape wraps a small dimension vector and provides helpers for element
/// counting, contiguous stride computation, broadcasting, and contiguity
/// checks.

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace metal_native {

/// Compact tensor shape descriptor.
///
/// Stores dimensions as int64_t (matching PyTorch convention) to allow
/// negative-one sentinels used during reshape inference.
class MNShape {
public:
    // -- Construction --------------------------------------------------------

    MNShape() = default;
    explicit MNShape(std::vector<int64_t> dims);
    MNShape(std::initializer_list<int64_t> dims);

    // -- Accessors -----------------------------------------------------------

    /// Number of dimensions (rank).
    size_t ndim() const noexcept { return dims_.size(); }

    /// Total number of elements (product of all dimensions).
    /// Returns 1 for a scalar (ndim == 0).
    int64_t numel() const noexcept;

    /// Access the i-th dimension. Negative indices count from the end.
    int64_t operator[](int64_t i) const;

    /// Raw pointer to the dimension array.
    const int64_t* data() const noexcept { return dims_.data(); }

    /// Const reference to the underlying vector.
    const std::vector<int64_t>& dims() const noexcept { return dims_; }

    // -- Stride helpers ------------------------------------------------------

    /// Compute the contiguous (row-major / C-order) strides for this shape.
    std::vector<int64_t> contiguous_strides() const;

    // -- Broadcast -----------------------------------------------------------

    /// Compute the broadcast shape of *this and @p other following NumPy rules.
    /// Throws MNException(InvalidArgument) if the shapes are incompatible.
    MNShape broadcast_with(const MNShape& other) const;

    // -- Contiguity check ----------------------------------------------------

    /// Return true if the given strides correspond to a contiguous (row-major)
    /// layout for this shape.
    bool is_contiguous(const std::vector<int64_t>& strides) const noexcept;

    // -- Comparison ----------------------------------------------------------

    bool operator==(const MNShape& other) const noexcept;
    bool operator!=(const MNShape& other) const noexcept;

    // -- Debug ---------------------------------------------------------------

    /// Human-readable representation, e.g. "[2, 3, 4]".
    std::string to_string() const;

private:
    std::vector<int64_t> dims_;
};

} // namespace metal_native
