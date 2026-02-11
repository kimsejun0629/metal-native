/// @file shape.cpp
/// @brief Implementation of MNShape.

#include "metal_native/core/shape.h"
#include "metal_native/core/error.h"

#include <algorithm>
#include <numeric>
#include <sstream>

namespace metal_native {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MNShape::MNShape(std::vector<int64_t> dims) : dims_(std::move(dims)) {}

MNShape::MNShape(std::initializer_list<int64_t> dims) : dims_(dims) {}

// ---------------------------------------------------------------------------
// numel
// ---------------------------------------------------------------------------

int64_t MNShape::numel() const noexcept {
    if (dims_.empty()) {
        return 1; // scalar
    }
    return std::accumulate(
        dims_.begin(), dims_.end(), int64_t{1}, std::multiplies<int64_t>());
}

// ---------------------------------------------------------------------------
// operator[]
// ---------------------------------------------------------------------------

int64_t MNShape::operator[](int64_t i) const {
    int64_t ndim_val = static_cast<int64_t>(dims_.size());
    int64_t idx = i;
    if (idx < 0) {
        idx += ndim_val;
    }
    MN_CHECK(idx >= 0 && idx < ndim_val,
             MetalNativeError::InvalidArgument,
             "MNShape index " + std::to_string(i) +
             " out of range for shape with " + std::to_string(ndim_val) +
             " dimensions");
    return dims_[static_cast<size_t>(idx)];
}

// ---------------------------------------------------------------------------
// contiguous_strides
// ---------------------------------------------------------------------------

std::vector<int64_t> MNShape::contiguous_strides() const {
    const size_t n = dims_.size();
    if (n == 0) {
        return {};
    }
    std::vector<int64_t> strides(n);
    strides[n - 1] = 1;
    for (size_t i = n - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * dims_[i];
    }
    return strides;
}

// ---------------------------------------------------------------------------
// broadcast_with
// ---------------------------------------------------------------------------

MNShape MNShape::broadcast_with(const MNShape& other) const {
    const size_t max_ndim = std::max(dims_.size(), other.dims_.size());
    std::vector<int64_t> result(max_ndim);

    // Walk from the trailing dimension backward.
    for (size_t i = 0; i < max_ndim; ++i) {
        // Dimensions from the right; default to 1 if exhausted.
        int64_t d1 = (i < dims_.size())
                          ? dims_[dims_.size() - 1 - i]
                          : int64_t{1};
        int64_t d2 = (i < other.dims_.size())
                          ? other.dims_[other.dims_.size() - 1 - i]
                          : int64_t{1};

        if (d1 == d2) {
            result[max_ndim - 1 - i] = d1;
        } else if (d1 == 1) {
            result[max_ndim - 1 - i] = d2;
        } else if (d2 == 1) {
            result[max_ndim - 1 - i] = d1;
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "shapes " + to_string() + " and " + other.to_string() +
                     " are not broadcastable");
        }
    }

    return MNShape(std::move(result));
}

// ---------------------------------------------------------------------------
// is_contiguous
// ---------------------------------------------------------------------------

bool MNShape::is_contiguous(const std::vector<int64_t>& strides) const noexcept {
    if (strides.size() != dims_.size()) {
        return false;
    }
    if (dims_.empty()) {
        return true; // scalar
    }
    int64_t expected = 1;
    for (size_t i = dims_.size(); i > 0; --i) {
        size_t idx = i - 1;
        // Dimensions of size 1 do not affect contiguity.
        if (dims_[idx] != 1) {
            if (strides[idx] != expected) {
                return false;
            }
            expected *= dims_[idx];
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

bool MNShape::operator==(const MNShape& other) const noexcept {
    return dims_ == other.dims_;
}

bool MNShape::operator!=(const MNShape& other) const noexcept {
    return dims_ != other.dims_;
}

// ---------------------------------------------------------------------------
// to_string
// ---------------------------------------------------------------------------

std::string MNShape::to_string() const {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < dims_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << dims_[i];
    }
    oss << ']';
    return oss.str();
}

} // namespace metal_native
