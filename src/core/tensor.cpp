/// @file tensor.cpp
/// @brief Implementation of MNTensor.

#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <sstream>

namespace metal_native {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MNTensor::MNTensor(std::shared_ptr<MNBuffer> buffer,
                   MNShape shape,
                   std::vector<int64_t> strides,
                   MNDType dtype,
                   size_t offset)
    : buffer_(std::move(buffer)),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      dtype_(dtype),
      offset_(offset) {
    MN_CHECK(buffer_ != nullptr,
             MetalNativeError::InvalidArgument,
             "MNTensor: buffer must not be null");
}

// ---------------------------------------------------------------------------
// Static factories
// ---------------------------------------------------------------------------

MNTensor MNTensor::empty(const MNShape& shape, MNDType dtype,
                         MNDevice& device, StorageMode mode) {
    const int64_t numel = shape.numel();
    const size_t dtype_bytes = dtype_size(dtype);

    // Check for integer overflow before allocation
    if (numel > 0 && dtype_bytes > 0) {
        if (static_cast<size_t>(numel) > SIZE_MAX / dtype_bytes) {
            MN_THROW(MetalNativeError::OutOfMemory,
                     "Tensor allocation would overflow: numel=" + std::to_string(numel) +
                     ", dtype_size=" + std::to_string(dtype_bytes));
        }
    }

    const size_t nbytes = static_cast<size_t>(numel) * dtype_bytes;
    // Ensure at least 1 byte allocation (Metal does not allow zero-length buffers).
    const size_t alloc = std::max(nbytes, size_t{1});

    auto buf = std::make_shared<MNBuffer>(device, alloc, mode);
    auto strides = shape.contiguous_strides();
    return MNTensor(std::move(buf), shape, std::move(strides), dtype);
}

MNTensor MNTensor::zeros(const MNShape& shape, MNDType dtype,
                         MNDevice& device) {
    MNTensor t = empty(shape, dtype, device);
    // Shared-mode buffers are CPU-accessible; memset to zero.
    std::memset(t.raw_data(), 0, t.nbytes());
    return t;
}

MNTensor MNTensor::ones(const MNShape& shape, MNDType dtype,
                        MNDevice& device) {
    MNTensor t = empty(shape, dtype, device);
    t.fill_(1.0);
    return t;
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

size_t MNTensor::nbytes() const noexcept {
    return static_cast<size_t>(numel()) * dtype_size(dtype_);
}

bool MNTensor::is_contiguous() const noexcept {
    return shape_.is_contiguous(strides_);
}

// ---------------------------------------------------------------------------
// reshape
// ---------------------------------------------------------------------------

MNTensor MNTensor::reshape(const MNShape& new_shape) const {
    MN_CHECK(is_contiguous(),
             MetalNativeError::InvalidArgument,
             "reshape requires a contiguous tensor");

    const auto& new_dims = new_shape.dims();
    int64_t inferred_idx = -1;
    int64_t known_product = 1;

    for (size_t i = 0; i < new_dims.size(); ++i) {
        if (new_dims[i] == -1) {
            MN_CHECK(inferred_idx == -1,
                     MetalNativeError::InvalidArgument,
                     "reshape: at most one dimension may be -1");
            inferred_idx = static_cast<int64_t>(i);
        } else {
            MN_CHECK(new_dims[i] >= 0,
                     MetalNativeError::InvalidArgument,
                     "reshape: dimensions must be non-negative (or -1)");
            known_product *= new_dims[i];
        }
    }

    std::vector<int64_t> resolved_dims = new_dims;
    const int64_t total = numel();

    if (inferred_idx >= 0) {
        MN_CHECK(known_product != 0 && total % known_product == 0,
                 MetalNativeError::InvalidArgument,
                 "reshape: cannot infer dimension; numel=" +
                 std::to_string(total) + " is not divisible by " +
                 std::to_string(known_product));
        resolved_dims[static_cast<size_t>(inferred_idx)] = total / known_product;
    } else {
        MN_CHECK(known_product == total,
                 MetalNativeError::InvalidArgument,
                 "reshape: new shape numel (" + std::to_string(known_product) +
                 ") != current numel (" + std::to_string(total) + ")");
    }

    MNShape resolved(std::move(resolved_dims));
    auto strides = resolved.contiguous_strides();
    return MNTensor(buffer_, std::move(resolved), std::move(strides),
                    dtype_, offset_);
}

// ---------------------------------------------------------------------------
// slice
// ---------------------------------------------------------------------------

MNTensor MNTensor::slice(int64_t dim, int64_t start, int64_t end) const {
    const int64_t ndim_val = static_cast<int64_t>(ndim());
    int64_t d = dim;
    if (d < 0) d += ndim_val;
    MN_CHECK(d >= 0 && d < ndim_val,
             MetalNativeError::InvalidArgument,
             "slice: dimension " + std::to_string(dim) + " out of range");

    const int64_t dim_size = shape_[d];

    // Normalise start.
    int64_t s = start;
    if (s < 0) s += dim_size;
    s = std::max(int64_t{0}, std::min(s, dim_size));

    // Normalise end (-1 means full extent).
    int64_t e = end;
    if (e < 0) e += dim_size;
    // Clamp (end == -1 after normalisation for dim_size == 0 is handled here).
    e = std::max(s, std::min(e, dim_size));

    // Build new shape: same everywhere except the sliced dimension.
    std::vector<int64_t> new_dims = shape_.dims();
    new_dims[static_cast<size_t>(d)] = e - s;

    // Compute the byte offset introduced by slicing.
    const size_t elem_size = dtype_size(dtype_);
    const size_t new_offset =
        offset_ +
        static_cast<size_t>(s) * static_cast<size_t>(strides_[static_cast<size_t>(d)]) * elem_size;

    return MNTensor(buffer_, MNShape(std::move(new_dims)), strides_,
                    dtype_, new_offset);
}

// ---------------------------------------------------------------------------
// clone
// ---------------------------------------------------------------------------

MNTensor MNTensor::clone() const {
    MN_CHECK(buffer_->data() != nullptr,
             MetalNativeError::InvalidArgument,
             "clone: cannot clone a Private-mode tensor on the CPU");

    // Allocate a fresh contiguous buffer and copy data.
    const size_t bytes = nbytes();
    const size_t alloc = std::max(bytes, size_t{1});

    // Obtain device from the buffer -- we re-use MNDevice::instance() since
    // there is only one device on Apple Silicon.
    auto new_buf = std::make_shared<MNBuffer>(
        MNDevice::instance(), alloc, StorageMode::Shared);

    if (is_contiguous()) {
        std::memcpy(new_buf->data(), raw_data(), bytes);
    } else {
        // Non-contiguous clone: iterate element-by-element.
        const size_t elem = dtype_size(dtype_);
        const int64_t n = numel();
        const size_t rank = ndim();
        const auto& dims = shape_.dims();

        auto* dst = static_cast<uint8_t*>(new_buf->data());
        auto* src = static_cast<const uint8_t*>(buffer_->data()) + offset_;

        // Multi-dimensional index iteration.
        std::vector<int64_t> idx(rank, 0);
        for (int64_t flat = 0; flat < n; ++flat) {
            // Compute source offset from strides.
            size_t src_off = 0;
            for (size_t d = 0; d < rank; ++d) {
                src_off += static_cast<size_t>(idx[d]) *
                           static_cast<size_t>(strides_[d]) * elem;
            }
            std::memcpy(dst + static_cast<size_t>(flat) * elem,
                        src + src_off, elem);

            // Advance multi-index (row-major order).
            for (size_t d = rank; d > 0; --d) {
                ++idx[d - 1];
                if (idx[d - 1] < dims[d - 1]) break;
                idx[d - 1] = 0;
            }
        }
    }

    auto strides = shape_.contiguous_strides();
    return MNTensor(std::move(new_buf), shape_, std::move(strides), dtype_);
}

// ---------------------------------------------------------------------------
// fill_ helpers
// ---------------------------------------------------------------------------

namespace {

/// Write @p value to every element of a non-contiguous tensor.
/// @p base is the raw pointer to the tensor's first element (offset-adjusted).
/// @p strides_vec and @p dims are the tensor's strides and shape dimensions.
/// @p elem_size is the byte size of one element.
template <typename T>
void fill_strided(uint8_t* base,
                  T value,
                  int64_t numel,
                  const std::vector<int64_t>& strides_vec,
                  const std::vector<int64_t>& dims,
                  size_t elem_size) {
    const size_t rank = dims.size();
    std::vector<int64_t> idx(rank, 0);

    for (int64_t flat = 0; flat < numel; ++flat) {
        size_t byte_off = 0;
        for (size_t d = 0; d < rank; ++d) {
            byte_off += static_cast<size_t>(idx[d]) *
                        static_cast<size_t>(strides_vec[d]) * elem_size;
        }
        *reinterpret_cast<T*>(base + byte_off) = value;

        // Advance multi-index.
        for (size_t d = rank; d > 0; --d) {
            ++idx[d - 1];
            if (idx[d - 1] < dims[d - 1]) break;
            idx[d - 1] = 0;
        }
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// fill_
// ---------------------------------------------------------------------------

void MNTensor::fill_(double value) {
    MN_CHECK(buffer_->data() != nullptr,
             MetalNativeError::InvalidArgument,
             "fill_: cannot fill a Private-mode tensor from the CPU");

    auto* base = static_cast<uint8_t*>(raw_data());
    const int64_t n = numel();
    const size_t elem_size = dtype_size(dtype_);
    const auto& dims = shape_.dims();

    // Fast path: contiguous tensor -- write sequentially.
    // For non-contiguous tensors we fall back to per-element addressing.

    switch (dtype_) {
        case MNDType::Float32: {
            auto v = static_cast<float>(value);
            if (is_contiguous()) {
                auto* p = reinterpret_cast<float*>(base);
                std::fill(p, p + n, v);
            } else {
                fill_strided<float>(base, v, n, strides_, dims, elem_size);
            }
            break;
        }
        case MNDType::Float16: {
            // __fp16 is a compiler-supported type on ARM64 with clang.
            auto v = static_cast<__fp16>(static_cast<float>(value));
            if (is_contiguous()) {
                auto* p = reinterpret_cast<__fp16*>(base);
                std::fill(p, p + n, v);
            } else {
                fill_strided<__fp16>(base, v, n, strides_, dims, elem_size);
            }
            break;
        }
        case MNDType::BFloat16: {
            // BFloat16 has no native C++ type.  Construct manually:
            //   sign(1) | exponent(8) | mantissa(7)  (truncate from float32).
            uint16_t bits;
            {
                float fv = static_cast<float>(value);
                uint32_t u;
                std::memcpy(&u, &fv, sizeof(u));
                bits = static_cast<uint16_t>(u >> 16);
            }
            if (is_contiguous()) {
                auto* p = reinterpret_cast<uint16_t*>(base);
                std::fill(p, p + n, bits);
            } else {
                fill_strided<uint16_t>(base, bits, n, strides_, dims, elem_size);
            }
            break;
        }
        case MNDType::Int64: {
            auto v = static_cast<int64_t>(value);
            if (is_contiguous()) {
                auto* p = reinterpret_cast<int64_t*>(base);
                std::fill(p, p + n, v);
            } else {
                fill_strided<int64_t>(base, v, n, strides_, dims, elem_size);
            }
            break;
        }
        case MNDType::Int32: {
            auto v = static_cast<int32_t>(value);
            if (is_contiguous()) {
                auto* p = reinterpret_cast<int32_t*>(base);
                std::fill(p, p + n, v);
            } else {
                fill_strided<int32_t>(base, v, n, strides_, dims, elem_size);
            }
            break;
        }
        case MNDType::Int16: {
            auto v = static_cast<int16_t>(value);
            if (is_contiguous()) {
                auto* p = reinterpret_cast<int16_t*>(base);
                std::fill(p, p + n, v);
            } else {
                fill_strided<int16_t>(base, v, n, strides_, dims, elem_size);
            }
            break;
        }
        case MNDType::Int8: {
            auto v = static_cast<int8_t>(value);
            if (is_contiguous()) {
                auto* p = reinterpret_cast<int8_t*>(base);
                std::fill(p, p + n, v);
            } else {
                fill_strided<int8_t>(base, v, n, strides_, dims, elem_size);
            }
            break;
        }
        case MNDType::UInt8: {
            auto v = static_cast<uint8_t>(value);
            if (is_contiguous()) {
                auto* p = reinterpret_cast<uint8_t*>(base);
                std::fill(p, p + n, v);
            } else {
                fill_strided<uint8_t>(base, v, n, strides_, dims, elem_size);
            }
            break;
        }
        case MNDType::Bool: {
            auto v = static_cast<uint8_t>(value != 0.0 ? 1 : 0);
            if (is_contiguous()) {
                auto* p = reinterpret_cast<uint8_t*>(base);
                std::fill(p, p + n, v);
            } else {
                fill_strided<uint8_t>(base, v, n, strides_, dims, elem_size);
            }
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// to_string
// ---------------------------------------------------------------------------

std::string MNTensor::to_string() const {
    std::ostringstream oss;
    oss << "MNTensor(shape=" << shape_.to_string()
        << ", dtype=" << dtype_name(dtype_)
        << ", contiguous=" << (is_contiguous() ? "true" : "false")
        << ", offset=" << offset_
        << ")";
    return oss.str();
}

} // namespace metal_native
