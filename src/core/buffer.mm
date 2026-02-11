/// @file buffer.mm
/// @brief Objective-C++ implementation of MNBuffer.

#import <Metal/Metal.h>

#include "metal_native/core/buffer.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

#include <utility>

namespace metal_native {

// ---------------------------------------------------------------------------
// Impl (pimpl -- hides the id<MTLBuffer> from pure-C++ headers)
// ---------------------------------------------------------------------------

struct MNBuffer::Impl {
    id<MTLBuffer> buffer  = nil;
    size_t        size    = 0;
    StorageMode   mode    = StorageMode::Shared;
    bool          no_copy = false; // true for create_zero_copy buffers
    std::function<void()> release_callback; // Optional callback for wrap_external
};

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

MNBuffer::MNBuffer() = default;

MNBuffer::MNBuffer(MNDevice& device, size_t size, StorageMode mode)
    : impl_(std::make_unique<Impl>()) {
    MN_CHECK(size > 0,
             MetalNativeError::InvalidArgument,
             "MNBuffer: size must be > 0");

    MN_CHECK(size <= device.max_buffer_length(),
             MetalNativeError::AllocationFailed,
             "MNBuffer: requested size (" + std::to_string(size) +
             ") exceeds device maximum (" +
             std::to_string(device.max_buffer_length()) + ")");

    // MTLStorageModeShared (0) or MTLStorageModePrivate (2) -- the enum values
    // are chosen to match, so a static_cast is correct.
    // Apply MTLResourceHazardTrackingModeUntracked for 3-8% performance improvement.
    // Safe because command buffer ordering already manages read/write dependencies.
    MTLResourceOptions options =
        (static_cast<MTLResourceOptions>(static_cast<uint32_t>(mode)) << MTLResourceStorageModeShift) |
        MTLResourceHazardTrackingModeUntracked;

    impl_->buffer = [device.metal_device() newBufferWithLength:size
                                                       options:options];
    MN_CHECK(impl_->buffer != nil,
             MetalNativeError::AllocationFailed,
             "MNBuffer: MTLDevice newBufferWithLength returned nil "
             "(requested " + std::to_string(size) + " bytes)");

    impl_->size = size;
    impl_->mode = mode;
}

// ---------------------------------------------------------------------------
// Destructor / move
// ---------------------------------------------------------------------------

MNBuffer::~MNBuffer() {
    if (impl_ && impl_->release_callback) {
        impl_->release_callback();
    }
}

MNBuffer::MNBuffer(MNBuffer&& other) noexcept = default;
MNBuffer& MNBuffer::operator=(MNBuffer&& other) noexcept = default;

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

void* MNBuffer::data() const noexcept {
    if (!impl_ || impl_->mode == StorageMode::Private) {
        return nullptr;
    }
    return [impl_->buffer contents];
}

uint64_t MNBuffer::gpu_address() const noexcept {
    if (!impl_) return 0;
    // gpuAddress is available on macOS 13+ / iOS 16+ (Metal 3).
    if (@available(macOS 13.0, iOS 16.0, *)) {
        return [impl_->buffer gpuAddress];
    }
    return 0;
}

size_t MNBuffer::size() const noexcept {
    return impl_ ? impl_->size : 0;
}

StorageMode MNBuffer::storage_mode() const noexcept {
    return impl_ ? impl_->mode : StorageMode::Shared;
}

id<MTLBuffer> MNBuffer::metal_buffer() const noexcept {
    return impl_ ? impl_->buffer : nil;
}

// ---------------------------------------------------------------------------
// create_zero_copy
// ---------------------------------------------------------------------------

std::unique_ptr<MNBuffer> MNBuffer::create_zero_copy(
    MNDevice& device, void* cpu_ptr, size_t size) {
    constexpr size_t kGPUPageSize = 16384; // 16 KB

    MN_CHECK(cpu_ptr != nullptr,
             MetalNativeError::InvalidArgument,
             "create_zero_copy: cpu_ptr must not be null");

    MN_CHECK(size > 0,
             MetalNativeError::InvalidArgument,
             "create_zero_copy: size must be > 0");

    MN_CHECK(reinterpret_cast<uintptr_t>(cpu_ptr) % kGPUPageSize == 0,
             MetalNativeError::InvalidArgument,
             "create_zero_copy: cpu_ptr must be aligned to GPU page size (16 KB)");

    MN_CHECK(size % kGPUPageSize == 0,
             MetalNativeError::InvalidArgument,
             "create_zero_copy: size must be a multiple of GPU page size (16 KB)");

    // Allocate a MNBuffer object without going through the normal constructor.
    auto buf = std::unique_ptr<MNBuffer>(new MNBuffer());
    buf->impl_ = std::make_unique<Impl>();

    // Apply MTLResourceHazardTrackingModeUntracked for 3-8% performance improvement.
    // Safe because command buffer ordering already manages read/write dependencies.
    buf->impl_->buffer = [device.metal_device()
        newBufferWithBytesNoCopy:cpu_ptr
                          length:size
                         options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                     deallocator:nil];

    MN_CHECK(buf->impl_->buffer != nil,
             MetalNativeError::AllocationFailed,
             "create_zero_copy: newBufferWithBytesNoCopy returned nil");

    buf->impl_->size    = size;
    buf->impl_->mode    = StorageMode::Shared;
    buf->impl_->no_copy = true;

    return buf;
}

// ---------------------------------------------------------------------------
// wrap_external
// ---------------------------------------------------------------------------

std::shared_ptr<MNBuffer> MNBuffer::wrap_external(
    MNDevice& device,
    void* data_ptr,
    size_t size,
    std::function<void()> release_callback) {

    MN_CHECK(data_ptr != nullptr,
             MetalNativeError::InvalidArgument,
             "wrap_external: data_ptr must not be null");

    MN_CHECK(size > 0,
             MetalNativeError::InvalidArgument,
             "wrap_external: size must be > 0");

    constexpr size_t kGPUPageSize = 16384; // 16 KB on Apple Silicon

    auto buf = std::shared_ptr<MNBuffer>(new MNBuffer());
    buf->impl_ = std::make_unique<Impl>();

    uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(data_ptr);
    bool ptr_aligned = (ptr_addr % kGPUPageSize) == 0;

    // Round size up to page boundary for newBufferWithBytesNoCopy.
    size_t aligned_size = (size + kGPUPageSize - 1) & ~(kGPUPageSize - 1);

    // Try zero-copy first if the pointer is page-aligned.
    if (ptr_aligned) {
        buf->impl_->buffer = [device.metal_device()
            newBufferWithBytesNoCopy:data_ptr
                              length:aligned_size
                             options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                         deallocator:nil];
    }

    if (buf->impl_->buffer == nil) {
        // Fallback: allocate a new buffer and memcpy data.
        // This handles non-page-aligned pointers (e.g. PyTorch MPS weight
        // tensors whose data_ptr is offset within a shared storage).
        // NOTE: MPS data_ptr() can SIGBUS during large memcpy. Callers
        // should prefer passing CPU-accessible pointers for large buffers.
        buf->impl_->buffer = [device.metal_device()
            newBufferWithLength:size
                        options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];

        MN_CHECK(buf->impl_->buffer != nil,
                 MetalNativeError::AllocationFailed,
                 "wrap_external: buffer allocation failed (size=" +
                 std::to_string(size) + " bytes)");

        std::memcpy([buf->impl_->buffer contents], data_ptr, size);

        buf->impl_->size    = size;
        buf->impl_->mode    = StorageMode::Shared;
        buf->impl_->no_copy = false;
        buf->impl_->release_callback = std::move(release_callback);
        return buf;
    }

    buf->impl_->size    = size;
    buf->impl_->mode    = StorageMode::Shared;
    buf->impl_->no_copy = true;
    buf->impl_->release_callback = std::move(release_callback);

    return buf;
}

} // namespace metal_native
