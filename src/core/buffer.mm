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
    MTLResourceOptions options =
        static_cast<MTLResourceOptions>(static_cast<uint32_t>(mode)) << MTLResourceStorageModeShift;

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

MNBuffer::~MNBuffer() = default;

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

    buf->impl_->buffer = [device.metal_device()
        newBufferWithBytesNoCopy:cpu_ptr
                          length:size
                         options:MTLResourceStorageModeShared
                     deallocator:nil];

    MN_CHECK(buf->impl_->buffer != nil,
             MetalNativeError::AllocationFailed,
             "create_zero_copy: newBufferWithBytesNoCopy returned nil");

    buf->impl_->size    = size;
    buf->impl_->mode    = StorageMode::Shared;
    buf->impl_->no_copy = true;

    return buf;
}

} // namespace metal_native
