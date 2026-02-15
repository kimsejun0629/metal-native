/// @file buffer.mm
/// @brief Objective-C++ implementation of MNBuffer.

#import <Metal/Metal.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

#include "metal_native/core/buffer.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/memory/allocator.h"

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

    // Allocator-backed buffer support: when from_allocator is true, the
    // destructor returns the block to the pool instead of releasing it.
    bool           from_allocator = false;
    AllocatedBlock alloc_block;
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
    if (impl_) {
        if (impl_->from_allocator) {
            // Return buffer to the allocator pool for reuse.
            MNDevice::instance().allocator().deallocate(impl_->alloc_block);
        }
        if (impl_->release_callback) {
            impl_->release_callback();
        }
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
// allocate_pooled
// ---------------------------------------------------------------------------

std::shared_ptr<MNBuffer> MNBuffer::allocate_pooled(
    MNDevice& device, size_t size, StorageMode mode) {
    MN_CHECK(size > 0,
             MetalNativeError::InvalidArgument,
             "allocate_pooled: size must be > 0");

    auto& alloc = device.allocator();
    AllocatedBlock block = alloc.allocate(size, mode);

    auto buf = std::shared_ptr<MNBuffer>(new MNBuffer());
    buf->impl_ = std::make_unique<Impl>();
    buf->impl_->buffer         = block.buffer;
    buf->impl_->size           = size;
    buf->impl_->mode           = mode;
    buf->impl_->from_allocator = true;
    buf->impl_->alloc_block    = block;

    return buf;
}

// ---------------------------------------------------------------------------
// Memory accessibility check (Mach VM region API)
// ---------------------------------------------------------------------------

/// Check whether [start, start+length) lies entirely within CPU-readable
/// VM regions.  Walks the VM map using mach_vm_region to verify every
/// page in the range has VM_PROT_READ.  This avoids SIGBUS when MPS
/// data_ptr() points to GPU-only IOSurface memory that is not mapped
/// into the CPU address space.
static bool is_range_cpu_readable(uintptr_t start, size_t length) {
    if (length == 0) return true;
    uintptr_t end = start + length;
    uintptr_t cursor = start;

    while (cursor < end) {
        mach_vm_address_t region_addr = static_cast<mach_vm_address_t>(cursor);
        mach_vm_size_t region_size = 0;
        vm_region_basic_info_data_64_t info;
        mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object_name = MACH_PORT_NULL;

        kern_return_t kr = mach_vm_region(
            mach_task_self(),
            &region_addr,
            &region_size,
            VM_REGION_BASIC_INFO_64,
            reinterpret_cast<vm_region_info_t>(&info),
            &info_count,
            &object_name);

        if (kr != KERN_SUCCESS) return false;

        // mach_vm_region returns the region AT or AFTER the query address.
        // If the returned region starts after our cursor there is an
        // unmapped gap -- the range is not fully readable.
        if (region_addr > static_cast<mach_vm_address_t>(cursor))
            return false;

        // Region must be CPU-readable.
        if (!(info.protection & VM_PROT_READ))
            return false;

        // Advance past this region.
        cursor = static_cast<uintptr_t>(region_addr + region_size);
    }

    return true;
}

// ---------------------------------------------------------------------------
// wrap_external
// ---------------------------------------------------------------------------

std::shared_ptr<MNBuffer> MNBuffer::wrap_external(
    MNDevice& device,
    void* data_ptr,
    size_t size,
    std::function<void()> release_callback,
    size_t* out_offset) {

    MN_CHECK(data_ptr != nullptr,
             MetalNativeError::InvalidArgument,
             "wrap_external: data_ptr must not be null");

    MN_CHECK(size > 0,
             MetalNativeError::InvalidArgument,
             "wrap_external: size must be > 0");

    constexpr size_t kGPUPageSize = 16384; // 16 KB on Apple Silicon

    uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(data_ptr);

    // Page-align the pointer DOWN for newBufferWithBytesNoCopy.
    uintptr_t aligned_addr = ptr_addr & ~(kGPUPageSize - 1);
    size_t alignment_offset = ptr_addr - aligned_addr;
    void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);

    // Total length must cover from aligned base through end of data,
    // rounded up to page boundary.
    size_t total_length = (size + alignment_offset + kGPUPageSize - 1)
                          & ~(kGPUPageSize - 1);

    // Verify the entire page-aligned range is CPU-readable before
    // calling newBufferWithBytesNoCopy.  MPS data_ptr() for large
    // tensors often points to GPU-only IOSurface memory where only a
    // small prefix (~4 MB) is CPU-mapped.  Accessing beyond that
    // causes SIGBUS / KERN_PROTECTION_FAILURE.
    if (!is_range_cpu_readable(aligned_addr, total_length)) {
        char hex[32];
        snprintf(hex, sizeof(hex), "0x%lx", (unsigned long)ptr_addr);
        MN_THROW(MetalNativeError::AllocationFailed,
                 std::string("wrap_external: memory range at ") + hex +
                 " (size=" + std::to_string(size) +
                 " bytes, aligned_length=" + std::to_string(total_length) +
                 ") is not fully CPU-accessible. "
                 "This typically happens with large MPS tensors whose backing "
                 "memory is GPU-only. Use tensor_from_cpu_data() with "
                 "t.cpu() as a fallback.");
    }

    auto buf = std::shared_ptr<MNBuffer>(new MNBuffer());
    buf->impl_ = std::make_unique<Impl>();

    buf->impl_->buffer = [device.metal_device()
        newBufferWithBytesNoCopy:aligned_ptr
                          length:total_length
                         options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                     deallocator:nil];

    MN_CHECK(buf->impl_->buffer != nil,
             MetalNativeError::AllocationFailed,
             "wrap_external: newBufferWithBytesNoCopy failed for pointer "
             "0x" + ([&]{
                 char hex[32];
                 snprintf(hex, sizeof(hex), "%lx", (unsigned long)ptr_addr);
                 return std::string(hex);
             })() + " (size=" + std::to_string(size) + " bytes). "
             "The pointer may not be in GPU-accessible shared memory.");

    buf->impl_->size    = size;
    buf->impl_->mode    = StorageMode::Shared;
    buf->impl_->no_copy = true;
    buf->impl_->release_callback = std::move(release_callback);

    if (out_offset) {
        *out_offset = alignment_offset;
    }

    return buf;
}

} // namespace metal_native
