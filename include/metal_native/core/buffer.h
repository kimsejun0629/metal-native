#pragma once

/// @file buffer.h
/// @brief GPU buffer abstraction backed by MTLBuffer.
///
/// MNBuffer wraps an id<MTLBuffer> with automatic lifetime management and
/// provides accessors for CPU/GPU pointers.  On Apple Silicon with
/// MTLStorageModeShared the CPU and GPU share the same physical page, so
/// `data()` returns a pointer usable by both sides (UMA zero-copy).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace metal_native {

class MNDevice;

/// Storage mode for GPU buffers.  Values are chosen to match the Metal API
/// constants so a static_cast is safe.
enum class StorageMode : uint32_t {
    /// Shared memory -- CPU and GPU access the same physical pages.
    /// This is the preferred mode on Apple Silicon (UMA).
    Shared = 0,

    /// Private memory -- GPU-only, not CPU-accessible.
    /// Useful for intermediate results that never leave the GPU.
    Private = 2,
};

/// Owning wrapper around an id<MTLBuffer>.
class MNBuffer {
public:
    // -- Construction --------------------------------------------------------

    /// Allocate a new Metal buffer of @p size bytes.
    ///
    /// @param device  The MNDevice that will own this buffer.
    /// @param size    Allocation size in bytes (must be > 0).
    /// @param mode    Storage mode (default: Shared for UMA zero-copy).
    MNBuffer(MNDevice& device, size_t size, StorageMode mode = StorageMode::Shared);

    ~MNBuffer();

    // Move-only semantics.
    MNBuffer(MNBuffer&& other) noexcept;
    MNBuffer& operator=(MNBuffer&& other) noexcept;
    MNBuffer(const MNBuffer&) = delete;
    MNBuffer& operator=(const MNBuffer&) = delete;

    // -- Accessors -----------------------------------------------------------

    /// CPU-accessible pointer to the buffer contents.
    /// Only valid when storage mode is Shared.  Returns nullptr for Private.
    void* data() const noexcept;

    /// GPU virtual address (MTLBuffer.gpuAddress, available macOS 13+).
    uint64_t gpu_address() const noexcept;

    /// Size of the allocation in bytes.
    size_t size() const noexcept;

    /// The storage mode used at allocation time.
    StorageMode storage_mode() const noexcept;

    // -- Raw Metal accessor --------------------------------------------------

#ifdef __OBJC__
    id<MTLBuffer> metal_buffer() const noexcept;
#else
    void* metal_buffer() const noexcept;
#endif

    // -- Factory -------------------------------------------------------------

    /// Create a Shared-mode buffer that wraps an existing CPU allocation with
    /// MTLDevice newBufferWithBytesNoCopy.  The caller is responsible for
    /// ensuring that @p cpu_ptr remains valid for the buffer's lifetime and
    /// that the memory is page-aligned (GPU page size = 16 KB).
    ///
    /// @param device   The MNDevice.
    /// @param cpu_ptr  Pointer to page-aligned memory.
    /// @param size     Size in bytes (must be a multiple of the GPU page size).
    /// @return         A zero-copy MNBuffer backed by the supplied memory.
    static std::unique_ptr<MNBuffer> create_zero_copy(
        MNDevice& device, void* cpu_ptr, size_t size);

    /// Allocate a buffer from the device's MetalSmartAllocator pool.
    /// The buffer is returned to the pool when the shared_ptr's reference
    /// count reaches zero, enabling buffer reuse without re-allocation.
    ///
    /// @param device  The MNDevice (provides the allocator).
    /// @param size    Allocation size in bytes (must be > 0).
    /// @param mode    Storage mode (default: Shared).
    /// @return Shared pointer to a pooled MNBuffer.
    static std::shared_ptr<MNBuffer> allocate_pooled(
        MNDevice& device, size_t size, StorageMode mode = StorageMode::Shared);

    /// Wrap an externally-owned pointer as an MNBuffer (zero-copy).
    /// The caller is responsible for keeping the external memory alive.
    /// @param device  The Metal device.
    /// @param data_ptr  CPU-accessible pointer to existing MTLBuffer contents.
    /// @param size  Size in bytes.
    /// @param release_callback  Optional callback invoked when the buffer is destroyed.
    /// @return Shared pointer to the wrapping buffer.
    static std::shared_ptr<MNBuffer> wrap_external(
        MNDevice& device,
        void* data_ptr,
        size_t size,
        std::function<void()> release_callback = nullptr);

private:
    /// Default constructor used only by create_zero_copy.
    MNBuffer();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
