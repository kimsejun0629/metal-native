#pragma once

/// @file icb_recorder.h
/// @brief Indirect Command Buffer recording and replay for transformer layers.
///
/// ICBRecorder captures GPU kernel dispatches into MTLIndirectCommandBuffer
/// objects, then replays them on subsequent inference passes with matching
/// shapes. This eliminates per-layer CPU dispatch overhead.
///
/// Scoped to prefill only - during autoregressive decoding, seq_len_k
/// changes every token, invalidating the ICB each step.
///
/// Thread-safe: all public methods are guarded by a mutex.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace metal_native {

class MNDevice;

/// Shape key for ICB cache lookup.
struct ICBShapeKey {
    uint32_t batch;
    uint32_t num_heads;
    uint32_t seq_len_q;
    uint32_t seq_len_k;
    uint32_t head_dim;

    bool operator==(const ICBShapeKey& other) const {
        return batch == other.batch && num_heads == other.num_heads &&
               seq_len_q == other.seq_len_q && seq_len_k == other.seq_len_k &&
               head_dim == other.head_dim;
    }
};

/// Hash for ICBShapeKey
struct ICBShapeKeyHash {
    size_t operator()(const ICBShapeKey& k) const {
        size_t h = std::hash<uint32_t>{}(k.batch);
        h ^= std::hash<uint32_t>{}(k.num_heads) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(k.seq_len_q) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(k.seq_len_k) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(k.head_dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

class ICBRecorder {
public:
    /// Access the singleton ICBRecorder.
    static ICBRecorder& instance();

    // Non-copyable, non-movable.
    ICBRecorder(const ICBRecorder&) = delete;
    ICBRecorder& operator=(const ICBRecorder&) = delete;

    /// Check if ICB recording/replay is enabled.
    bool enabled() const noexcept;

    /// Enable/disable ICB recording.
    void set_enabled(bool enable);

    /// Check if we have a cached ICB for the given shape.
    bool has_cached(const ICBShapeKey& key) const;

    /// Begin recording dispatches for a transformer layer with the given shape.
    /// Returns true if recording started (no cache hit).
    /// Returns false if a cached ICB exists (caller should call replay instead).
    bool begin_recording(const ICBShapeKey& key);

    /// Record a compute dispatch into the current ICB being recorded.
    /// Call this instead of directly encoding to a command encoder.
#ifdef __OBJC__
    void record_dispatch(id<MTLComputePipelineState> pipeline,
                         NSArray<id<MTLBuffer>>* buffers,
                         const uint32_t* buffer_offsets,
                         uint32_t buffer_count,
                         const void* bytes_data,
                         uint32_t bytes_length,
                         uint32_t bytes_index,
                         MTLSize grid_size,
                         MTLSize threadgroup_size,
                         const uint32_t* threadgroup_memory_lengths,
                         uint32_t threadgroup_memory_count);
#endif

    /// Finish recording. The ICB is compiled and cached.
    void end_recording();

    /// Replay a cached ICB for the given shape key.
    /// Returns true if replay succeeded, false if no cache hit.
#ifdef __OBJC__
    bool replay(const ICBShapeKey& key, id<MTLComputeCommandEncoder> encoder);
#else
    bool replay(const ICBShapeKey& key, void* encoder);
#endif

    /// Invalidate all cached ICBs.
    void invalidate_all();

    /// Invalidate a specific cached ICB.
    void invalidate(const ICBShapeKey& key);

    /// Number of cached ICBs.
    size_t cache_size() const;

    /// Maximum number of commands per ICB.
    static constexpr uint32_t MAX_COMMANDS = 32;

    /// Maximum number of cached ICBs.
    static constexpr size_t MAX_CACHE_SIZE = 64;

private:
    ICBRecorder();
    ~ICBRecorder();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
