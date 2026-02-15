#pragma once

/// @file command_pipeline.h
/// @brief Triple-buffered MTLCommandBuffer management for GPU command submission.
///
/// CommandPipeline maintains a ring buffer of MTLCommandBuffer instances,
/// allowing overlap between CPU encoding and GPU execution.  The default
/// configuration uses 3 buffers (triple-buffering), which is the sweet spot
/// for Apple Silicon GPUs.
///
/// The pipeline integrates with BackpressureController to throttle submission
/// when the GPU falls behind.
///
/// Thread-safe: all public methods are guarded by a mutex.

#include <cstddef>
#include <cstdint>
#include <memory>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace metal_native {

class MNDevice;

class CommandPipeline {
public:
    // Forward declaration for RAII batch scope
    class BatchScope;
    // Forward declaration for RAII encoder scope
    class EncoderScope;
    /// Construct a pipeline backed by the given device.
    ///
    /// @param device        The MNDevice that owns the command queue.
    /// @param buffer_count  Number of command buffer slots (default: 3).
    explicit CommandPipeline(MNDevice& device, size_t buffer_count = 3);
    ~CommandPipeline();

    // Non-copyable, non-movable.
    CommandPipeline(const CommandPipeline&) = delete;
    CommandPipeline& operator=(const CommandPipeline&) = delete;
    CommandPipeline(CommandPipeline&&) = delete;
    CommandPipeline& operator=(CommandPipeline&&) = delete;

    // -- Buffer lifecycle ----------------------------------------------------

#ifdef __OBJC__
    /// Return the current command buffer for encoding compute/blit commands.
    ///
    /// If no buffer exists yet, one is created from the device's command queue.
    id<MTLCommandBuffer> current_buffer();
#else
    /// Opaque handle to the current MTLCommandBuffer (cast in .mm files).
    void* current_buffer();
#endif

    /// Commit the current command buffer and create a new one for the
    /// next encoding pass.
    ///
    /// This adds a completion handler, commits the buffer, and advances
    /// the ring index.  Blocks if all buffer slots are in-flight
    /// (backpressure).
    ///
    /// OPT-5: This method enables command buffer batching by allowing
    /// consecutive operations to encode into fresh buffers without
    /// waiting for GPU completion. The backpressure controller ensures
    /// CPU doesn't get too far ahead of GPU (default: 3 buffers).
    ///
    /// USE THIS by default in operation implementations unless you need
    /// explicit synchronization (e.g., reading results back to CPU).
    void commit_and_continue();

    /// Commit the current command buffer (final submission, no rotation).
    ///
    /// NOTE: This does NOT wait for GPU completion. It simply commits
    /// the buffer and marks the pipeline as having no active buffer.
    /// The next call to current_buffer() will create a new one.
    ///
    /// WHEN TO USE: Use commit() when you're done with a sequence of
    /// operations and don't expect another operation immediately after.
    /// For most operation implementations, prefer commit_and_continue().
    void commit();

    /// Block until all in-flight command buffers have completed on the GPU.
    void synchronize();

    // -- Lazy commit mode ----------------------------------------------------

    /// Enable/disable lazy commit mode.
    ///
    /// When enabled, commit_and_continue() becomes a no-op — operations
    /// accumulate in the current command buffer until synchronize() or
    /// flush() is called.  This matches PyTorch MPS behavior where many
    /// operations are batched into a single command buffer submission.
    ///
    /// Default: false (each operation commits immediately).
    void set_lazy_commit(bool enable);
    bool lazy_commit() const noexcept;

    /// Flush: commit the current buffer if one is active (even in lazy mode).
    void flush();

    // -- Batch scoping -------------------------------------------------------

    /// Create a RAII batch scope that groups operations into a single command buffer.
    ///
    /// On construction, the scope enables lazy_commit mode. On destruction,
    /// it flushes accumulated commands and restores the previous lazy_commit state.
    ///
    /// Usage:
    ///   {
    ///       auto scope = pipeline.batch_scope();
    ///       // Multiple operations accumulate in one command buffer
    ///       op1();
    ///       op2();
    ///       op3();
    ///   } // flush happens here automatically
    BatchScope batch_scope();

    // -- Encoder scoping -----------------------------------------------------

    /// Create a RAII encoder scope that keeps a single compute command encoder
    /// open across multiple operations.
    ///
    /// Normally each operation creates its own encoder, encodes, calls endEncoding,
    /// and commits. This creates a new command buffer for every operation.
    /// EncoderScope keeps ONE encoder alive across multiple ops, reducing overhead.
    ///
    /// On construction, the scope creates/retrieves a compute command encoder.
    /// On destruction, it ends encoding and calls commit_and_continue().
    ///
    /// Usage:
    ///   {
    ///       auto scope = pipeline.encoder_scope();
    ///       // Multiple operations encode into the same encoder
    ///       op1();  // uses current_encoder()
    ///       op2();  // uses current_encoder()
    ///       op3();  // uses current_encoder()
    ///   } // endEncoding + commit happens here automatically
    ///
    /// Smart auto-commit: if too many operations are encoded in one encoder
    /// (default: 32), the encoder is ended and a new one is created to avoid
    /// GPU stalls from very long command buffers.
    EncoderScope encoder_scope();

    /// Create a combined scope for transformer layers: batch + encoder.
    ///
    /// This is a convenience method that combines batch_scope() and encoder_scope()
    /// for maximum batching efficiency in transformer layers.
    ///
    /// Usage:
    ///   {
    ///       auto scope = pipeline.transformer_layer_scope();
    ///       // All ops in the layer: lazy commit + single encoder
    ///       rmsnorm(); qkv_proj(); rope(); attention(); ...
    ///   } // endEncoding + flush happens here
    EncoderScope transformer_layer_scope();

#ifdef __OBJC__
    /// Return the current active encoder if an encoder scope is active,
    /// otherwise return nil. Operations should check this before creating
    /// their own encoder.
    ///
    /// When an encoder scope is active, operations should use this encoder
    /// instead of creating a new one from current_buffer().
    id<MTLComputeCommandEncoder> current_encoder();
#else
    /// Opaque handle to the current MTLComputeCommandEncoder (cast in .mm files).
    void* current_encoder();
#endif

    /// Check if an encoder scope is currently active.
    bool has_active_encoder() const noexcept;

    /// Set the maximum number of operations before auto-flushing the encoder
    /// within an encoder scope (default: 32).
    /// This also resets the adaptive value to the specified max_ops.
    /// Values are clamped to [1, 512].
    void set_encoder_max_ops(size_t max_ops);
    size_t encoder_max_ops() const noexcept;

    /// Get the current adaptive encoder max ops value (self-tuning).
    size_t adaptive_encoder_max_ops() const;

    /// Get the current encoder duration EMA in milliseconds (for profiling).
    double encoder_duration_ema() const;

    // -- Operation counting --------------------------------------------------

    /// Increment the operation counter. When the count reaches the auto-flush
    /// threshold, the current buffer is flushed even in lazy mode.
    /// This prevents command buffers from growing unboundedly.
    void record_op();

    /// Set the maximum number of operations before auto-flush (default: 64).
    void set_auto_flush_threshold(size_t threshold);
    size_t auto_flush_threshold() const noexcept;

    // -- Queries -------------------------------------------------------------

    /// Number of committed-but-not-completed buffers currently in flight.
    size_t in_flight_count() const;

    /// Total number of buffer slots in the ring.
    size_t buffer_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// BatchScope RAII guard
// ---------------------------------------------------------------------------

/// RAII guard that enables lazy commit mode for the lifetime of the scope.
/// On construction, enables lazy_commit. On destruction, flushes the
/// accumulated commands and restores the previous lazy_commit state.
///
/// Nested scopes are supported: inner scopes save/restore the lazy state
/// set by outer scopes, and all flushes happen at the appropriate nesting level.
class CommandPipeline::BatchScope {
public:
    explicit BatchScope(CommandPipeline& pipeline);
    ~BatchScope();

    // Non-copyable, non-movable
    BatchScope(const BatchScope&) = delete;
    BatchScope& operator=(const BatchScope&) = delete;
    BatchScope(BatchScope&&) = delete;
    BatchScope& operator=(BatchScope&&) = delete;

private:
    CommandPipeline& pipeline_;
    bool previous_lazy_state_;
};

// ---------------------------------------------------------------------------
// EncoderScope RAII guard
// ---------------------------------------------------------------------------

/// RAII guard that keeps a single compute command encoder open for the
/// lifetime of the scope.
///
/// On construction, creates a compute command encoder from current_buffer().
/// On destruction, ends encoding and calls commit_and_continue().
///
/// This reduces GPU dispatch overhead by batching consecutive kernel dispatches
/// into a single command encoder instead of creating/ending an encoder for
/// each operation.
///
/// Nested scopes are supported: inner scopes save/restore the encoder state.
/// The innermost scope owns the active encoder.
///
/// Can be combined with BatchScope for maximum batching efficiency:
///   {
///       auto batch = pipeline.batch_scope();      // lazy commit
///       auto encoder = pipeline.encoder_scope();  // single encoder
///       // All ops: one encoder, one command buffer
///   }
class CommandPipeline::EncoderScope {
public:
    explicit EncoderScope(CommandPipeline& pipeline);
    ~EncoderScope();

    // Non-copyable, non-movable
    EncoderScope(const EncoderScope&) = delete;
    EncoderScope& operator=(const EncoderScope&) = delete;
    EncoderScope(EncoderScope&&) = delete;
    EncoderScope& operator=(EncoderScope&&) = delete;

private:
    CommandPipeline& pipeline_;
    bool previous_encoder_state_;
    bool previous_lazy_state_;
};

} // namespace metal_native
