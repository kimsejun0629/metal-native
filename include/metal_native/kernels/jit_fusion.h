#pragma once

/// @file jit_fusion.h
/// @brief Runtime JIT fusion compiler for consecutive elementwise operations.
///
/// JITFusionCompiler detects fusible sequences of elementwise operations,
/// generates vectorized MSL kernel source code at runtime, compiles it
/// via [MTLDevice newLibraryWithSource:], and caches the compiled pipeline.
///
/// On first invocation with a new operation sequence, falls back to
/// individual kernels while compiling the fused kernel on a background
/// thread. Subsequent invocations use the cached fused pipeline.
///
/// Supported ops: add, sub, mul, div, exp, log, neg, abs, sqrt,
///                relu, gelu, silu, tanh, sigmoid
/// Max chain length: 8 ops
/// Only same-shape, same-dtype, contiguous tensors (no broadcasting)
///
/// Thread-safe: all public methods are guarded by a mutex.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace metal_native {

class MNDevice;

/// Identifies an elementwise operation for JIT fusion.
enum class FusedOp : uint8_t {
    Add = 0,   // binary: a + b
    Sub,       // binary: a - b
    Mul,       // binary: a * b
    Div,       // binary: a / b
    Exp,       // unary: exp(x)
    Log,       // unary: log(x)
    Neg,       // unary: -x
    Abs,       // unary: abs(x)
    Sqrt,      // unary: sqrt(x)
    ReLU,      // unary: max(x, 0)
    GELU,      // unary: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    SiLU,      // unary: x * sigmoid(x)
    Tanh,      // unary: tanh(x)
    Sigmoid,   // unary: 1 / (1 + exp(-x))
};

/// Whether a FusedOp is unary or binary.
bool is_unary_op(FusedOp op);

/// Get MSL expression for a FusedOp (used in code generation).
/// For unary: returns expression with placeholder "x"
/// For binary: returns expression with placeholders "a", "b"
const char* fused_op_msl_expr(FusedOp op);

/// Get human-readable name for a FusedOp.
const char* fused_op_name(FusedOp op);

/// Describes the dtype for generated code.
enum class FusedDType : uint8_t {
    Float32 = 0,
    Float16 = 1,
};

/// Key for the fusion cache: sequence of ops + dtype.
struct FusionKey {
    std::vector<FusedOp> ops;
    FusedDType dtype;
    uint32_t num_inputs;  // total number of unique input tensors

    bool operator==(const FusionKey& other) const;
};

/// Hash for FusionKey.
struct FusionKeyHash {
    size_t operator()(const FusionKey& k) const;
};

class JITFusionCompiler {
public:
    /// Access the singleton JITFusionCompiler.
    static JITFusionCompiler& instance();

    // Non-copyable, non-movable.
    JITFusionCompiler(const JITFusionCompiler&) = delete;
    JITFusionCompiler& operator=(const JITFusionCompiler&) = delete;

    /// Check if JIT fusion is enabled.
    bool enabled() const noexcept;

    /// Enable/disable JIT fusion.
    void set_enabled(bool enable);

    /// Check if a fused pipeline is available for the given key.
    /// Returns true if cached and ready to use.
    bool has_cached(const FusionKey& key) const;

    /// Get or compile a fused pipeline for the given operation sequence.
    /// If not cached, triggers async compilation and returns nullptr.
    /// Caller should fall back to individual kernels when nullptr is returned.
#ifdef __OBJC__
    id<MTLComputePipelineState> get_pipeline(const FusionKey& key);
#else
    void* get_pipeline(const FusionKey& key);
#endif

    /// Generate MSL source code for a fused kernel.
    /// This is public for testing/debugging purposes.
    std::string generate_msl(const FusionKey& key) const;

    /// Invalidate all cached pipelines.
    void invalidate_all();

    /// Number of cached pipelines.
    size_t cache_size() const;

    /// Number of compilation jobs currently in flight.
    size_t pending_compilations() const;

    /// Maximum chain length for fusion.
    static constexpr size_t MAX_CHAIN_LENGTH = 8;

    /// Maximum cache size (LRU eviction).
    static constexpr size_t MAX_CACHE_SIZE = 128;

private:
    JITFusionCompiler();
    ~JITFusionCompiler();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
