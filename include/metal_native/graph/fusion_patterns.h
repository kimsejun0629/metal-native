#pragma once

/// @file fusion_patterns.h
/// @brief Pattern matching for operation fusion in MPSGraph.
///
/// FusionRegistry maintains a catalog of fusible operation sequences and
/// provides pattern matching to identify opportunities for fusion. Fusion
/// reduces memory traffic and kernel launch overhead by combining multiple
/// operations into a single compiled unit.
///
/// Built-in patterns include:
///   - Conv2D + BatchNorm + ReLU → Fused conv
///   - Linear + GELU → Fused linear activation
///   - MatMul + Bias + Activation → Fused gemm
///
/// Thread-safe: the registry uses a mutex for concurrent access.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace metal_native {

/// Operation type identifier for pattern matching.
enum class OpType : uint8_t {
    Unknown = 0,
    Conv2D,
    BatchNorm,
    ReLU,
    GELU,
    Softmax,
    MatMul,
    Linear,
    BiasAdd,
    LayerNorm,
    Add,
    Mul,
    SiLU,
    RMSNorm,
    FusedResidualRMSNorm,
    RoPE,
    SwiGLU,
    Residual,
    Sub,
    Div,
    Exp,
    Log,
    Neg,
    Abs,
    Sqrt,
};

/// Convert OpType to human-readable string.
const char* op_type_name(OpType type) noexcept;

/// Descriptor for a fusible operation sequence.
struct FusionPattern {
    /// Sequence of operation types that can be fused.
    std::vector<OpType> sequence;

    /// Name of the fused operation (for debugging/profiling).
    std::string fused_name;

    /// Estimated speedup factor (used for prioritizing patterns).
    float speedup_factor = 1.0f;

    FusionPattern() = default;
    FusionPattern(std::vector<OpType> seq,
                  std::string name,
                  float speedup = 1.0f)
        : sequence(std::move(seq)),
          fused_name(std::move(name)),
          speedup_factor(speedup) {}
};

/// Result of a pattern match attempt.
struct FusionMatch {
    /// True if a pattern was matched.
    bool matched = false;

    /// Index in the operation sequence where the match starts.
    size_t start_index = 0;

    /// Number of operations consumed by this match.
    size_t length = 0;

    /// The matched pattern (nullptr if no match).
    const FusionPattern* pattern = nullptr;
};

/// Singleton registry of fusible operation patterns.
class FusionRegistry {
public:
    /// Access the global fusion registry.
    static FusionRegistry& instance();

    // Non-copyable, non-movable.
    FusionRegistry(const FusionRegistry&) = delete;
    FusionRegistry& operator=(const FusionRegistry&) = delete;
    FusionRegistry(FusionRegistry&&) = delete;
    FusionRegistry& operator=(FusionRegistry&&) = delete;

    // -- Pattern registration ------------------------------------------------

    /// Register a new fusion pattern.
    ///
    /// Patterns are matched in the order they are registered, so register
    /// longer/more specific patterns first.
    void register_pattern(FusionPattern pattern);

    /// Clear all registered patterns.
    void clear_patterns();

    // -- Pattern matching ----------------------------------------------------

    /// Attempt to match a fusion pattern starting at the given index in
    /// the operation sequence.
    ///
    /// @param ops          Sequence of operation types.
    /// @param start_index  Index to begin matching from.
    /// @return             Match result (empty if no pattern matched).
    FusionMatch try_fuse(const std::vector<OpType>& ops,
                         size_t start_index = 0) const;

    /// Scan the entire operation sequence and return all fusion opportunities.
    ///
    /// This performs a greedy left-to-right scan, consuming the longest
    /// matching pattern at each position.
    ///
    /// @param ops  Sequence of operation types.
    /// @return     Vector of all fusion matches found.
    std::vector<FusionMatch> find_all_fusions(const std::vector<OpType>& ops) const;

    // -- Built-in patterns ---------------------------------------------------

    /// Register the default set of fusion patterns.
    ///
    /// This includes common patterns like Conv+BN+ReLU, Linear+GELU, etc.
    void register_builtin_patterns();

    // -- Statistics ----------------------------------------------------------

    /// Number of registered patterns.
    size_t pattern_count() const;

private:
    FusionRegistry();
    ~FusionRegistry();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
