#pragma once

/// @file kernel_registry.h
/// @brief Singleton registry mapping kernel names to MTLComputePipelineState.
///
/// The KernelRegistry loads compiled Metal libraries (.metallib) and creates
/// pipeline state objects on demand.  Pipeline states are cached after first
/// creation so subsequent lookups are O(1).

#include <cstddef>
#include <memory>
#include <string>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace metal_native {

class KernelRegistry {
public:
    // -- Singleton access ----------------------------------------------------

    /// Return the process-wide KernelRegistry instance.
    static KernelRegistry& instance();

    // -- Registration --------------------------------------------------------

    /// Register a kernel function name that can be looked up later.
    /// The function must exist in a loaded Metal library.
    void register_kernel(const std::string& name, const std::string& function_name);

    // -- Pipeline access -----------------------------------------------------

#ifdef __OBJC__
    /// Get (or create and cache) the MTLComputePipelineState for a kernel.
    /// Throws MNException(KernelCompilationFailed) if the function is not
    /// found or pipeline creation fails.
    id<MTLComputePipelineState> get_pipeline(const std::string& name);
#else
    /// Opaque handle to the pipeline state (cast in .mm translation units).
    void* get_pipeline(const std::string& name);
#endif

    // -- Library loading -----------------------------------------------------

    /// Load kernels from a .metallib file at the given path.
    /// Throws MNException(KernelCompilationFailed) on failure.
    void load_library(const std::string& path);

    /// Load the default Metal library compiled into the application bundle.
    /// Throws MNException(KernelCompilationFailed) if no default library exists.
    void load_default_library();

    // -- Precompilation ------------------------------------------------------

    /// Precompile all registered kernel pipelines asynchronously.
    /// This eliminates cold-start latency on first kernel invocation.
    /// Compilation errors are logged but do not throw exceptions.
    /// Should be called after load_library() or load_default_library().
    void precompile_pipelines();

    // -- Queries -------------------------------------------------------------

    /// Number of registered kernel entries.
    size_t size() const;

    // -- Non-copyable / non-movable ------------------------------------------
    KernelRegistry(const KernelRegistry&) = delete;
    KernelRegistry& operator=(const KernelRegistry&) = delete;
    KernelRegistry(KernelRegistry&&) = delete;
    KernelRegistry& operator=(KernelRegistry&&) = delete;

private:
    KernelRegistry();
    ~KernelRegistry();

    /// Attempt to load precompiled metallib automatically on first use.
    /// Called once via std::call_once from get_pipeline().
    void load_precompiled_library();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
