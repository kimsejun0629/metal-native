#pragma once

/// @file kernel_launch.h
/// @brief Launch configuration and dispatch helpers for Metal compute kernels.
///
/// Provides utilities to compute optimal threadgroup sizes and grid dimensions,
/// and a convenience function to encode a compute dispatch.

#include <cstddef>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace metal_native {

/// Grid and threadgroup sizes for a compute dispatch.
struct LaunchConfig {
#ifdef __OBJC__
    MTLSize grid_size;
    MTLSize threadgroup_size;
#else
    struct { size_t width, height, depth; } grid_size;
    struct { size_t width, height, depth; } threadgroup_size;
#endif
};

// -- Config builders ---------------------------------------------------------

#ifdef __OBJC__

/// Compute an optimal 1D launch configuration.
///
/// The threadgroup size is clamped to the pipeline's maxTotalThreadsPerThreadgroup
/// and rounded to the pipeline's threadExecutionWidth.  The grid is sized to
/// cover @p total_elements.
LaunchConfig compute_launch_config_1d(id<MTLComputePipelineState> pipeline,
                                      size_t total_elements);

/// Compute an optimal 2D launch configuration.
///
/// Each threadgroup dimension is chosen so that the product does not exceed
/// the pipeline's maximum, and the grid covers (width x height).
LaunchConfig compute_launch_config_2d(id<MTLComputePipelineState> pipeline,
                                      size_t width, size_t height);

/// Dispatch a compute kernel with the given launch configuration.
///
/// Calls setComputePipelineState and dispatchThreads on the encoder.
void dispatch_kernel(id<MTLComputeCommandEncoder> encoder,
                     id<MTLComputePipelineState> pipeline,
                     const LaunchConfig& config);

#else

// Opaque signatures for pure-C++ translation units.
LaunchConfig compute_launch_config_1d(void* pipeline, size_t total_elements);
LaunchConfig compute_launch_config_2d(void* pipeline, size_t width, size_t height);
void dispatch_kernel(void* encoder, void* pipeline, const LaunchConfig& config);

#endif

} // namespace metal_native
