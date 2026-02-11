/// @file kernel_launch.mm
/// @brief Objective-C++ implementation of launch configuration helpers.

#import <Metal/Metal.h>

#include "metal_native/kernels/kernel_launch.h"
#include "metal_native/core/error.h"

#include <algorithm>
#include <cmath>

namespace metal_native {

// ---------------------------------------------------------------------------
// 1D launch configuration
// ---------------------------------------------------------------------------

LaunchConfig compute_launch_config_1d(id<MTLComputePipelineState> pipeline,
                                      size_t total_elements) {
    MN_CHECK(pipeline != nil,
             MetalNativeError::InvalidArgument,
             "compute_launch_config_1d: pipeline must not be nil");
    MN_CHECK(total_elements > 0,
             MetalNativeError::InvalidArgument,
             "compute_launch_config_1d: total_elements must be > 0");

    NSUInteger max_threads = [pipeline maxTotalThreadsPerThreadgroup];
    NSUInteger exec_width  = [pipeline threadExecutionWidth];

    // Choose threadgroup size as a multiple of the execution width,
    // clamped to the pipeline maximum and at most 256 (sweet spot for
    // element-wise kernels on Apple GPUs).
    NSUInteger tg_size = std::min(max_threads, static_cast<NSUInteger>(256));
    // Round down to a multiple of the execution width.
    tg_size = (tg_size / exec_width) * exec_width;
    if (tg_size == 0) tg_size = exec_width;

    LaunchConfig config;
    config.grid_size        = MTLSizeMake(total_elements, 1, 1);
    config.threadgroup_size = MTLSizeMake(tg_size, 1, 1);
    return config;
}

// ---------------------------------------------------------------------------
// 2D launch configuration
// ---------------------------------------------------------------------------

LaunchConfig compute_launch_config_2d(id<MTLComputePipelineState> pipeline,
                                      size_t width, size_t height) {
    MN_CHECK(pipeline != nil,
             MetalNativeError::InvalidArgument,
             "compute_launch_config_2d: pipeline must not be nil");
    MN_CHECK(width > 0 && height > 0,
             MetalNativeError::InvalidArgument,
             "compute_launch_config_2d: width and height must be > 0");

    NSUInteger max_threads = [pipeline maxTotalThreadsPerThreadgroup];

    // Use square-ish threadgroups.  32x32 = 1024 is ideal for transpose-
    // style kernels; clamp each dimension so the product stays within max.
    NSUInteger tg_w = 32;
    NSUInteger tg_h = 32;

    while (tg_w * tg_h > max_threads) {
        if (tg_w > tg_h) {
            tg_w /= 2;
        } else {
            tg_h /= 2;
        }
    }

    LaunchConfig config;
    config.grid_size        = MTLSizeMake(width, height, 1);
    config.threadgroup_size = MTLSizeMake(tg_w, tg_h, 1);
    return config;
}

// ---------------------------------------------------------------------------
// Dispatch helper
// ---------------------------------------------------------------------------

void dispatch_kernel(id<MTLComputeCommandEncoder> encoder,
                     id<MTLComputePipelineState> pipeline,
                     const LaunchConfig& config) {
    MN_CHECK(encoder != nil,
             MetalNativeError::InvalidArgument,
             "dispatch_kernel: encoder must not be nil");
    MN_CHECK(pipeline != nil,
             MetalNativeError::InvalidArgument,
             "dispatch_kernel: pipeline must not be nil");

    [encoder setComputePipelineState:pipeline];
    [encoder dispatchThreads:config.grid_size
       threadsPerThreadgroup:config.threadgroup_size];
}

} // namespace metal_native
