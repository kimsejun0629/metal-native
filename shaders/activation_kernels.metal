/// @file activation_kernels.metal
/// @brief Activation function kernels for Metal GPU compute.

#include <metal_stdlib>
#include "common/metal_types.h"
#include "common/math_utils.h"

using namespace metal;

// ---------------------------------------------------------------------------
// ReLU kernels
// ---------------------------------------------------------------------------

kernel void relu_fp32(device const float* input  [[buffer(0)]],
                      device float*       output [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = max(input[id], 0.0f);
}

kernel void relu_fp16(device const half* input  [[buffer(0)]],
                      device half*       output [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = max(input[id], (half)0.0h);
}

// ---------------------------------------------------------------------------
// GELU kernels (fast tanh approximation)
// ---------------------------------------------------------------------------

kernel void gelu_fp32(device const float* input  [[buffer(0)]],
                      device float*       output [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = gelu_fast(input[id]);
}

kernel void gelu_fp16(device const half* input  [[buffer(0)]],
                      device half*       output [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = gelu_fast(input[id]);
}

// ---------------------------------------------------------------------------
// SiLU (Swish) kernels
// ---------------------------------------------------------------------------

kernel void silu_fp32(device const float* input  [[buffer(0)]],
                      device float*       output [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = silu(input[id]);
}

kernel void silu_fp16(device const half* input  [[buffer(0)]],
                      device half*       output [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = silu(input[id]);
}
