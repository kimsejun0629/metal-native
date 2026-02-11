/// @file activation_kernels.metal
/// @brief Activation function kernels for Metal GPU compute.
/// @note All kernels use float4/half4 vectorization for improved throughput.

#include <metal_stdlib>
#include "common/metal_types.h"
#include "common/math_utils.h"

using namespace metal;

// ---------------------------------------------------------------------------
// Helper functions for vectorized activations
// ---------------------------------------------------------------------------

/// Fast GELU approximation for float4: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline float4 gelu_fast_vec(float4 x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    float4 x3 = x * x * x;
    float4 inner = sqrt_2_over_pi * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanh(inner));
}

/// Fast GELU approximation for half4
inline half4 gelu_fast_vec(half4 x) {
    const half sqrt_2_over_pi = 0.7978845608h;
    half4 x3 = x * x * x;
    half4 inner = sqrt_2_over_pi * (x + 0.044715h * x3);
    return 0.5h * x * (1.0h + tanh(inner));
}

/// SiLU (Swish) for float4: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
inline float4 silu_vec(float4 x) {
    return x / (1.0f + exp(-x));
}

/// SiLU (Swish) for half4
inline half4 silu_vec(half4 x) {
    return x / (1.0h + exp(-x));
}

// ---------------------------------------------------------------------------
// ReLU kernels
// ---------------------------------------------------------------------------

kernel void relu_fp32(device const float* input  [[buffer(0)]],
                      device float*       output [[buffer(1)]],
                      constant uint&      num_elements [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        // Vectorized path: process 4 elements at once
        float4 vi = float4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        float4 vr = max(vi, float4(0.0f));
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        // Tail handling: process remaining elements one by one
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = max(input[i], 0.0f);
        }
    }
}

kernel void relu_fp16(device const half* input  [[buffer(0)]],
                      device half*       output [[buffer(1)]],
                      constant uint&     num_elements [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        // Vectorized path: process 4 elements at once
        half4 vi = half4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        half4 vr = max(vi, half4(0.0h));
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        // Tail handling: process remaining elements one by one
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = max(input[i], 0.0h);
        }
    }
}

// ---------------------------------------------------------------------------
// GELU kernels (fast tanh approximation)
// ---------------------------------------------------------------------------

kernel void gelu_fp32(device const float* input  [[buffer(0)]],
                      device float*       output [[buffer(1)]],
                      constant uint&      num_elements [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        // Vectorized path: process 4 elements at once
        float4 vi = float4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        float4 vr = gelu_fast_vec(vi);
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        // Tail handling: process remaining elements one by one
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = gelu_fast(input[i]);
        }
    }
}

kernel void gelu_fp16(device const half* input  [[buffer(0)]],
                      device half*       output [[buffer(1)]],
                      constant uint&     num_elements [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        // Vectorized path: process 4 elements at once
        half4 vi = half4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        half4 vr = gelu_fast_vec(vi);
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        // Tail handling: process remaining elements one by one
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = gelu_fast(input[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// SiLU (Swish) kernels
// ---------------------------------------------------------------------------

kernel void silu_fp32(device const float* input  [[buffer(0)]],
                      device float*       output [[buffer(1)]],
                      constant uint&      num_elements [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        // Vectorized path: process 4 elements at once
        float4 vi = float4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        float4 vr = silu_vec(vi);
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        // Tail handling: process remaining elements one by one
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = silu(input[i]);
        }
    }
}

kernel void silu_fp16(device const half* input  [[buffer(0)]],
                      device half*       output [[buffer(1)]],
                      constant uint&     num_elements [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        // Vectorized path: process 4 elements at once
        half4 vi = half4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        half4 vr = silu_vec(vi);
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        // Tail handling: process remaining elements one by one
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = silu(input[i]);
        }
    }
}
