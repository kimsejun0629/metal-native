/// @file elementwise_kernels.metal
/// @brief Element-wise arithmetic and math kernels for Metal GPU compute.
/// @note All kernels use float4/half4 vectorization for 2-4x throughput improvement.

#include <metal_stdlib>
#include "common/metal_types.h"

using namespace metal;

// ---------------------------------------------------------------------------
// Binary element-wise: add
// ---------------------------------------------------------------------------

kernel void add_fp32(device const float* a      [[buffer(0)]],
                     device const float* b      [[buffer(1)]],
                     device float*       output [[buffer(2)]],
                     constant uint&      num_elements [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        // Process 4 elements - compiler will vectorize this
        float4 va = float4(a[idx], a[idx+1], a[idx+2], a[idx+3]);
        float4 vb = float4(b[idx], b[idx+1], b[idx+2], b[idx+3]);
        float4 vr = va + vb;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        // Tail: handle remaining elements
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = a[i] + b[i];
        }
    }
}

kernel void add_fp16(device const half* a      [[buffer(0)]],
                     device const half* b      [[buffer(1)]],
                     device half*       output [[buffer(2)]],
                     constant uint&     num_elements [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        half4 va = half4(a[idx], a[idx+1], a[idx+2], a[idx+3]);
        half4 vb = half4(b[idx], b[idx+1], b[idx+2], b[idx+3]);
        half4 vr = va + vb;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = a[i] + b[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Binary element-wise: sub
// ---------------------------------------------------------------------------

kernel void sub_fp32(device const float* a      [[buffer(0)]],
                     device const float* b      [[buffer(1)]],
                     device float*       output [[buffer(2)]],
                     constant uint&      num_elements [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        float4 va = float4(a[idx], a[idx+1], a[idx+2], a[idx+3]);
        float4 vb = float4(b[idx], b[idx+1], b[idx+2], b[idx+3]);
        float4 vr = va - vb;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = a[i] - b[i];
        }
    }
}

kernel void sub_fp16(device const half* a      [[buffer(0)]],
                     device const half* b      [[buffer(1)]],
                     device half*       output [[buffer(2)]],
                     constant uint&     num_elements [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        half4 va = half4(a[idx], a[idx+1], a[idx+2], a[idx+3]);
        half4 vb = half4(b[idx], b[idx+1], b[idx+2], b[idx+3]);
        half4 vr = va - vb;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = a[i] - b[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Binary element-wise: mul
// ---------------------------------------------------------------------------

kernel void mul_fp32(device const float* a      [[buffer(0)]],
                     device const float* b      [[buffer(1)]],
                     device float*       output [[buffer(2)]],
                     constant uint&      num_elements [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        float4 va = float4(a[idx], a[idx+1], a[idx+2], a[idx+3]);
        float4 vb = float4(b[idx], b[idx+1], b[idx+2], b[idx+3]);
        float4 vr = va * vb;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = a[i] * b[i];
        }
    }
}

kernel void mul_fp16(device const half* a      [[buffer(0)]],
                     device const half* b      [[buffer(1)]],
                     device half*       output [[buffer(2)]],
                     constant uint&     num_elements [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        half4 va = half4(a[idx], a[idx+1], a[idx+2], a[idx+3]);
        half4 vb = half4(b[idx], b[idx+1], b[idx+2], b[idx+3]);
        half4 vr = va * vb;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = a[i] * b[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Binary element-wise: div
// ---------------------------------------------------------------------------

kernel void div_fp32(device const float* a      [[buffer(0)]],
                     device const float* b      [[buffer(1)]],
                     device float*       output [[buffer(2)]],
                     constant uint&      num_elements [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        float4 va = float4(a[idx], a[idx+1], a[idx+2], a[idx+3]);
        float4 vb = float4(b[idx], b[idx+1], b[idx+2], b[idx+3]);
        float4 vr = va / vb;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = a[i] / b[i];
        }
    }
}

kernel void div_fp16(device const half* a      [[buffer(0)]],
                     device const half* b      [[buffer(1)]],
                     device half*       output [[buffer(2)]],
                     constant uint&     num_elements [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        half4 va = half4(a[idx], a[idx+1], a[idx+2], a[idx+3]);
        half4 vb = half4(b[idx], b[idx+1], b[idx+2], b[idx+3]);
        half4 vr = va / vb;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = a[i] / b[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Unary element-wise: exp
// ---------------------------------------------------------------------------

kernel void exp_fp32(device const float* input  [[buffer(0)]],
                     device float*       output [[buffer(1)]],
                     constant uint&      num_elements [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        float4 vi = float4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        float4 vr = exp(vi);
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = exp(input[i]);
        }
    }
}

kernel void exp_fp16(device const half* input  [[buffer(0)]],
                     device half*       output [[buffer(1)]],
                     constant uint&     num_elements [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        half4 vi = half4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        half4 vr = exp(vi);
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = exp(input[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Unary element-wise: log
// ---------------------------------------------------------------------------

kernel void log_fp32(device const float* input  [[buffer(0)]],
                     device float*       output [[buffer(1)]],
                     constant uint&      num_elements [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        float4 vi = float4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        float4 vr = log(vi);
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = log(input[i]);
        }
    }
}

kernel void log_fp16(device const half* input  [[buffer(0)]],
                     device half*       output [[buffer(1)]],
                     constant uint&     num_elements [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        half4 vi = half4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        half4 vr = log(vi);
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = log(input[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar element-wise: add_scalar
// ---------------------------------------------------------------------------

kernel void add_scalar_fp32(device const float* input  [[buffer(0)]],
                            device float*       output [[buffer(1)]],
                            device const float* scalar [[buffer(2)]],
                            constant uint&      num_elements [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    const float s = scalar[0];
    if (idx + 3 < num_elements) {
        float4 vi = float4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        float4 vr = vi + s;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = input[i] + s;
        }
    }
}

kernel void add_scalar_fp16(device const half* input  [[buffer(0)]],
                            device half*       output [[buffer(1)]],
                            device const half* scalar [[buffer(2)]],
                            constant uint&     num_elements [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    const half s = scalar[0];
    if (idx + 3 < num_elements) {
        half4 vi = half4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        half4 vr = vi + s;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = input[i] + s;
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar element-wise: mul_scalar
// ---------------------------------------------------------------------------

kernel void mul_scalar_fp32(device const float* input  [[buffer(0)]],
                            device float*       output [[buffer(1)]],
                            device const float* scalar [[buffer(2)]],
                            constant uint&      num_elements [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    const float s = scalar[0];
    if (idx + 3 < num_elements) {
        float4 vi = float4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        float4 vr = vi * s;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = input[i] * s;
        }
    }
}

kernel void mul_scalar_fp16(device const half* input  [[buffer(0)]],
                            device half*       output [[buffer(1)]],
                            device const half* scalar [[buffer(2)]],
                            constant uint&     num_elements [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    const half s = scalar[0];
    if (idx + 3 < num_elements) {
        half4 vi = half4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        half4 vr = vi * s;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = input[i] * s;
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar element-wise: sub_scalar
// ---------------------------------------------------------------------------

kernel void sub_scalar_fp32(device const float* input  [[buffer(0)]],
                            device float*       output [[buffer(1)]],
                            device const float* scalar [[buffer(2)]],
                            constant uint&      num_elements [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    const float s = scalar[0];
    if (idx + 3 < num_elements) {
        float4 vi = float4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        float4 vr = vi - s;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = input[i] - s;
        }
    }
}

kernel void sub_scalar_fp16(device const half* input  [[buffer(0)]],
                            device half*       output [[buffer(1)]],
                            device const half* scalar [[buffer(2)]],
                            constant uint&     num_elements [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    const half s = scalar[0];
    if (idx + 3 < num_elements) {
        half4 vi = half4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        half4 vr = vi - s;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = input[i] - s;
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar element-wise: div_scalar
// ---------------------------------------------------------------------------

kernel void div_scalar_fp32(device const float* input  [[buffer(0)]],
                            device float*       output [[buffer(1)]],
                            device const float* scalar [[buffer(2)]],
                            constant uint&      num_elements [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    const float s = scalar[0];
    if (idx + 3 < num_elements) {
        float4 vi = float4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        float4 vr = vi / s;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = input[i] / s;
        }
    }
}

kernel void div_scalar_fp16(device const half* input  [[buffer(0)]],
                            device half*       output [[buffer(1)]],
                            device const half* scalar [[buffer(2)]],
                            constant uint&     num_elements [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    const half s = scalar[0];
    if (idx + 3 < num_elements) {
        half4 vi = half4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        half4 vr = vi / s;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = input[i] / s;
        }
    }
}

// ---------------------------------------------------------------------------
// Unary element-wise: sqrt
// ---------------------------------------------------------------------------

kernel void sqrt_fp32(device const float* input  [[buffer(0)]],
                      device float*       output [[buffer(1)]],
                      constant uint&      num_elements [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        float4 vi = float4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        float4 vr = sqrt(vi);
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = sqrt(input[i]);
        }
    }
}

kernel void sqrt_fp16(device const half* input  [[buffer(0)]],
                      device half*       output [[buffer(1)]],
                      constant uint&     num_elements [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        half4 vi = half4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        half4 vr = sqrt(vi);
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = sqrt(input[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Unary element-wise: abs
// ---------------------------------------------------------------------------

kernel void abs_fp32(device const float* input  [[buffer(0)]],
                     device float*       output [[buffer(1)]],
                     constant uint&      num_elements [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        float4 vi = float4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        float4 vr = abs(vi);
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = abs(input[i]);
        }
    }
}

kernel void abs_fp16(device const half* input  [[buffer(0)]],
                     device half*       output [[buffer(1)]],
                     constant uint&     num_elements [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        half4 vi = half4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        half4 vr = abs(vi);
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = abs(input[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Unary element-wise: neg
// ---------------------------------------------------------------------------

kernel void neg_fp32(device const float* input  [[buffer(0)]],
                     device float*       output [[buffer(1)]],
                     constant uint&      num_elements [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        float4 vi = float4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        float4 vr = -vi;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = -input[i];
        }
    }
}

kernel void neg_fp16(device const half* input  [[buffer(0)]],
                     device half*       output [[buffer(1)]],
                     constant uint&     num_elements [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        half4 vi = half4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        half4 vr = -vi;
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = -input[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Conditional: clamp
// ---------------------------------------------------------------------------

kernel void clamp_fp32(device const float* input   [[buffer(0)]],
                       device float*       output  [[buffer(1)]],
                       constant float*     bounds  [[buffer(2)]],  // [min, max]
                       constant uint&      num_elements [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    const float min_val = bounds[0];
    const float max_val = bounds[1];
    if (idx + 3 < num_elements) {
        float4 vi = float4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        float4 vr = clamp(vi, min_val, max_val);
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = clamp(input[i], min_val, max_val);
        }
    }
}

kernel void clamp_fp16(device const half* input   [[buffer(0)]],
                       device half*       output  [[buffer(1)]],
                       constant float*    bounds  [[buffer(2)]],
                       constant uint&     num_elements [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    const half min_val = half(bounds[0]);
    const half max_val = half(bounds[1]);
    if (idx + 3 < num_elements) {
        half4 vi = half4(input[idx], input[idx+1], input[idx+2], input[idx+3]);
        half4 vr = clamp(vi, min_val, max_val);
        output[idx]   = vr.x;
        output[idx+1] = vr.y;
        output[idx+2] = vr.z;
        output[idx+3] = vr.w;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = clamp(input[i], min_val, max_val);
        }
    }
}

// ---------------------------------------------------------------------------
// Conditional: where
// ---------------------------------------------------------------------------

kernel void where_fp32(device const bool*  condition [[buffer(0)]],
                       device const float* x         [[buffer(1)]],
                       device const float* y         [[buffer(2)]],
                       device float*       output    [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    output[id] = condition[id] ? x[id] : y[id];
}

kernel void where_fp16(device const bool* condition [[buffer(0)]],
                       device const half* x         [[buffer(1)]],
                       device const half* y         [[buffer(2)]],
                       device half*       output    [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    output[id] = condition[id] ? x[id] : y[id];
}
