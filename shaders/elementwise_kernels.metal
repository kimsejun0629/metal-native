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
        // Coalesced 128-bit vector load
        float4 va = *reinterpret_cast<device const float4*>(a + idx);
        float4 vb = *reinterpret_cast<device const float4*>(b + idx);
        float4 vr = va + vb;
        *reinterpret_cast<device float4*>(output + idx) = vr;
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
        half4 va = *reinterpret_cast<device const half4*>(a + idx);
        half4 vb = *reinterpret_cast<device const half4*>(b + idx);
        half4 vr = va + vb;
        *reinterpret_cast<device half4*>(output + idx) = vr;
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
        float4 va = *reinterpret_cast<device const float4*>(a + idx);
        float4 vb = *reinterpret_cast<device const float4*>(b + idx);
        float4 vr = va - vb;
        *reinterpret_cast<device float4*>(output + idx) = vr;
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
        half4 va = *reinterpret_cast<device const half4*>(a + idx);
        half4 vb = *reinterpret_cast<device const half4*>(b + idx);
        half4 vr = va - vb;
        *reinterpret_cast<device half4*>(output + idx) = vr;
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
        float4 va = *reinterpret_cast<device const float4*>(a + idx);
        float4 vb = *reinterpret_cast<device const float4*>(b + idx);
        float4 vr = va * vb;
        *reinterpret_cast<device float4*>(output + idx) = vr;
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
        half4 va = *reinterpret_cast<device const half4*>(a + idx);
        half4 vb = *reinterpret_cast<device const half4*>(b + idx);
        half4 vr = va * vb;
        *reinterpret_cast<device half4*>(output + idx) = vr;
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
        float4 va = *reinterpret_cast<device const float4*>(a + idx);
        float4 vb = *reinterpret_cast<device const float4*>(b + idx);
        float4 vr = va / vb;
        *reinterpret_cast<device float4*>(output + idx) = vr;
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
        half4 va = *reinterpret_cast<device const half4*>(a + idx);
        half4 vb = *reinterpret_cast<device const half4*>(b + idx);
        half4 vr = va / vb;
        *reinterpret_cast<device half4*>(output + idx) = vr;
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
        float4 vi = *reinterpret_cast<device const float4*>(input + idx);
        float4 vr = exp(vi);
        *reinterpret_cast<device float4*>(output + idx) = vr;
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
        half4 vi = *reinterpret_cast<device const half4*>(input + idx);
        half4 vr = exp(vi);
        *reinterpret_cast<device half4*>(output + idx) = vr;
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
        float4 vi = *reinterpret_cast<device const float4*>(input + idx);
        float4 vr = log(vi);
        *reinterpret_cast<device float4*>(output + idx) = vr;
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
        half4 vi = *reinterpret_cast<device const half4*>(input + idx);
        half4 vr = log(vi);
        *reinterpret_cast<device half4*>(output + idx) = vr;
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
        float4 vi = *reinterpret_cast<device const float4*>(input + idx);
        float4 vr = vi + s;
        *reinterpret_cast<device float4*>(output + idx) = vr;
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
        half4 vi = *reinterpret_cast<device const half4*>(input + idx);
        half4 vr = vi + s;
        *reinterpret_cast<device half4*>(output + idx) = vr;
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
        float4 vi = *reinterpret_cast<device const float4*>(input + idx);
        float4 vr = vi * s;
        *reinterpret_cast<device float4*>(output + idx) = vr;
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
        half4 vi = *reinterpret_cast<device const half4*>(input + idx);
        half4 vr = vi * s;
        *reinterpret_cast<device half4*>(output + idx) = vr;
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
        float4 vi = *reinterpret_cast<device const float4*>(input + idx);
        float4 vr = vi - s;
        *reinterpret_cast<device float4*>(output + idx) = vr;
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
        half4 vi = *reinterpret_cast<device const half4*>(input + idx);
        half4 vr = vi - s;
        *reinterpret_cast<device half4*>(output + idx) = vr;
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
        float4 vi = *reinterpret_cast<device const float4*>(input + idx);
        float4 vr = vi / s;
        *reinterpret_cast<device float4*>(output + idx) = vr;
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
        half4 vi = *reinterpret_cast<device const half4*>(input + idx);
        half4 vr = vi / s;
        *reinterpret_cast<device half4*>(output + idx) = vr;
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
        float4 vi = *reinterpret_cast<device const float4*>(input + idx);
        float4 vr = sqrt(vi);
        *reinterpret_cast<device float4*>(output + idx) = vr;
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
        half4 vi = *reinterpret_cast<device const half4*>(input + idx);
        half4 vr = sqrt(vi);
        *reinterpret_cast<device half4*>(output + idx) = vr;
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
        float4 vi = *reinterpret_cast<device const float4*>(input + idx);
        float4 vr = abs(vi);
        *reinterpret_cast<device float4*>(output + idx) = vr;
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
        half4 vi = *reinterpret_cast<device const half4*>(input + idx);
        half4 vr = abs(vi);
        *reinterpret_cast<device half4*>(output + idx) = vr;
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
        float4 vi = *reinterpret_cast<device const float4*>(input + idx);
        float4 vr = -vi;
        *reinterpret_cast<device float4*>(output + idx) = vr;
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
        half4 vi = *reinterpret_cast<device const half4*>(input + idx);
        half4 vr = -vi;
        *reinterpret_cast<device half4*>(output + idx) = vr;
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
        float4 vi = *reinterpret_cast<device const float4*>(input + idx);
        float4 vr = clamp(vi, min_val, max_val);
        *reinterpret_cast<device float4*>(output + idx) = vr;
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
        half4 vi = *reinterpret_cast<device const half4*>(input + idx);
        half4 vr = clamp(vi, min_val, max_val);
        *reinterpret_cast<device half4*>(output + idx) = vr;
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

// ---------------------------------------------------------------------------
// BFloat16 element-wise kernels
// ---------------------------------------------------------------------------
// BFloat16 is stored as ushort in Metal buffers. Convert to float for
// computation, then convert back to ushort. All math done in FP32.

kernel void add_bf16(device const ushort* a      [[buffer(0)]],
                     device const ushort* b      [[buffer(1)]],
                     device ushort*       output [[buffer(2)]],
                     constant uint&       num_elements [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        ushort4 va = *reinterpret_cast<device const ushort4*>(a + idx);
        ushort4 vb = *reinterpret_cast<device const ushort4*>(b + idx);
        float4 fa = bf16x4_to_float4(va);
        float4 fb = bf16x4_to_float4(vb);
        float4 fr = fa + fb;
        ushort4 vr = float4_to_bf16x4(fr);
        *reinterpret_cast<device ushort4*>(output + idx) = vr;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            float result = bf16_to_float(a[i]) + bf16_to_float(b[i]);
            output[i] = float_to_bf16(result);
        }
    }
}

kernel void sub_bf16(device const ushort* a      [[buffer(0)]],
                     device const ushort* b      [[buffer(1)]],
                     device ushort*       output [[buffer(2)]],
                     constant uint&       num_elements [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        ushort4 va = *reinterpret_cast<device const ushort4*>(a + idx);
        ushort4 vb = *reinterpret_cast<device const ushort4*>(b + idx);
        float4 fa = bf16x4_to_float4(va);
        float4 fb = bf16x4_to_float4(vb);
        float4 fr = fa - fb;
        ushort4 vr = float4_to_bf16x4(fr);
        *reinterpret_cast<device ushort4*>(output + idx) = vr;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            float result = bf16_to_float(a[i]) - bf16_to_float(b[i]);
            output[i] = float_to_bf16(result);
        }
    }
}

kernel void mul_bf16(device const ushort* a      [[buffer(0)]],
                     device const ushort* b      [[buffer(1)]],
                     device ushort*       output [[buffer(2)]],
                     constant uint&       num_elements [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        ushort4 va = *reinterpret_cast<device const ushort4*>(a + idx);
        ushort4 vb = *reinterpret_cast<device const ushort4*>(b + idx);
        float4 fa = bf16x4_to_float4(va);
        float4 fb = bf16x4_to_float4(vb);
        float4 fr = fa * fb;
        ushort4 vr = float4_to_bf16x4(fr);
        *reinterpret_cast<device ushort4*>(output + idx) = vr;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            float result = bf16_to_float(a[i]) * bf16_to_float(b[i]);
            output[i] = float_to_bf16(result);
        }
    }
}

kernel void div_bf16(device const ushort* a      [[buffer(0)]],
                     device const ushort* b      [[buffer(1)]],
                     device ushort*       output [[buffer(2)]],
                     constant uint&       num_elements [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        ushort4 va = *reinterpret_cast<device const ushort4*>(a + idx);
        ushort4 vb = *reinterpret_cast<device const ushort4*>(b + idx);
        float4 fa = bf16x4_to_float4(va);
        float4 fb = bf16x4_to_float4(vb);
        float4 fr = fa / fb;
        ushort4 vr = float4_to_bf16x4(fr);
        *reinterpret_cast<device ushort4*>(output + idx) = vr;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            float result = bf16_to_float(a[i]) / bf16_to_float(b[i]);
            output[i] = float_to_bf16(result);
        }
    }
}

kernel void exp_bf16(device const ushort* input  [[buffer(0)]],
                     device ushort*       output [[buffer(1)]],
                     constant uint&       num_elements [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        ushort4 vi = *reinterpret_cast<device const ushort4*>(input + idx);
        float4 fi = bf16x4_to_float4(vi);
        float4 fr = exp(fi);
        ushort4 vr = float4_to_bf16x4(fr);
        *reinterpret_cast<device ushort4*>(output + idx) = vr;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = float_to_bf16(exp(bf16_to_float(input[i])));
        }
    }
}

kernel void log_bf16(device const ushort* input  [[buffer(0)]],
                     device ushort*       output [[buffer(1)]],
                     constant uint&       num_elements [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        ushort4 vi = *reinterpret_cast<device const ushort4*>(input + idx);
        float4 fi = bf16x4_to_float4(vi);
        float4 fr = log(fi);
        ushort4 vr = float4_to_bf16x4(fr);
        *reinterpret_cast<device ushort4*>(output + idx) = vr;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = float_to_bf16(log(bf16_to_float(input[i])));
        }
    }
}

kernel void neg_bf16(device const ushort* input  [[buffer(0)]],
                     device ushort*       output [[buffer(1)]],
                     constant uint&       num_elements [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        ushort4 vi = *reinterpret_cast<device const ushort4*>(input + idx);
        float4 fi = bf16x4_to_float4(vi);
        float4 fr = -fi;
        ushort4 vr = float4_to_bf16x4(fr);
        *reinterpret_cast<device ushort4*>(output + idx) = vr;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = float_to_bf16(-bf16_to_float(input[i]));
        }
    }
}

kernel void abs_bf16(device const ushort* input  [[buffer(0)]],
                     device ushort*       output [[buffer(1)]],
                     constant uint&       num_elements [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        ushort4 vi = *reinterpret_cast<device const ushort4*>(input + idx);
        float4 fi = bf16x4_to_float4(vi);
        float4 fr = abs(fi);
        ushort4 vr = float4_to_bf16x4(fr);
        *reinterpret_cast<device ushort4*>(output + idx) = vr;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = float_to_bf16(abs(bf16_to_float(input[i])));
        }
    }
}

kernel void sqrt_bf16(device const ushort* input  [[buffer(0)]],
                      device ushort*       output [[buffer(1)]],
                      constant uint&       num_elements [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        ushort4 vi = *reinterpret_cast<device const ushort4*>(input + idx);
        float4 fi = bf16x4_to_float4(vi);
        float4 fr = sqrt(fi);
        ushort4 vr = float4_to_bf16x4(fr);
        *reinterpret_cast<device ushort4*>(output + idx) = vr;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = float_to_bf16(sqrt(bf16_to_float(input[i])));
        }
    }
}

kernel void clamp_bf16(device const ushort* input   [[buffer(0)]],
                       device ushort*       output  [[buffer(1)]],
                       constant float*      bounds  [[buffer(2)]],  // [min, max]
                       constant uint&       num_elements [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    const float min_val = bounds[0];
    const float max_val = bounds[1];
    if (idx + 3 < num_elements) {
        ushort4 vi = *reinterpret_cast<device const ushort4*>(input + idx);
        float4 fi = bf16x4_to_float4(vi);
        float4 fr = clamp(fi, min_val, max_val);
        ushort4 vr = float4_to_bf16x4(fr);
        *reinterpret_cast<device ushort4*>(output + idx) = vr;
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = float_to_bf16(clamp(bf16_to_float(input[i]), min_val, max_val));
        }
    }
}

kernel void where_bf16(device const bool*   condition [[buffer(0)]],
                       device const ushort* x         [[buffer(1)]],
                       device const ushort* y         [[buffer(2)]],
                       device ushort*       output    [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    output[id] = condition[id] ? x[id] : y[id];
}
