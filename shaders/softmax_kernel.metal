#include "common/metal_types.h"
#include "common/simd_utils.h"
#include <metal_stdlib>
using namespace metal;

// Numerically stable softmax using three-pass algorithm:
// Pass 1: Find maximum value
// Pass 2: Compute exp(x - max) and sum
// Pass 3: Normalize by dividing by sum

// Pass 1: Find max value along reduction dimension
kernel void softmax_find_max(
    device const float* input       [[buffer(0)]],
    device float* max_vals          [[buffer(1)]],
    constant uint& outer_size       [[buffer(2)]],  // Product of dims before reduction
    constant uint& reduce_size      [[buffer(3)]],  // Size of reduction dimension
    constant uint& inner_size       [[buffer(4)]],  // Product of dims after reduction
    uint3 gid                       [[thread_position_in_grid]]
) {
    const uint outer_idx = gid.y;
    const uint inner_idx = gid.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    const uint base_offset = outer_idx * reduce_size * inner_size + inner_idx;

    // Find max across reduction dimension
    float local_max = -INFINITY;
    for (uint i = 0; i < reduce_size; i++) {
        float val = input[base_offset + i * inner_size];
        local_max = max(local_max, val);
    }

    // Each thread writes its own result
    max_vals[outer_idx * inner_size + inner_idx] = local_max;
}

// Pass 2: Compute exp(x - max) and sum
kernel void softmax_exp_sum(
    device const float* input       [[buffer(0)]],
    device const float* max_vals    [[buffer(1)]],
    device float* exp_vals          [[buffer(2)]],
    device float* sum_exp           [[buffer(3)]],
    constant uint& outer_size       [[buffer(4)]],
    constant uint& reduce_size      [[buffer(5)]],
    constant uint& inner_size       [[buffer(6)]],
    uint3 gid                       [[thread_position_in_grid]]
) {
    const uint outer_idx = gid.y;
    const uint inner_idx = gid.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    const uint base_offset = outer_idx * reduce_size * inner_size + inner_idx;
    const float max_val = max_vals[outer_idx * inner_size + inner_idx];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (uint i = 0; i < reduce_size; i++) {
        const uint idx = base_offset + i * inner_size;
        float exp_val = exp(input[idx] - max_val);
        exp_vals[idx] = exp_val;
        local_sum += exp_val;
    }

    // Each thread writes its own result
    sum_exp[outer_idx * inner_size + inner_idx] = local_sum;
}

// Pass 3: Normalize
kernel void softmax_normalize(
    device const float* exp_vals    [[buffer(0)]],
    device const float* sum_exp     [[buffer(1)]],
    device float* output            [[buffer(2)]],
    constant uint& outer_size       [[buffer(3)]],
    constant uint& reduce_size      [[buffer(4)]],
    constant uint& inner_size       [[buffer(5)]],
    uint3 gid                       [[thread_position_in_grid]]
) {
    const uint outer_idx = gid.y;
    const uint reduce_idx = gid.z;
    const uint inner_idx = gid.x;

    if (outer_idx >= outer_size || reduce_idx >= reduce_size || inner_idx >= inner_size) {
        return;
    }

    const uint idx = outer_idx * reduce_size * inner_size + reduce_idx * inner_size + inner_idx;
    const float sum = sum_exp[outer_idx * inner_size + inner_idx];

    output[idx] = exp_vals[idx] / sum;
}

// FP16 versions with FP32 accumulation

kernel void softmax_find_max_fp16(
    device const half* input        [[buffer(0)]],
    device float* max_vals          [[buffer(1)]],
    constant uint& outer_size       [[buffer(2)]],
    constant uint& reduce_size      [[buffer(3)]],
    constant uint& inner_size       [[buffer(4)]],
    uint3 gid                       [[thread_position_in_grid]]
) {
    const uint outer_idx = gid.y;
    const uint inner_idx = gid.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    const uint base_offset = outer_idx * reduce_size * inner_size + inner_idx;

    float local_max = -INFINITY;
    for (uint i = 0; i < reduce_size; i++) {
        float val = float(input[base_offset + i * inner_size]);
        local_max = max(local_max, val);
    }

    // Each thread writes its own result
    max_vals[outer_idx * inner_size + inner_idx] = local_max;
}

kernel void softmax_exp_sum_fp16(
    device const half* input        [[buffer(0)]],
    device const float* max_vals    [[buffer(1)]],
    device half* exp_vals           [[buffer(2)]],
    device float* sum_exp           [[buffer(3)]],
    constant uint& outer_size       [[buffer(4)]],
    constant uint& reduce_size      [[buffer(5)]],
    constant uint& inner_size       [[buffer(6)]],
    uint3 gid                       [[thread_position_in_grid]]
) {
    const uint outer_idx = gid.y;
    const uint inner_idx = gid.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    const uint base_offset = outer_idx * reduce_size * inner_size + inner_idx;
    const float max_val = max_vals[outer_idx * inner_size + inner_idx];

    float local_sum = 0.0f;  // FP32 accumulation
    for (uint i = 0; i < reduce_size; i++) {
        const uint idx = base_offset + i * inner_size;
        float exp_val = exp(float(input[idx]) - max_val);
        exp_vals[idx] = half(exp_val);
        local_sum += exp_val;
    }

    // Each thread writes its own result
    sum_exp[outer_idx * inner_size + inner_idx] = local_sum;
}

kernel void softmax_normalize_fp16(
    device const half* exp_vals     [[buffer(0)]],
    device const float* sum_exp     [[buffer(1)]],
    device half* output             [[buffer(2)]],
    constant uint& outer_size       [[buffer(3)]],
    constant uint& reduce_size      [[buffer(4)]],
    constant uint& inner_size       [[buffer(5)]],
    uint3 gid                       [[thread_position_in_grid]]
) {
    const uint outer_idx = gid.y;
    const uint reduce_idx = gid.z;
    const uint inner_idx = gid.x;

    if (outer_idx >= outer_size || reduce_idx >= reduce_size || inner_idx >= inner_size) {
        return;
    }

    const uint idx = outer_idx * reduce_size * inner_size + reduce_idx * inner_size + inner_idx;
    const float sum = sum_exp[outer_idx * inner_size + inner_idx];

    output[idx] = half(float(exp_vals[idx]) / sum);
}

// Log-softmax: log(softmax(x)) = x - max - log(sum(exp(x - max)))
// More numerically stable than log(softmax(x))

kernel void log_softmax_kernel(
    device const float* input       [[buffer(0)]],
    device const float* max_vals    [[buffer(1)]],
    device const float* sum_exp     [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& outer_size       [[buffer(4)]],
    constant uint& reduce_size      [[buffer(5)]],
    constant uint& inner_size       [[buffer(6)]],
    uint3 gid                       [[thread_position_in_grid]]
) {
    const uint outer_idx = gid.y;
    const uint reduce_idx = gid.z;
    const uint inner_idx = gid.x;

    if (outer_idx >= outer_size || reduce_idx >= reduce_size || inner_idx >= inner_size) {
        return;
    }

    const uint idx = outer_idx * reduce_size * inner_size + reduce_idx * inner_size + inner_idx;
    const float max_val = max_vals[outer_idx * inner_size + inner_idx];
    const float sum = sum_exp[outer_idx * inner_size + inner_idx];

    output[idx] = input[idx] - max_val - log(sum);
}

kernel void log_softmax_kernel_fp16(
    device const half* input        [[buffer(0)]],
    device const float* max_vals    [[buffer(1)]],
    device const float* sum_exp     [[buffer(2)]],
    device half* output             [[buffer(3)]],
    constant uint& outer_size       [[buffer(4)]],
    constant uint& reduce_size      [[buffer(5)]],
    constant uint& inner_size       [[buffer(6)]],
    uint3 gid                       [[thread_position_in_grid]]
) {
    const uint outer_idx = gid.y;
    const uint reduce_idx = gid.z;
    const uint inner_idx = gid.x;

    if (outer_idx >= outer_size || reduce_idx >= reduce_size || inner_idx >= inner_size) {
        return;
    }

    const uint idx = outer_idx * reduce_size * inner_size + reduce_idx * inner_size + inner_idx;
    const float max_val = max_vals[outer_idx * inner_size + inner_idx];
    const float sum = sum_exp[outer_idx * inner_size + inner_idx];

    output[idx] = half(float(input[idx]) - max_val - log(sum));
}

// Online single-pass softmax: computes max, exp-sum, and output in one pass.
// Each thread handles one (outer_idx, inner_idx) pair.
kernel void softmax_online_fp32(
    device const float* input     [[buffer(0)]],
    device float* output          [[buffer(1)]],
    constant uint& outer_size     [[buffer(2)]],
    constant uint& reduce_size    [[buffer(3)]],
    constant uint& inner_size     [[buffer(4)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    const uint inner_idx = gid.x;
    const uint outer_idx = gid.y;
    if (inner_idx >= inner_size || outer_idx >= outer_size) return;

    const uint base = outer_idx * reduce_size * inner_size + inner_idx;
    const uint stride = inner_size;

    // Pass 1: find max (online)
    float max_val = -INFINITY;
    for (uint i = 0; i < reduce_size; i++) {
        float val = input[base + i * stride];
        max_val = max(max_val, val);
    }

    // Pass 2: compute exp and sum
    float sum_exp = 0.0f;
    for (uint i = 0; i < reduce_size; i++) {
        sum_exp += exp(input[base + i * stride] - max_val);
    }

    // Pass 3: normalize and write output
    float inv_sum = 1.0f / sum_exp;
    for (uint i = 0; i < reduce_size; i++) {
        output[base + i * stride] = exp(input[base + i * stride] - max_val) * inv_sum;
    }
}

kernel void softmax_online_fp16(
    device const half* input      [[buffer(0)]],
    device half* output           [[buffer(1)]],
    constant uint& outer_size     [[buffer(2)]],
    constant uint& reduce_size    [[buffer(3)]],
    constant uint& inner_size     [[buffer(4)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    const uint inner_idx = gid.x;
    const uint outer_idx = gid.y;
    if (inner_idx >= inner_size || outer_idx >= outer_size) return;

    const uint base = outer_idx * reduce_size * inner_size + inner_idx;
    const uint stride = inner_size;

    float max_val = -INFINITY;
    for (uint i = 0; i < reduce_size; i++) {
        max_val = max(max_val, float(input[base + i * stride]));
    }

    float sum_exp = 0.0f;
    for (uint i = 0; i < reduce_size; i++) {
        sum_exp += exp(float(input[base + i * stride]) - max_val);
    }

    float inv_sum = 1.0f / sum_exp;
    for (uint i = 0; i < reduce_size; i++) {
        output[base + i * stride] = half(exp(float(input[base + i * stride]) - max_val) * inv_sum);
    }
}

// SIMD-cooperative softmax: 32 threads cooperate on each (outer_idx, inner_idx) pair
// 2-pass design: Pass 1 = find max + sum, Pass 2 = normalize and write
kernel void softmax_simd_cooperative_fp32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    // Each threadgroup handles one (outer_idx, inner_idx) pair
    // gid.x = inner_idx, gid.y = outer_idx
    const uint inner_idx = gid.x;
    const uint outer_idx = gid.y;

    if (inner_idx >= inner_size || outer_idx >= outer_size) return;

    const uint base = outer_idx * reduce_size * inner_size + inner_idx;
    const uint stride = inner_size;  // stride between consecutive reduce elements

    // Pass 1: Find max using SIMD cooperative reduction
    float thread_max = -INFINITY;
    for (uint i = lane; i < reduce_size; i += 32) {
        thread_max = max(thread_max, input[base + i * stride]);
    }
    float row_max = simd_max(thread_max);

    // Pass 1 continued: Compute sum of exp(x - max)
    float thread_sum = 0.0f;
    for (uint i = lane; i < reduce_size; i += 32) {
        thread_sum += exp(input[base + i * stride] - row_max);
    }
    float row_sum = simd_sum(thread_sum);

    // Pass 2: Normalize and write output
    float inv_sum = 1.0f / row_sum;
    for (uint i = lane; i < reduce_size; i += 32) {
        output[base + i * stride] = exp(input[base + i * stride] - row_max) * inv_sum;
    }
}

kernel void softmax_simd_cooperative_fp16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint inner_idx = gid.x;
    const uint outer_idx = gid.y;

    if (inner_idx >= inner_size || outer_idx >= outer_size) return;

    const uint base = outer_idx * reduce_size * inner_size + inner_idx;
    const uint stride = inner_size;

    // Pass 1: Find max using SIMD cooperative reduction (FP32 accumulation)
    float thread_max = -INFINITY;
    for (uint i = lane; i < reduce_size; i += 32) {
        thread_max = max(thread_max, float(input[base + i * stride]));
    }
    float row_max = simd_max(thread_max);

    // Pass 1 continued: Compute sum of exp(x - max) (FP32 accumulation)
    float thread_sum = 0.0f;
    for (uint i = lane; i < reduce_size; i += 32) {
        thread_sum += exp(float(input[base + i * stride]) - row_max);
    }
    float row_sum = simd_sum(thread_sum);

    // Pass 2: Normalize and write output
    float inv_sum = 1.0f / row_sum;
    for (uint i = lane; i < reduce_size; i += 32) {
        output[base + i * stride] = half(exp(float(input[base + i * stride]) - row_max) * inv_sum);
    }
}

// Single-pass log_softmax: output[i] = input[i] - max - log(sum_exp)
kernel void log_softmax_online_fp32(
    device const float* input     [[buffer(0)]],
    device float* output          [[buffer(1)]],
    constant uint& outer_size     [[buffer(2)]],
    constant uint& reduce_size    [[buffer(3)]],
    constant uint& inner_size     [[buffer(4)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    const uint inner_idx = gid.x;
    const uint outer_idx = gid.y;
    if (inner_idx >= inner_size || outer_idx >= outer_size) return;

    const uint base = outer_idx * reduce_size * inner_size + inner_idx;
    const uint stride = inner_size;

    float max_val = -INFINITY;
    for (uint i = 0; i < reduce_size; i++) {
        max_val = max(max_val, input[base + i * stride]);
    }

    float sum_exp = 0.0f;
    for (uint i = 0; i < reduce_size; i++) {
        sum_exp += exp(input[base + i * stride] - max_val);
    }

    float log_sum = log(sum_exp);
    for (uint i = 0; i < reduce_size; i++) {
        output[base + i * stride] = input[base + i * stride] - max_val - log_sum;
    }
}

kernel void log_softmax_online_fp16(
    device const half* input      [[buffer(0)]],
    device half* output           [[buffer(1)]],
    constant uint& outer_size     [[buffer(2)]],
    constant uint& reduce_size    [[buffer(3)]],
    constant uint& inner_size     [[buffer(4)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    const uint inner_idx = gid.x;
    const uint outer_idx = gid.y;
    if (inner_idx >= inner_size || outer_idx >= outer_size) return;

    const uint base = outer_idx * reduce_size * inner_size + inner_idx;
    const uint stride = inner_size;

    float max_val = -INFINITY;
    for (uint i = 0; i < reduce_size; i++) {
        max_val = max(max_val, float(input[base + i * stride]));
    }

    float sum_exp = 0.0f;
    for (uint i = 0; i < reduce_size; i++) {
        sum_exp += exp(float(input[base + i * stride]) - max_val);
    }

    float log_sum = log(sum_exp);
    for (uint i = 0; i < reduce_size; i++) {
        output[base + i * stride] = half(float(input[base + i * stride]) - max_val - log_sum);
    }
}

// Vectorized softmax kernels for contiguous case (inner_size == 1)
// Each thread processes reduce_size/4 float4 elements with stride=1

kernel void softmax_online_vec4_fp32(
    device const float* input     [[buffer(0)]],
    device float* output          [[buffer(1)]],
    constant uint& outer_size     [[buffer(2)]],
    constant uint& reduce_size    [[buffer(3)]],  // must be multiple of 4
    constant uint& inner_size     [[buffer(4)]],  // must be 1
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= outer_size) return;

    const uint base = gid * reduce_size;
    const uint vec_count = reduce_size / 4;
    device const float4* in4 = reinterpret_cast<device const float4*>(input + base);
    device float4* out4 = reinterpret_cast<device float4*>(output + base);

    // Pass 1: find max
    float4 max4 = float4(-INFINITY);
    for (uint i = 0; i < vec_count; i++) {
        max4 = max(max4, in4[i]);
    }
    float max_val = max(max(max4.x, max4.y), max(max4.z, max4.w));

    // Pass 2: exp and sum
    float sum_exp = 0.0f;
    for (uint i = 0; i < vec_count; i++) {
        float4 e = exp(in4[i] - float4(max_val));
        sum_exp += e.x + e.y + e.z + e.w;
    }

    // Pass 3: normalize
    float inv_sum = 1.0f / sum_exp;
    for (uint i = 0; i < vec_count; i++) {
        out4[i] = exp(in4[i] - float4(max_val)) * inv_sum;
    }
}

kernel void softmax_online_vec4_fp16(
    device const half* input      [[buffer(0)]],
    device half* output           [[buffer(1)]],
    constant uint& outer_size     [[buffer(2)]],
    constant uint& reduce_size    [[buffer(3)]],  // must be multiple of 4
    constant uint& inner_size     [[buffer(4)]],  // must be 1
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= outer_size) return;

    const uint base = gid * reduce_size;
    const uint vec_count = reduce_size / 4;
    device const half4* in4 = reinterpret_cast<device const half4*>(input + base);
    device half4* out4 = reinterpret_cast<device half4*>(output + base);

    // Pass 1: find max (accumulate in float for stability)
    float4 max4 = float4(-INFINITY);
    for (uint i = 0; i < vec_count; i++) {
        float4 v = float4(in4[i]);
        max4 = max(max4, v);
    }
    float max_val = max(max(max4.x, max4.y), max(max4.z, max4.w));

    // Pass 2: exp and sum (accumulate in float)
    float sum_exp = 0.0f;
    for (uint i = 0; i < vec_count; i++) {
        float4 v = float4(in4[i]);
        float4 e = exp(v - float4(max_val));
        sum_exp += e.x + e.y + e.z + e.w;
    }

    // Pass 3: normalize
    float inv_sum = 1.0f / sum_exp;
    for (uint i = 0; i < vec_count; i++) {
        float4 v = float4(in4[i]);
        float4 result = exp(v - float4(max_val)) * inv_sum;
        out4[i] = half4(result);
    }
}

kernel void softmax_simd_cooperative_vec4_fp32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],  // must be multiple of 4
    constant uint& inner_size [[buffer(4)]],  // must be 1
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (gid >= outer_size) return;

    const uint base = gid * reduce_size;
    const uint vec_count = reduce_size / 4;
    device const float4* in4 = reinterpret_cast<device const float4*>(input + base);
    device float4* out4 = reinterpret_cast<device float4*>(output + base);

    // Pass 1: Find max using SIMD cooperative reduction
    float thread_max = -INFINITY;
    for (uint i = lane; i < vec_count; i += 32) {
        float4 v = in4[i];
        float local_max = max(max(v.x, v.y), max(v.z, v.w));
        thread_max = max(thread_max, local_max);
    }
    float row_max = simd_max(thread_max);

    // Pass 2: Compute sum of exp(x - max)
    float thread_sum = 0.0f;
    for (uint i = lane; i < vec_count; i += 32) {
        float4 v = in4[i];
        float4 e = exp(v - float4(row_max));
        thread_sum += e.x + e.y + e.z + e.w;
    }
    float row_sum = simd_sum(thread_sum);

    // Pass 3: Normalize and write output
    float inv_sum = 1.0f / row_sum;
    for (uint i = lane; i < vec_count; i += 32) {
        float4 v = in4[i];
        out4[i] = exp(v - float4(row_max)) * inv_sum;
    }
}

kernel void softmax_simd_cooperative_vec4_fp16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],  // must be multiple of 4
    constant uint& inner_size [[buffer(4)]],  // must be 1
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (gid >= outer_size) return;

    const uint base = gid * reduce_size;
    const uint vec_count = reduce_size / 4;
    device const half4* in4 = reinterpret_cast<device const half4*>(input + base);
    device half4* out4 = reinterpret_cast<device half4*>(output + base);

    // Pass 1: Find max using SIMD cooperative reduction (FP32 accumulation)
    float thread_max = -INFINITY;
    for (uint i = lane; i < vec_count; i += 32) {
        float4 v = float4(in4[i]);
        float local_max = max(max(v.x, v.y), max(v.z, v.w));
        thread_max = max(thread_max, local_max);
    }
    float row_max = simd_max(thread_max);

    // Pass 2: Compute sum of exp(x - max) (FP32 accumulation)
    float thread_sum = 0.0f;
    for (uint i = lane; i < vec_count; i += 32) {
        float4 v = float4(in4[i]);
        float4 e = exp(v - float4(row_max));
        thread_sum += e.x + e.y + e.z + e.w;
    }
    float row_sum = simd_sum(thread_sum);

    // Pass 3: Normalize and write output
    float inv_sum = 1.0f / row_sum;
    for (uint i = lane; i < vec_count; i += 32) {
        float4 v = float4(in4[i]);
        float4 result = exp(v - float4(row_max)) * inv_sum;
        out4[i] = half4(result);
    }
}
