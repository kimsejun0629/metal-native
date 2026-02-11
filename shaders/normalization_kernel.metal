#include "common/metal_types.h"
#include "common/simd_utils.h"
#include <metal_stdlib>
using namespace metal;

// Welford's online algorithm for numerically stable mean and variance
// mean_new = mean_old + (x - mean_old) / count
// M2_new = M2_old + (x - mean_old) * (x - mean_new)
// variance = M2 / count

struct WelfordState {
    float mean;
    float M2;
    uint count;
};

inline WelfordState welford_update(WelfordState state, float value) {
    WelfordState result;
    result.count = state.count + 1;
    float delta = value - state.mean;
    result.mean = state.mean + delta / float(result.count);
    float delta2 = value - result.mean;
    result.M2 = state.M2 + delta * delta2;
    return result;
}

// LayerNorm: y = (x - mean) / sqrt(variance + eps) * weight + bias
// Normalizes over the last dimension(s)

kernel void layer_norm_kernel(
    device const float* input       [[buffer(0)]],
    device const float* weight      [[buffer(1)]],
    device const float* bias        [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& batch_size       [[buffer(4)]],  // Product of all dims except normalized
    constant uint& norm_size        [[buffer(5)]],  // Product of normalized dimensions
    constant float& eps             [[buffer(6)]],
    uint2 gid                       [[thread_position_in_grid]],
    uint  simd_lane_id              [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = gid.y;

    if (batch_idx >= batch_size) return;

    const uint base_offset = batch_idx * norm_size;
    const uint lane = gid.x;  // 0..31

    // Cooperative statistics computation: each of 32 threads handles strided elements
    float partial_sum = 0.0f;
    float partial_sq_sum = 0.0f;

    for (uint i = lane; i < norm_size; i += 32) {
        float val = input[base_offset + i];
        partial_sum += val;
        partial_sq_sum += val * val;
    }

    // SIMD reduction to get totals across all 32 threads
    float total_sum = simd_sum(partial_sum);
    float total_sq_sum = simd_sum(partial_sq_sum);

    float mean = total_sum / float(norm_size);
    float variance = total_sq_sum / float(norm_size) - mean * mean;
    float inv_std = 1.0f / sqrt(variance + eps);

    // Normalize and apply affine transform (all 32 threads participate)
    for (uint i = lane; i < norm_size; i += 32) {
        float normalized = (input[base_offset + i] - mean) * inv_std;
        output[base_offset + i] = normalized * weight[i] + bias[i];
    }
}

kernel void layer_norm_kernel_fp16(
    device const half* input        [[buffer(0)]],
    device const half* weight       [[buffer(1)]],
    device const half* bias         [[buffer(2)]],
    device half* output             [[buffer(3)]],
    constant uint& batch_size       [[buffer(4)]],
    constant uint& norm_size        [[buffer(5)]],
    constant float& eps             [[buffer(6)]],
    uint2 gid                       [[thread_position_in_grid]],
    uint  simd_lane_id              [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = gid.y;

    if (batch_idx >= batch_size) return;

    const uint base_offset = batch_idx * norm_size;
    const uint lane = gid.x;  // 0..31

    // Cooperative statistics computation with FP32 accumulation for numerical stability
    float partial_sum = 0.0f;
    float partial_sq_sum = 0.0f;

    for (uint i = lane; i < norm_size; i += 32) {
        float val = float(input[base_offset + i]);
        partial_sum += val;
        partial_sq_sum += val * val;
    }

    // SIMD reduction to get totals across all 32 threads
    float total_sum = simd_sum(partial_sum);
    float total_sq_sum = simd_sum(partial_sq_sum);

    float mean = total_sum / float(norm_size);
    float variance = total_sq_sum / float(norm_size) - mean * mean;
    float inv_std = 1.0f / sqrt(variance + eps);

    // Normalize and apply affine transform (all 32 threads participate)
    for (uint i = lane; i < norm_size; i += 32) {
        float normalized = (float(input[base_offset + i]) - mean) * inv_std;
        output[base_offset + i] = half(normalized * float(weight[i]) + float(bias[i]));
    }
}

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
// No mean centering, no bias

kernel void rms_norm_kernel(
    device const float* input       [[buffer(0)]],
    device const float* weight      [[buffer(1)]],
    device float* output            [[buffer(2)]],
    constant uint& batch_size       [[buffer(3)]],
    constant uint& norm_size        [[buffer(4)]],
    constant float& eps             [[buffer(5)]],
    uint2 gid                       [[thread_position_in_grid]],
    uint  simd_lane_id              [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = gid.y;

    if (batch_idx >= batch_size) return;

    const uint base_offset = batch_idx * norm_size;
    const uint lane = gid.x;  // 0..31

    // Cooperative RMS computation: each of 32 threads handles strided elements
    float partial_sq_sum = 0.0f;

    for (uint i = lane; i < norm_size; i += 32) {
        float val = input[base_offset + i];
        partial_sq_sum += val * val;
    }

    // SIMD reduction to get total across all 32 threads
    float total_sq_sum = simd_sum(partial_sq_sum);

    float rms = sqrt(total_sq_sum / float(norm_size) + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and apply weight (all 32 threads participate)
    for (uint i = lane; i < norm_size; i += 32) {
        output[base_offset + i] = input[base_offset + i] * inv_rms * weight[i];
    }
}

kernel void rms_norm_kernel_fp16(
    device const half* input        [[buffer(0)]],
    device const half* weight       [[buffer(1)]],
    device half* output             [[buffer(2)]],
    constant uint& batch_size       [[buffer(3)]],
    constant uint& norm_size        [[buffer(4)]],
    constant float& eps             [[buffer(5)]],
    uint2 gid                       [[thread_position_in_grid]],
    uint  simd_lane_id              [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = gid.y;

    if (batch_idx >= batch_size) return;

    const uint base_offset = batch_idx * norm_size;
    const uint lane = gid.x;  // 0..31

    // Cooperative RMS computation with FP32 accumulation
    float partial_sq_sum = 0.0f;

    for (uint i = lane; i < norm_size; i += 32) {
        float val = float(input[base_offset + i]);
        partial_sq_sum += val * val;
    }

    // SIMD reduction to get total across all 32 threads
    float total_sq_sum = simd_sum(partial_sq_sum);

    float rms = sqrt(total_sq_sum / float(norm_size) + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and apply weight (all 32 threads participate)
    for (uint i = lane; i < norm_size; i += 32) {
        output[base_offset + i] = half(float(input[base_offset + i]) * inv_rms * float(weight[i]));
    }
}

// BatchNorm: normalize over batch dimension for each channel
// Training: compute batch statistics and update running statistics
// Inference: use running statistics

kernel void batch_norm_training_kernel(
    device const float* input           [[buffer(0)]],  // [batch, channels, spatial...]
    device float* output                [[buffer(1)]],
    device float* running_mean          [[buffer(2)]],  // [channels]
    device float* running_var           [[buffer(3)]],  // [channels]
    device const float* weight          [[buffer(4)]],  // [channels]
    device const float* bias            [[buffer(5)]],  // [channels]
    constant uint& batch_size           [[buffer(6)]],
    constant uint& channels             [[buffer(7)]],
    constant uint& spatial_size         [[buffer(8)]],  // Product of spatial dimensions
    constant float& momentum            [[buffer(9)]],
    constant float& eps                 [[buffer(10)]],
    uint2 gid                           [[thread_position_in_grid]],
    uint  simd_lane_id                  [[thread_index_in_simdgroup]]
) {
    const uint lane = gid.x;  // 0..31
    const uint c = gid.y;     // channel index

    if (c >= channels) return;

    const uint total_elements = batch_size * spatial_size;

    // Cooperative statistics computation: 32 threads iterate with stride
    float partial_sum = 0.0f;
    float partial_sq_sum = 0.0f;

    for (uint i = lane; i < total_elements; i += 32) {
        const uint b = i / spatial_size;
        const uint s = i % spatial_size;
        const uint idx = b * channels * spatial_size + c * spatial_size + s;
        float val = input[idx];
        partial_sum += val;
        partial_sq_sum += val * val;
    }

    // SIMD reduction to get totals across all 32 threads
    float total_sum = simd_sum(partial_sum);
    float total_sq_sum = simd_sum(partial_sq_sum);

    float batch_mean = total_sum / float(total_elements);
    float batch_var = total_sq_sum / float(total_elements) - batch_mean * batch_mean;

    // Only lane 0 updates running statistics to avoid race conditions
    if (lane == 0) {
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * batch_mean;
        running_var[c] = (1.0f - momentum) * running_var[c] + momentum * batch_var;
    }

    // Normalize and apply affine transform (all 32 threads participate)
    float inv_std = 1.0f / sqrt(batch_var + eps);

    for (uint i = lane; i < total_elements; i += 32) {
        const uint b = i / spatial_size;
        const uint s = i % spatial_size;
        const uint idx = b * channels * spatial_size + c * spatial_size + s;
        float normalized = (input[idx] - batch_mean) * inv_std;
        output[idx] = normalized * weight[c] + bias[c];
    }
}

kernel void batch_norm_inference_kernel(
    device const float* input           [[buffer(0)]],
    device float* output                [[buffer(1)]],
    device const float* running_mean    [[buffer(2)]],
    device const float* running_var     [[buffer(3)]],
    device const float* weight          [[buffer(4)]],
    device const float* bias            [[buffer(5)]],
    constant uint& batch_size           [[buffer(6)]],
    constant uint& channels             [[buffer(7)]],
    constant uint& spatial_size         [[buffer(8)]],
    constant float& eps                 [[buffer(9)]],
    uint3 gid                           [[thread_position_in_grid]]
) {
    const uint b = gid.z;
    const uint c = gid.y;
    const uint s = gid.x;

    if (b >= batch_size || c >= channels || s >= spatial_size) return;

    const uint idx = b * channels * spatial_size + c * spatial_size + s;

    float mean = running_mean[c];
    float var = running_var[c];
    float inv_std = 1.0f / sqrt(var + eps);

    float normalized = (input[idx] - mean) * inv_std;
    output[idx] = normalized * weight[c] + bias[c];
}

// GroupNorm: divide channels into groups and normalize each group
// Similar to LayerNorm but operates on channel groups

kernel void group_norm_kernel(
    device const float* input       [[buffer(0)]],  // [batch, channels, spatial...]
    device const float* weight      [[buffer(1)]],  // [channels]
    device const float* bias        [[buffer(2)]],  // [channels]
    device float* output            [[buffer(3)]],
    constant uint& batch_size       [[buffer(4)]],
    constant uint& channels         [[buffer(5)]],
    constant uint& spatial_size     [[buffer(6)]],
    constant uint& num_groups       [[buffer(7)]],
    constant float& eps             [[buffer(8)]],
    uint3 gid                       [[thread_position_in_grid]]
) {
    const uint lane = gid.x;       // 0..31
    const uint group_idx = gid.y;  // group index
    const uint batch_idx = gid.z;  // batch index

    if (batch_idx >= batch_size || group_idx >= num_groups) return;

    const uint channels_per_group = channels / num_groups;
    const uint group_size = channels_per_group * spatial_size;

    const uint group_start = group_idx * channels_per_group;
    const uint batch_offset = batch_idx * channels * spatial_size;

    // Cooperative statistics computation: 32 threads iterate with stride
    float partial_sum = 0.0f;
    float partial_sq_sum = 0.0f;

    for (uint i = lane; i < group_size; i += 32) {
        const uint c = i / spatial_size;
        const uint s = i % spatial_size;
        const uint idx = batch_offset + (group_start + c) * spatial_size + s;
        float val = input[idx];
        partial_sum += val;
        partial_sq_sum += val * val;
    }

    // SIMD reduction to get totals across all 32 threads
    float total_sum = simd_sum(partial_sum);
    float total_sq_sum = simd_sum(partial_sq_sum);

    float mean = total_sum / float(group_size);
    float variance = total_sq_sum / float(group_size) - mean * mean;
    float inv_std = 1.0f / sqrt(variance + eps);

    // Normalize and apply affine transform (all 32 threads participate)
    for (uint i = lane; i < group_size; i += 32) {
        const uint c = i / spatial_size;
        const uint s = i % spatial_size;
        const uint channel_idx = group_start + c;
        const uint idx = batch_offset + channel_idx * spatial_size + s;
        float normalized = (input[idx] - mean) * inv_std;
        output[idx] = normalized * weight[channel_idx] + bias[channel_idx];
    }
}

kernel void group_norm_kernel_fp16(
    device const half* input        [[buffer(0)]],
    device const half* weight       [[buffer(1)]],
    device const half* bias         [[buffer(2)]],
    device half* output             [[buffer(3)]],
    constant uint& batch_size       [[buffer(4)]],
    constant uint& channels         [[buffer(5)]],
    constant uint& spatial_size     [[buffer(6)]],
    constant uint& num_groups       [[buffer(7)]],
    constant float& eps             [[buffer(8)]],
    uint3 gid                       [[thread_position_in_grid]]
) {
    const uint lane = gid.x;       // 0..31
    const uint group_idx = gid.y;  // group index
    const uint batch_idx = gid.z;  // batch index

    if (batch_idx >= batch_size || group_idx >= num_groups) return;

    const uint channels_per_group = channels / num_groups;
    const uint group_size = channels_per_group * spatial_size;

    const uint group_start = group_idx * channels_per_group;
    const uint batch_offset = batch_idx * channels * spatial_size;

    // Cooperative statistics computation with FP32 accumulation
    float partial_sum = 0.0f;
    float partial_sq_sum = 0.0f;

    for (uint i = lane; i < group_size; i += 32) {
        const uint c = i / spatial_size;
        const uint s = i % spatial_size;
        const uint idx = batch_offset + (group_start + c) * spatial_size + s;
        float val = float(input[idx]);
        partial_sum += val;
        partial_sq_sum += val * val;
    }

    // SIMD reduction to get totals across all 32 threads
    float total_sum = simd_sum(partial_sum);
    float total_sq_sum = simd_sum(partial_sq_sum);

    float mean = total_sum / float(group_size);
    float variance = total_sq_sum / float(group_size) - mean * mean;
    float inv_std = 1.0f / sqrt(variance + eps);

    // Normalize and apply affine transform (all 32 threads participate)
    for (uint i = lane; i < group_size; i += 32) {
        const uint c = i / spatial_size;
        const uint s = i % spatial_size;
        const uint channel_idx = group_start + c;
        const uint idx = batch_offset + channel_idx * spatial_size + s;
        float normalized = (float(input[idx]) - mean) * inv_std;
        output[idx] = half(normalized * float(weight[channel_idx]) + float(bias[channel_idx]));
    }
}

// ============================================================================
// VECTORIZED KERNELS (Task 1.2)
// ============================================================================
// These kernels use float4/half4 vectorized loads and threadgroup memory caching
// for improved memory bandwidth utilization on large norm_size (>= 128).
//
// Key optimizations:
// 1. float4/half4 loads: 4x elements per memory transaction
// 2. Threadgroup caching: Second pass reads from fast threadgroup memory instead of device memory
// 3. Efficient tail handling: Scalar loads for remainder elements
//
// Constraints:
// - Threadgroup memory limit: 32KB per threadgroup
// - norm_size * sizeof(float) must be <= 32KB
// - For norm_size > 8192, dispatcher falls back to non-cached kernels
// ============================================================================

kernel void layer_norm_vec_fp32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& norm_size [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    threadgroup float* shared_input [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = gid.y;
    if (batch_idx >= batch_size) return;

    const uint base_offset = batch_idx * norm_size;

    // Number of float4 vectors and remainder elements
    const uint vec_count = norm_size / 4;
    const uint remainder = norm_size % 4;

    // Pass 1: Cooperative statistics with float4 loads + threadgroup caching
    float partial_sum = 0.0f;
    float partial_sq_sum = 0.0f;

    // Vectorized loads (4 elements at once)
    for (uint i = lane; i < vec_count; i += 32) {
        float4 val = *reinterpret_cast<const device float4*>(input + base_offset + i * 4);

        // Cache in threadgroup memory for Pass 2
        *reinterpret_cast<threadgroup float4*>(shared_input + i * 4) = val;

        partial_sum += val.x + val.y + val.z + val.w;
        partial_sq_sum += dot(val, val);
    }

    // Handle remainder elements (scalar)
    uint rem_start = vec_count * 4;
    if (lane < remainder) {
        float val = input[base_offset + rem_start + lane];
        shared_input[rem_start + lane] = val;
        partial_sum += val;
        partial_sq_sum += val * val;
    }

    // SIMD reduction
    float total_sum = simd_sum(partial_sum);
    float total_sq_sum = simd_sum(partial_sq_sum);

    float mean = total_sum / float(norm_size);
    float variance = total_sq_sum / float(norm_size) - mean * mean;
    float inv_std = 1.0f / sqrt(max(variance, 0.0f) + eps);

    // Ensure all threads have cached their data
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: Normalize using cached data + vectorized store
    for (uint i = lane; i < vec_count; i += 32) {
        float4 val = *reinterpret_cast<const threadgroup float4*>(shared_input + i * 4);
        float4 w = *reinterpret_cast<const device float4*>(weight + i * 4);
        float4 b = *reinterpret_cast<const device float4*>(bias + i * 4);

        float4 normalized = (val - mean) * inv_std;
        float4 result = normalized * w + b;

        *reinterpret_cast<device float4*>(output + base_offset + i * 4) = result;
    }

    // Handle remainder
    if (lane < remainder) {
        uint idx = rem_start + lane;
        float val = shared_input[idx];
        float normalized = (val - mean) * inv_std;
        output[base_offset + idx] = normalized * weight[idx] + bias[idx];
    }
}

kernel void layer_norm_vec_fp16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& norm_size [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    threadgroup float* shared_input [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = gid.y;
    if (batch_idx >= batch_size) return;

    const uint base_offset = batch_idx * norm_size;

    // Number of half4 vectors and remainder elements
    const uint vec_count = norm_size / 4;
    const uint remainder = norm_size % 4;

    // Pass 1: Cooperative statistics with half4 loads + threadgroup caching (FP32 accumulation)
    float partial_sum = 0.0f;
    float partial_sq_sum = 0.0f;

    // Vectorized loads (4 elements at once), convert to float for accumulation
    for (uint i = lane; i < vec_count; i += 32) {
        half4 val_h = *reinterpret_cast<const device half4*>(input + base_offset + i * 4);
        float4 val = float4(val_h);

        // Cache in threadgroup memory (as float for numerical stability)
        *reinterpret_cast<threadgroup float4*>(shared_input + i * 4) = val;

        partial_sum += val.x + val.y + val.z + val.w;
        partial_sq_sum += dot(val, val);
    }

    // Handle remainder elements (scalar)
    uint rem_start = vec_count * 4;
    if (lane < remainder) {
        float val = float(input[base_offset + rem_start + lane]);
        shared_input[rem_start + lane] = val;
        partial_sum += val;
        partial_sq_sum += val * val;
    }

    // SIMD reduction
    float total_sum = simd_sum(partial_sum);
    float total_sq_sum = simd_sum(partial_sq_sum);

    float mean = total_sum / float(norm_size);
    float variance = total_sq_sum / float(norm_size) - mean * mean;
    float inv_std = 1.0f / sqrt(max(variance, 0.0f) + eps);

    // Ensure all threads have cached their data
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: Normalize using cached data + vectorized store
    for (uint i = lane; i < vec_count; i += 32) {
        float4 val = *reinterpret_cast<const threadgroup float4*>(shared_input + i * 4);
        float4 w = float4(*reinterpret_cast<const device half4*>(weight + i * 4));
        float4 b = float4(*reinterpret_cast<const device half4*>(bias + i * 4));

        float4 normalized = (val - mean) * inv_std;
        float4 result = normalized * w + b;

        *reinterpret_cast<device half4*>(output + base_offset + i * 4) = half4(result);
    }

    // Handle remainder
    if (lane < remainder) {
        uint idx = rem_start + lane;
        float val = shared_input[idx];
        float normalized = (val - mean) * inv_std;
        output[base_offset + idx] = half(normalized * float(weight[idx]) + float(bias[idx]));
    }
}

kernel void rms_norm_vec_fp32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& norm_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    threadgroup float* shared_input [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = gid.y;
    if (batch_idx >= batch_size) return;

    const uint base_offset = batch_idx * norm_size;

    // Number of float4 vectors and remainder elements
    const uint vec_count = norm_size / 4;
    const uint remainder = norm_size % 4;

    // Pass 1: Cooperative RMS computation with float4 loads + threadgroup caching
    float partial_sq_sum = 0.0f;

    // Vectorized loads (4 elements at once)
    for (uint i = lane; i < vec_count; i += 32) {
        float4 val = *reinterpret_cast<const device float4*>(input + base_offset + i * 4);

        // Cache in threadgroup memory for Pass 2
        *reinterpret_cast<threadgroup float4*>(shared_input + i * 4) = val;

        partial_sq_sum += dot(val, val);
    }

    // Handle remainder elements (scalar)
    uint rem_start = vec_count * 4;
    if (lane < remainder) {
        float val = input[base_offset + rem_start + lane];
        shared_input[rem_start + lane] = val;
        partial_sq_sum += val * val;
    }

    // SIMD reduction
    float total_sq_sum = simd_sum(partial_sq_sum);

    float rms = sqrt(total_sq_sum / float(norm_size) + eps);
    float inv_rms = 1.0f / rms;

    // Ensure all threads have cached their data
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: Normalize using cached data + vectorized store
    for (uint i = lane; i < vec_count; i += 32) {
        float4 val = *reinterpret_cast<const threadgroup float4*>(shared_input + i * 4);
        float4 w = *reinterpret_cast<const device float4*>(weight + i * 4);

        float4 result = val * inv_rms * w;

        *reinterpret_cast<device float4*>(output + base_offset + i * 4) = result;
    }

    // Handle remainder
    if (lane < remainder) {
        uint idx = rem_start + lane;
        float val = shared_input[idx];
        output[base_offset + idx] = val * inv_rms * weight[idx];
    }
}

kernel void rms_norm_vec_fp16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& norm_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    threadgroup float* shared_input [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = gid.y;
    if (batch_idx >= batch_size) return;

    const uint base_offset = batch_idx * norm_size;

    // Number of half4 vectors and remainder elements
    const uint vec_count = norm_size / 4;
    const uint remainder = norm_size % 4;

    // Pass 1: Cooperative RMS computation with half4 loads + threadgroup caching (FP32 accumulation)
    float partial_sq_sum = 0.0f;

    // Vectorized loads (4 elements at once), convert to float for accumulation
    for (uint i = lane; i < vec_count; i += 32) {
        half4 val_h = *reinterpret_cast<const device half4*>(input + base_offset + i * 4);
        float4 val = float4(val_h);

        // Cache in threadgroup memory (as float for numerical stability)
        *reinterpret_cast<threadgroup float4*>(shared_input + i * 4) = val;

        partial_sq_sum += dot(val, val);
    }

    // Handle remainder elements (scalar)
    uint rem_start = vec_count * 4;
    if (lane < remainder) {
        float val = float(input[base_offset + rem_start + lane]);
        shared_input[rem_start + lane] = val;
        partial_sq_sum += val * val;
    }

    // SIMD reduction
    float total_sq_sum = simd_sum(partial_sq_sum);

    float rms = sqrt(total_sq_sum / float(norm_size) + eps);
    float inv_rms = 1.0f / rms;

    // Ensure all threads have cached their data
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: Normalize using cached data + vectorized store
    for (uint i = lane; i < vec_count; i += 32) {
        float4 val = *reinterpret_cast<const threadgroup float4*>(shared_input + i * 4);
        float4 w = float4(*reinterpret_cast<const device half4*>(weight + i * 4));

        float4 result = val * inv_rms * w;

        *reinterpret_cast<device half4*>(output + base_offset + i * 4) = half4(result);
    }

    // Handle remainder
    if (lane < remainder) {
        uint idx = rem_start + lane;
        float val = shared_input[idx];
        output[base_offset + idx] = half(val * inv_rms * float(weight[idx]));
    }
}
