#include <metal_stdlib>
using namespace metal;

// RMSNorm FP32: output = (x / sqrt(mean(x^2) + eps)) * weight
kernel void fused_rms_norm_fp32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& norm_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tid.y;
    if (row >= batch_size) return;

    const uint base_offset = row * norm_size;

    // Pass 1: Compute sum of squares with vectorized loads
    float sum_sq = 0.0f;

    // Vectorized processing for 4-aligned chunks
    const uint vec_end = (norm_size / 4) * 4;
    for (uint i = lane * 4; i < vec_end; i += 128) {
        float4 val = *reinterpret_cast<device const float4*>(&input[base_offset + i]);
        sum_sq += dot(val, val);
    }

    // Scalar tail
    for (uint i = vec_end + lane; i < norm_size; i += 32) {
        float val = input[base_offset + i];
        sum_sq += val * val;
    }

    // SIMD reduction
    sum_sq = simd_sum(sum_sq);

    // Compute inverse RMS
    const float inv_rms = 1.0f / sqrt(sum_sq / float(norm_size) + eps);

    // Pass 2: Normalize and apply weight with vectorized stores
    for (uint i = lane * 4; i < vec_end; i += 128) {
        float4 val = *reinterpret_cast<device const float4*>(&input[base_offset + i]);
        float4 w = *reinterpret_cast<device const float4*>(&weight[i]);
        float4 result = val * inv_rms * w;
        *reinterpret_cast<device float4*>(&output[base_offset + i]) = result;
    }

    // Scalar tail
    for (uint i = vec_end + lane; i < norm_size; i += 32) {
        output[base_offset + i] = input[base_offset + i] * inv_rms * weight[i];
    }
}

// RMSNorm FP16: output = (x / sqrt(mean(x^2) + eps)) * weight
// Uses FP32 accumulation for numerical stability
kernel void fused_rms_norm_fp16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& norm_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tid.y;
    if (row >= batch_size) return;

    const uint base_offset = row * norm_size;

    // Pass 1: Compute sum of squares with FP32 accumulation
    float sum_sq = 0.0f;

    // Vectorized processing for 4-aligned chunks
    const uint vec_end = (norm_size / 4) * 4;
    for (uint i = lane * 4; i < vec_end; i += 128) {
        half4 val_h = *reinterpret_cast<device const half4*>(&input[base_offset + i]);
        float4 val = float4(val_h);
        sum_sq += dot(val, val);
    }

    // Scalar tail
    for (uint i = vec_end + lane; i < norm_size; i += 32) {
        float val = float(input[base_offset + i]);
        sum_sq += val * val;
    }

    // SIMD reduction
    sum_sq = simd_sum(sum_sq);

    // Compute inverse RMS
    const float inv_rms = 1.0f / sqrt(sum_sq / float(norm_size) + eps);

    // Pass 2: Normalize and apply weight with vectorized stores
    for (uint i = lane * 4; i < vec_end; i += 128) {
        half4 val_h = *reinterpret_cast<device const half4*>(&input[base_offset + i]);
        half4 w_h = *reinterpret_cast<device const half4*>(&weight[i]);
        float4 val = float4(val_h);
        float4 w = float4(w_h);
        half4 result = half4(val * inv_rms * w);
        *reinterpret_cast<device half4*>(&output[base_offset + i]) = result;
    }

    // Scalar tail
    for (uint i = vec_end + lane; i < norm_size; i += 32) {
        float val = float(input[base_offset + i]);
        float w = float(weight[i]);
        output[base_offset + i] = half(val * inv_rms * w);
    }
}

// LayerNorm FP32: output = ((x - mean) / sqrt(var + eps)) * weight + bias
kernel void fused_layer_norm_fp32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& norm_size [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    threadgroup float* shared_input [[threadgroup(0)]],
    uint2 tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tid.y;
    if (row >= batch_size) return;

    const uint base_offset = row * norm_size;

    // Pass 1: Load to threadgroup memory and compute mean and variance
    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Vectorized processing for 4-aligned chunks
    const uint vec_end = (norm_size / 4) * 4;
    for (uint i = lane * 4; i < vec_end; i += 128) {
        float4 val = *reinterpret_cast<device const float4*>(&input[base_offset + i]);

        // Store to threadgroup memory
        shared_input[i] = val.x;
        shared_input[i + 1] = val.y;
        shared_input[i + 2] = val.z;
        shared_input[i + 3] = val.w;

        sum += val.x + val.y + val.z + val.w;
        sum_sq += dot(val, val);
    }

    // Scalar tail
    for (uint i = vec_end + lane; i < norm_size; i += 32) {
        float val = input[base_offset + i];
        shared_input[i] = val;
        sum += val;
        sum_sq += val * val;
    }

    // SIMD reduction
    sum = simd_sum(sum);
    sum_sq = simd_sum(sum_sq);

    // Compute mean and inverse std
    const float mean = sum / float(norm_size);
    float variance = sum_sq / float(norm_size) - mean * mean;
    variance = max(variance, 0.0f);  // Numerical safety
    const float inv_std = 1.0f / sqrt(variance + eps);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: Normalize from cached data and apply weight + bias
    for (uint i = lane * 4; i < vec_end; i += 128) {
        float4 val = float4(shared_input[i], shared_input[i + 1], shared_input[i + 2], shared_input[i + 3]);
        float4 w = *reinterpret_cast<device const float4*>(&weight[i]);
        float4 b = *reinterpret_cast<device const float4*>(&bias[i]);
        float4 result = (val - mean) * inv_std * w + b;
        *reinterpret_cast<device float4*>(&output[base_offset + i]) = result;
    }

    // Scalar tail
    for (uint i = vec_end + lane; i < norm_size; i += 32) {
        output[base_offset + i] = (shared_input[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

// LayerNorm FP16: output = ((x - mean) / sqrt(var + eps)) * weight + bias
// Uses FP32 accumulation for numerical stability
kernel void fused_layer_norm_fp16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& norm_size [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    threadgroup float* shared_input [[threadgroup(0)]],
    uint2 tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tid.y;
    if (row >= batch_size) return;

    const uint base_offset = row * norm_size;

    // Pass 1: Load to threadgroup memory and compute mean and variance with FP32 accumulation
    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Vectorized processing for 4-aligned chunks
    const uint vec_end = (norm_size / 4) * 4;
    for (uint i = lane * 4; i < vec_end; i += 128) {
        half4 val_h = *reinterpret_cast<device const half4*>(&input[base_offset + i]);
        float4 val = float4(val_h);

        // Store to threadgroup memory as FP32
        shared_input[i] = val.x;
        shared_input[i + 1] = val.y;
        shared_input[i + 2] = val.z;
        shared_input[i + 3] = val.w;

        sum += val.x + val.y + val.z + val.w;
        sum_sq += dot(val, val);
    }

    // Scalar tail
    for (uint i = vec_end + lane; i < norm_size; i += 32) {
        float val = float(input[base_offset + i]);
        shared_input[i] = val;
        sum += val;
        sum_sq += val * val;
    }

    // SIMD reduction
    sum = simd_sum(sum);
    sum_sq = simd_sum(sum_sq);

    // Compute mean and inverse std
    const float mean = sum / float(norm_size);
    float variance = sum_sq / float(norm_size) - mean * mean;
    variance = max(variance, 0.0f);  // Numerical safety
    const float inv_std = 1.0f / sqrt(variance + eps);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: Normalize from cached data and apply weight + bias
    for (uint i = lane * 4; i < vec_end; i += 128) {
        float4 val = float4(shared_input[i], shared_input[i + 1], shared_input[i + 2], shared_input[i + 3]);
        half4 w_h = *reinterpret_cast<device const half4*>(&weight[i]);
        half4 b_h = *reinterpret_cast<device const half4*>(&bias[i]);
        float4 w = float4(w_h);
        float4 b = float4(b_h);
        half4 result = half4((val - mean) * inv_std * w + b);
        *reinterpret_cast<device half4*>(&output[base_offset + i]) = result;
    }

    // Scalar tail
    for (uint i = vec_end + lane; i < norm_size; i += 32) {
        float val = shared_input[i];
        float w = float(weight[i]);
        float b = float(bias[i]);
        output[base_offset + i] = half((val - mean) * inv_std * w + b);
    }
}
