#include <metal_stdlib>
using namespace metal;

// Fused Residual + RMSNorm: output = rms_norm(input + residual) * weight
// Single kernel reads input+residual, computes sum, then normalizes in-place
// Uses 32-thread SIMD cooperative reduction (same pattern as fused_norm.metal)

kernel void fused_residual_rms_norm_fp32(
    device const float* input      [[buffer(0)]],
    device const float* residual   [[buffer(1)]],
    device const float* weight     [[buffer(2)]],
    device float*       output     [[buffer(3)]],
    constant uint&      batch_size [[buffer(4)]],
    constant uint&      norm_size  [[buffer(5)]],
    constant float&     eps        [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint row  = tid.y;     // batch index
    if (row >= batch_size) return;

    uint base = row * norm_size;

    // Pass 1: Compute sum of squares of (input + residual)
    // Also write the sum to output for Pass 2
    float sum_sq = 0.0f;
    for (uint i = lane; i < norm_size; i += 32) {
        float val = input[base + i] + residual[base + i];
        output[base + i] = val;  // Store sum for pass 2
        sum_sq += val * val;
    }
    sum_sq = simd_sum(sum_sq);

    float rms_inv = rsqrt(sum_sq / float(norm_size) + eps);

    // Pass 2: Normalize and scale
    for (uint i = lane; i < norm_size; i += 32) {
        output[base + i] = output[base + i] * rms_inv * weight[i];
    }
}

// FP16 variant - compute in FP32 for numerical stability
kernel void fused_residual_rms_norm_fp16(
    device const half*  input      [[buffer(0)]],
    device const half*  residual   [[buffer(1)]],
    device const half*  weight     [[buffer(2)]],
    device half*        output     [[buffer(3)]],
    constant uint&      batch_size [[buffer(4)]],
    constant uint&      norm_size  [[buffer(5)]],
    constant float&     eps        [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint row  = tid.y;
    if (row >= batch_size) return;

    uint base = row * norm_size;

    float sum_sq = 0.0f;
    for (uint i = lane; i < norm_size; i += 32) {
        float val = float(input[base + i]) + float(residual[base + i]);
        output[base + i] = half(val);
        sum_sq += val * val;
    }
    sum_sq = simd_sum(sum_sq);

    float rms_inv = rsqrt(sum_sq / float(norm_size) + eps);

    for (uint i = lane; i < norm_size; i += 32) {
        float val = float(output[base + i]);
        output[base + i] = half(val * rms_inv * float(weight[i]));
    }
}
