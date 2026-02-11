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

// ============================================================================
// LARGE FUSED RESIDUAL NORM KERNELS (norm_size >= 2048)
// ============================================================================
// 256 threads per threadgroup (8 SIMD groups), one threadgroup per row.
// ============================================================================

constant constexpr uint RESID_LARGE_THREADS = 256;
constant constexpr uint RESID_LARGE_SIMD_GROUPS = RESID_LARGE_THREADS / 32;  // 8

kernel void fused_residual_rms_norm_large_fp32(
    device const float* input      [[buffer(0)]],
    device const float* residual   [[buffer(1)]],
    device const float* weight     [[buffer(2)]],
    device float*       output     [[buffer(3)]],
    constant uint&      batch_size [[buffer(4)]],
    constant uint&      norm_size  [[buffer(5)]],
    constant float&     eps        [[buffer(6)]],
    threadgroup float*  shared     [[threadgroup(0)]],  // 8 floats
    uint  tg_idx        [[threadgroup_position_in_grid]],
    uint  tid_in_tg     [[thread_index_in_threadgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]])
{
    uint row = tg_idx;
    if (row >= batch_size) return;

    uint base = row * norm_size;

    // Pass 1: Compute sum of squares of (input + residual), store sum to output
    float sum_sq = 0.0f;
    for (uint i = tid_in_tg; i < norm_size; i += RESID_LARGE_THREADS) {
        float val = input[base + i] + residual[base + i];
        output[base + i] = val;
        sum_sq += val * val;
    }

    // Intra-SIMD reduction
    sum_sq = simd_sum(sum_sq);

    if (simd_lane_id == 0) {
        shared[simd_group_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inter-SIMD reduction
    float total_sq = 0.0f;
    if (simd_group_id == 0 && simd_lane_id < RESID_LARGE_SIMD_GROUPS) {
        total_sq = shared[simd_lane_id];
    }
    if (simd_group_id == 0) {
        total_sq = simd_sum(total_sq);
        if (simd_lane_id == 0) {
            shared[0] = total_sq;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = rsqrt(shared[0] / float(norm_size) + eps);

    // Pass 2: Normalize and scale
    for (uint i = tid_in_tg; i < norm_size; i += RESID_LARGE_THREADS) {
        output[base + i] = output[base + i] * rms_inv * weight[i];
    }
}

kernel void fused_residual_rms_norm_large_fp16(
    device const half*  input      [[buffer(0)]],
    device const half*  residual   [[buffer(1)]],
    device const half*  weight     [[buffer(2)]],
    device half*        output     [[buffer(3)]],
    constant uint&      batch_size [[buffer(4)]],
    constant uint&      norm_size  [[buffer(5)]],
    constant float&     eps        [[buffer(6)]],
    threadgroup float*  shared     [[threadgroup(0)]],
    uint  tg_idx        [[threadgroup_position_in_grid]],
    uint  tid_in_tg     [[thread_index_in_threadgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]])
{
    uint row = tg_idx;
    if (row >= batch_size) return;

    uint base = row * norm_size;

    // Pass 1: FP32 accumulation
    float sum_sq = 0.0f;
    for (uint i = tid_in_tg; i < norm_size; i += RESID_LARGE_THREADS) {
        float val = float(input[base + i]) + float(residual[base + i]);
        output[base + i] = half(val);
        sum_sq += val * val;
    }

    sum_sq = simd_sum(sum_sq);

    if (simd_lane_id == 0) {
        shared[simd_group_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = 0.0f;
    if (simd_group_id == 0 && simd_lane_id < RESID_LARGE_SIMD_GROUPS) {
        total_sq = shared[simd_lane_id];
    }
    if (simd_group_id == 0) {
        total_sq = simd_sum(total_sq);
        if (simd_lane_id == 0) {
            shared[0] = total_sq;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = rsqrt(shared[0] / float(norm_size) + eps);

    // Pass 2: FP32 math -> FP16 store
    for (uint i = tid_in_tg; i < norm_size; i += RESID_LARGE_THREADS) {
        float val = float(output[base + i]);
        output[base + i] = half(val * rms_inv * float(weight[i]));
    }
}
