/// @file dequantize_kernel.metal
/// @brief Dequantization kernels for INT8/INT4 to FP16.

#include <metal_stdlib>
using namespace metal;

/// INT8 -> FP16 dequantization kernel.
/// Each thread dequantizes one element: output[i] = half(input[i]) * scale[group_idx]
kernel void dequantize_int8_to_fp16(
    device const char* input       [[buffer(0)]],
    device const half* scales      [[buffer(1)]],
    device half* output            [[buffer(2)]],
    constant uint& num_elements    [[buffer(3)]],
    constant uint& group_size      [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]])
{
    if (gid >= num_elements) return;

    uint group_idx = gid / group_size;
    half scale = scales[group_idx];
    output[gid] = half(input[gid]) * scale;
}

/// INT4 -> FP16 dequantization kernel (2 elements per byte).
/// Each thread processes one byte, producing 2 output elements.
kernel void dequantize_int4_to_fp16(
    device const uchar* input      [[buffer(0)]],
    device const half* scales      [[buffer(1)]],
    device half* output            [[buffer(2)]],
    constant uint& num_elements    [[buffer(3)]],
    constant uint& group_size      [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]])
{
    // Each thread handles 2 output elements (one byte)
    uint byte_idx = gid;
    uint out_idx = gid * 2;

    if (out_idx >= num_elements) return;

    uchar packed = input[byte_idx];
    // Low nibble = first element, high nibble = second element
    int lo = int(packed & 0x0F) - 8;  // signed: subtract 8 for zero-point
    int hi = int((packed >> 4) & 0x0F) - 8;

    uint group_idx_lo = out_idx / group_size;
    half scale_lo = scales[group_idx_lo];
    output[out_idx] = half(lo) * scale_lo;

    if (out_idx + 1 < num_elements) {
        uint group_idx_hi = (out_idx + 1) / group_size;
        half scale_hi = scales[group_idx_hi];
        output[out_idx + 1] = half(hi) * scale_hi;
    }
}
