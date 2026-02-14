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

/// Vectorized INT8 -> FP16 dequantization (8 elements per thread).
/// Uses vectorized loads/stores for improved memory bandwidth.
kernel void dequantize_int8_to_fp16_vec(
    device const char* input       [[buffer(0)]],
    device const half* scales      [[buffer(1)]],
    device half* output            [[buffer(2)]],
    constant uint& num_elements    [[buffer(3)]],
    constant uint& group_size      [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]])
{
    uint base = gid * 8;
    if (base >= num_elements) return;

    // Load 8 bytes at once using char4 vectors
    device const char4* in4 = reinterpret_cast<device const char4*>(input + base);
    char4 v0 = in4[0];
    char4 v1 = in4[1];

    // Determine scale for first 4 elements
    uint group_idx = base / group_size;
    half scale = scales[group_idx];

    // Convert first 4 elements to FP16 with scaling
    half4 out0 = half4(half(v0.x), half(v0.y), half(v0.z), half(v0.w)) * scale;

    // Handle second 4 elements (check for group boundary crossing)
    half4 out1;
    if (base + 4 < num_elements) {
        // Check if group changes mid-vector (only relevant if group_size < 8)
        uint group_idx2 = (base + 4) / group_size;
        half scale2 = (group_idx2 != group_idx) ? scales[group_idx2] : scale;
        out1 = half4(half(v1.x), half(v1.y), half(v1.z), half(v1.w)) * scale2;
    }

    // Store 8 half values using half4 vectors
    device half4* out4 = reinterpret_cast<device half4*>(output + base);
    out4[0] = out0;
    if (base + 4 < num_elements) {
        out4[1] = out1;
    }
}

/// Vectorized INT4 -> FP16 dequantization (16 elements per thread from 8 bytes).
/// Uses vectorized loads/stores for improved memory bandwidth.
kernel void dequantize_int4_to_fp16_vec(
    device const uchar* input      [[buffer(0)]],
    device const half* scales      [[buffer(1)]],
    device half* output            [[buffer(2)]],
    constant uint& num_elements    [[buffer(3)]],
    constant uint& group_size      [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]])
{
    // Each thread processes 8 bytes → 16 output elements
    uint byte_base = gid * 8;
    uint out_base = byte_base * 2;  // 2 elements per byte
    if (out_base >= num_elements) return;

    // Load 8 bytes at once using uchar4 vectors
    device const uchar4* in4 = reinterpret_cast<device const uchar4*>(input + byte_base);
    uchar4 bytes_lo = in4[0];  // bytes 0-3
    uchar4 bytes_hi = in4[1];  // bytes 4-7

    uint group_idx = out_base / group_size;
    half scale = scales[group_idx];

    // Unpack and dequantize bytes_lo (8 elements from 4 bytes)
    // Byte 0-1 → out0 (4 elements)
    half4 out0 = half4(
        half(int(bytes_lo.x & 0x0F) - 8) * scale,
        half(int((bytes_lo.x >> 4) & 0x0F) - 8) * scale,
        half(int(bytes_lo.y & 0x0F) - 8) * scale,
        half(int((bytes_lo.y >> 4) & 0x0F) - 8) * scale
    );

    // Byte 2-3 → out1 (4 elements)
    half4 out1 = half4(
        half(int(bytes_lo.z & 0x0F) - 8) * scale,
        half(int((bytes_lo.z >> 4) & 0x0F) - 8) * scale,
        half(int(bytes_lo.w & 0x0F) - 8) * scale,
        half(int((bytes_lo.w >> 4) & 0x0F) - 8) * scale
    );

    // Byte 4-5 → out2 (4 elements) - check group boundary
    uint group_idx2 = (out_base + 8) / group_size;
    half scale2 = (group_idx2 != group_idx) ? scales[group_idx2] : scale;

    half4 out2 = half4(
        half(int(bytes_hi.x & 0x0F) - 8) * scale2,
        half(int((bytes_hi.x >> 4) & 0x0F) - 8) * scale2,
        half(int(bytes_hi.y & 0x0F) - 8) * scale2,
        half(int((bytes_hi.y >> 4) & 0x0F) - 8) * scale2
    );

    // Byte 6-7 → out3 (4 elements)
    half4 out3 = half4(
        half(int(bytes_hi.z & 0x0F) - 8) * scale2,
        half(int((bytes_hi.z >> 4) & 0x0F) - 8) * scale2,
        half(int(bytes_hi.w & 0x0F) - 8) * scale2,
        half(int((bytes_hi.w >> 4) & 0x0F) - 8) * scale2
    );

    // Store 16 half values using half4 vectors
    device half4* out4 = reinterpret_cast<device half4*>(output + out_base);
    out4[0] = out0;
    if (out_base + 4 < num_elements) out4[1] = out1;
    if (out_base + 8 < num_elements) out4[2] = out2;
    if (out_base + 12 < num_elements) out4[3] = out3;
}
