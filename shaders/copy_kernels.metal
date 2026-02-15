/// @file copy_kernels.metal
/// @brief Copy and transpose kernels for Metal GPU compute.

#include <metal_stdlib>
#include "common/metal_types.h"

using namespace metal;

// ---------------------------------------------------------------------------
// Flat copy kernels
// ---------------------------------------------------------------------------

kernel void copy_fp32(device const float* input  [[buffer(0)]],
                      device float*       output [[buffer(1)]],
                      constant uint&      num_elements [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        *reinterpret_cast<device float4*>(output + idx) = *reinterpret_cast<device const float4*>(input + idx);
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = input[i];
        }
    }
}

kernel void copy_fp16(device const half* input  [[buffer(0)]],
                      device half*       output [[buffer(1)]],
                      constant uint&      num_elements [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    const uint idx = id * 4;
    if (idx + 3 < num_elements) {
        *reinterpret_cast<device half4*>(output + idx) = *reinterpret_cast<device const half4*>(input + idx);
    } else {
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            output[i] = input[i];
        }
    }
}

// ---------------------------------------------------------------------------
// 2D transpose kernels using threadgroup memory for coalesced access
// ---------------------------------------------------------------------------

// Tile dimensions for the transpose (32x32 is optimal for Apple GPUs).
constant constexpr uint TILE_DIM = 32;

// Padding to avoid bank conflicts in threadgroup memory.
constant constexpr uint TILE_PAD = 1;

kernel void transpose_2d_fp32(device const float* input   [[buffer(0)]],
                              device float*       output  [[buffer(1)]],
                              device const uint*  dims    [[buffer(2)]],
                              uint2 gid   [[thread_position_in_grid]],
                              uint2 lid   [[thread_position_in_threadgroup]],
                              uint2 tgid  [[threadgroup_position_in_grid]]) {
    // dims[0] = rows, dims[1] = cols of the input matrix
    const uint rows = dims[0];
    const uint cols = dims[1];

    threadgroup float tile[TILE_DIM][TILE_DIM + TILE_PAD];

    // Read from input (row-major) into threadgroup tile
    uint in_x = tgid.x * TILE_DIM + lid.x;
    uint in_y = tgid.y * TILE_DIM + lid.y;
    if (in_x < cols && in_y < rows) {
        tile[lid.y][lid.x] = input[in_y * cols + in_x];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write from threadgroup tile into output (transposed)
    uint out_x = tgid.y * TILE_DIM + lid.x;
    uint out_y = tgid.x * TILE_DIM + lid.y;
    if (out_x < rows && out_y < cols) {
        output[out_y * rows + out_x] = tile[lid.x][lid.y];
    }
}

kernel void transpose_2d_fp16(device const half* input   [[buffer(0)]],
                              device half*       output  [[buffer(1)]],
                              device const uint* dims    [[buffer(2)]],
                              uint2 gid   [[thread_position_in_grid]],
                              uint2 lid   [[thread_position_in_threadgroup]],
                              uint2 tgid  [[threadgroup_position_in_grid]]) {
    const uint rows = dims[0];
    const uint cols = dims[1];

    threadgroup half tile[TILE_DIM][TILE_DIM + TILE_PAD];

    uint in_x = tgid.x * TILE_DIM + lid.x;
    uint in_y = tgid.y * TILE_DIM + lid.y;
    if (in_x < cols && in_y < rows) {
        tile[lid.y][lid.x] = input[in_y * cols + in_x];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint out_x = tgid.y * TILE_DIM + lid.x;
    uint out_y = tgid.x * TILE_DIM + lid.y;
    if (out_x < rows && out_y < cols) {
        output[out_y * rows + out_x] = tile[lid.x][lid.y];
    }
}

// ---------------------------------------------------------------------------
// Dtype cast kernels (vectorized for throughput)
// ---------------------------------------------------------------------------

/// Cast FP32 to FP16 (vectorized: 4 elements per thread)
kernel void cast_fp32_to_fp16(device const float* input  [[buffer(0)]],
                               device half*        output [[buffer(1)]],
                               constant uint&      count  [[buffer(2)]],
                               uint tid [[thread_position_in_grid]]) {
    const uint base_idx = tid * 4;
    if (base_idx + 3 < count) {
        // Vectorized path: process 4 elements
        float4 val = *reinterpret_cast<device const float4*>(input + base_idx);
        half4 converted = half4(val);
        *reinterpret_cast<device half4*>(output + base_idx) = converted;
    } else if (base_idx < count) {
        // Scalar tail: handle remaining elements
        for (uint i = base_idx; i < count; ++i) {
            output[i] = half(input[i]);
        }
    }
}

/// Cast FP16 to FP32 (vectorized: 4 elements per thread)
kernel void cast_fp16_to_fp32(device const half*  input  [[buffer(0)]],
                               device float*       output [[buffer(1)]],
                               constant uint&      count  [[buffer(2)]],
                               uint tid [[thread_position_in_grid]]) {
    const uint base_idx = tid * 4;
    if (base_idx + 3 < count) {
        // Vectorized path: process 4 elements
        half4 val = *reinterpret_cast<device const half4*>(input + base_idx);
        float4 converted = float4(val);
        *reinterpret_cast<device float4*>(output + base_idx) = converted;
    } else if (base_idx < count) {
        // Scalar tail: handle remaining elements
        for (uint i = base_idx; i < count; ++i) {
            output[i] = float(input[i]);
        }
    }
}
