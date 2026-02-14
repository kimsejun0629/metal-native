/// @file reduction_kernel.metal
/// @brief Metal kernels for reduction operations using parallel SIMD reduction.
///
/// Each kernel uses a threadgroup of 256 threads cooperatively reducing each output element:
/// 1. Each thread handles a stride of the reduction dimension
/// 2. Use simd_sum/simd_max/simd_min for intra-SIMD-group reduction (32 threads → 1 value)
/// 3. Store SIMD group results in threadgroup memory (8 values for 256 threads / 32 per SIMD)
/// 4. Final reduction by first SIMD group across SIMD group results

#include <metal_stdlib>
#include "common/metal_types.h"

using namespace metal;

constant uint THREADGROUP_SIZE = 256;
constant uint SIMD_SIZE = 32;
constant uint NUM_SIMD_GROUPS = THREADGROUP_SIZE / SIMD_SIZE;  // 8

// ---------------------------------------------------------------------------
// Reduce sum kernels
// ---------------------------------------------------------------------------

/// Reduce sum for FP32 with FP32 accumulation using parallel SIMD reduction.
///
/// Dispatch: threadgroups per grid = num_output, threads per threadgroup = 256
kernel void reduce_sum_fp32(
    device const float* input      [[buffer(0)]],
    device float*       output     [[buffer(1)]],
    constant uint32_t*  constants  [[buffer(2)]],  // [outer_size, reduce_size, inner_size]
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    const uint32_t outer_size  = constants[0];
    const uint32_t reduce_size = constants[1];
    const uint32_t inner_size  = constants[2];

    const uint32_t num_output = outer_size * inner_size;
    if (gid >= num_output) return;

    const uint32_t outer_idx = gid / inner_size;
    const uint32_t inner_idx = gid % inner_size;

    // Phase 1: Strided partial reduction
    float partial = 0.0f;
    for (uint32_t r = tid; r < reduce_size; r += THREADGROUP_SIZE) {
        const uint32_t input_idx = (outer_idx * reduce_size + r) * inner_size + inner_idx;
        partial += input[input_idx];
    }

    // Phase 2: SIMD-group reduction
    partial = simd_sum(partial);

    // Phase 3: Store to shared memory
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        float val = shared[tid];
        val = simd_sum(val);
        if (tid == 0) {
            output[gid] = val;
        }
    }
}

/// Reduce sum for FP16 with FP32 accumulation using parallel SIMD reduction.
kernel void reduce_sum_fp16(
    device const half* input      [[buffer(0)]],
    device float*      output     [[buffer(1)]],
    constant uint32_t* constants  [[buffer(2)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    const uint32_t outer_size  = constants[0];
    const uint32_t reduce_size = constants[1];
    const uint32_t inner_size  = constants[2];

    const uint32_t num_output = outer_size * inner_size;
    if (gid >= num_output) return;

    const uint32_t outer_idx = gid / inner_size;
    const uint32_t inner_idx = gid % inner_size;

    // Phase 1: Strided partial reduction with FP32 accumulation
    float partial = 0.0f;
    for (uint32_t r = tid; r < reduce_size; r += THREADGROUP_SIZE) {
        const uint32_t input_idx = (outer_idx * reduce_size + r) * inner_size + inner_idx;
        partial += float(input[input_idx]);
    }

    // Phase 2: SIMD-group reduction
    partial = simd_sum(partial);

    // Phase 3: Store to shared memory
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        float val = shared[tid];
        val = simd_sum(val);
        if (tid == 0) {
            output[gid] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Reduce mean kernels
// ---------------------------------------------------------------------------

kernel void reduce_mean_fp32(
    device const float* input      [[buffer(0)]],
    device float*       output     [[buffer(1)]],
    constant uint32_t*  constants  [[buffer(2)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    const uint32_t outer_size  = constants[0];
    const uint32_t reduce_size = constants[1];
    const uint32_t inner_size  = constants[2];

    const uint32_t num_output = outer_size * inner_size;
    if (gid >= num_output) return;

    const uint32_t outer_idx = gid / inner_size;
    const uint32_t inner_idx = gid % inner_size;

    // Phase 1: Strided partial reduction
    float partial = 0.0f;
    for (uint32_t r = tid; r < reduce_size; r += THREADGROUP_SIZE) {
        const uint32_t input_idx = (outer_idx * reduce_size + r) * inner_size + inner_idx;
        partial += input[input_idx];
    }

    // Phase 2: SIMD-group reduction
    partial = simd_sum(partial);

    // Phase 3: Store to shared memory
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        float val = shared[tid];
        val = simd_sum(val);
        if (tid == 0) {
            output[gid] = val / float(reduce_size);
        }
    }
}

kernel void reduce_mean_fp16(
    device const half* input      [[buffer(0)]],
    device float*      output     [[buffer(1)]],
    constant uint32_t* constants  [[buffer(2)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    const uint32_t outer_size  = constants[0];
    const uint32_t reduce_size = constants[1];
    const uint32_t inner_size  = constants[2];

    const uint32_t num_output = outer_size * inner_size;
    if (gid >= num_output) return;

    const uint32_t outer_idx = gid / inner_size;
    const uint32_t inner_idx = gid % inner_size;

    // Phase 1: Strided partial reduction with FP32 accumulation
    float partial = 0.0f;
    for (uint32_t r = tid; r < reduce_size; r += THREADGROUP_SIZE) {
        const uint32_t input_idx = (outer_idx * reduce_size + r) * inner_size + inner_idx;
        partial += float(input[input_idx]);
    }

    // Phase 2: SIMD-group reduction
    partial = simd_sum(partial);

    // Phase 3: Store to shared memory
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        float val = shared[tid];
        val = simd_sum(val);
        if (tid == 0) {
            output[gid] = val / float(reduce_size);
        }
    }
}

// ---------------------------------------------------------------------------
// Reduce max kernels
// ---------------------------------------------------------------------------

kernel void reduce_max_fp32(
    device const float* input      [[buffer(0)]],
    device float*       output     [[buffer(1)]],
    constant uint32_t*  constants  [[buffer(2)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    const uint32_t outer_size  = constants[0];
    const uint32_t reduce_size = constants[1];
    const uint32_t inner_size  = constants[2];

    const uint32_t num_output = outer_size * inner_size;
    if (gid >= num_output) return;

    const uint32_t outer_idx = gid / inner_size;
    const uint32_t inner_idx = gid % inner_size;

    // Phase 1: Strided partial reduction
    float partial = -INFINITY;
    for (uint32_t r = tid; r < reduce_size; r += THREADGROUP_SIZE) {
        const uint32_t input_idx = (outer_idx * reduce_size + r) * inner_size + inner_idx;
        partial = max(partial, input[input_idx]);
    }

    // Phase 2: SIMD-group reduction
    partial = simd_max(partial);

    // Phase 3: Store to shared memory
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        float val = shared[tid];
        val = simd_max(val);
        if (tid == 0) {
            output[gid] = val;
        }
    }
}

kernel void reduce_max_fp16(
    device const half* input      [[buffer(0)]],
    device half*       output     [[buffer(1)]],
    constant uint32_t* constants  [[buffer(2)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]])
{
    const uint32_t outer_size  = constants[0];
    const uint32_t reduce_size = constants[1];
    const uint32_t inner_size  = constants[2];

    const uint32_t num_output = outer_size * inner_size;
    if (gid >= num_output) return;

    const uint32_t outer_idx = gid / inner_size;
    const uint32_t inner_idx = gid % inner_size;

    // Phase 1: Strided partial reduction
    half partial = half(-INFINITY);
    for (uint32_t r = tid; r < reduce_size; r += THREADGROUP_SIZE) {
        const uint32_t input_idx = (outer_idx * reduce_size + r) * inner_size + inner_idx;
        partial = max(partial, input[input_idx]);
    }

    // Phase 2: SIMD-group reduction
    partial = simd_max(partial);

    // Phase 3: Store to shared memory
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        half val = shared[tid];
        val = simd_max(val);
        if (tid == 0) {
            output[gid] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Reduce min kernels
// ---------------------------------------------------------------------------

kernel void reduce_min_fp32(
    device const float* input      [[buffer(0)]],
    device float*       output     [[buffer(1)]],
    constant uint32_t*  constants  [[buffer(2)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    const uint32_t outer_size  = constants[0];
    const uint32_t reduce_size = constants[1];
    const uint32_t inner_size  = constants[2];

    const uint32_t num_output = outer_size * inner_size;
    if (gid >= num_output) return;

    const uint32_t outer_idx = gid / inner_size;
    const uint32_t inner_idx = gid % inner_size;

    // Phase 1: Strided partial reduction
    float partial = INFINITY;
    for (uint32_t r = tid; r < reduce_size; r += THREADGROUP_SIZE) {
        const uint32_t input_idx = (outer_idx * reduce_size + r) * inner_size + inner_idx;
        partial = min(partial, input[input_idx]);
    }

    // Phase 2: SIMD-group reduction
    partial = simd_min(partial);

    // Phase 3: Store to shared memory
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        float val = shared[tid];
        val = simd_min(val);
        if (tid == 0) {
            output[gid] = val;
        }
    }
}

kernel void reduce_min_fp16(
    device const half* input      [[buffer(0)]],
    device half*       output     [[buffer(1)]],
    constant uint32_t* constants  [[buffer(2)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]])
{
    const uint32_t outer_size  = constants[0];
    const uint32_t reduce_size = constants[1];
    const uint32_t inner_size  = constants[2];

    const uint32_t num_output = outer_size * inner_size;
    if (gid >= num_output) return;

    const uint32_t outer_idx = gid / inner_size;
    const uint32_t inner_idx = gid % inner_size;

    // Phase 1: Strided partial reduction
    half partial = half(INFINITY);
    for (uint32_t r = tid; r < reduce_size; r += THREADGROUP_SIZE) {
        const uint32_t input_idx = (outer_idx * reduce_size + r) * inner_size + inner_idx;
        partial = min(partial, input[input_idx]);
    }

    // Phase 2: SIMD-group reduction
    partial = simd_min(partial);

    // Phase 3: Store to shared memory
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        half val = shared[tid];
        val = simd_min(val);
        if (tid == 0) {
            output[gid] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Argmax kernels
// ---------------------------------------------------------------------------

/// Helper struct for argmax/argmin reduction
struct ValueIndexPair {
    float value;
    uint32_t index;
};

/// Manual SIMD reduction for argmax (no built-in simd_argmax)
inline ValueIndexPair simd_argmax_reduce(ValueIndexPair partial, uint simd_lid) {
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = simd_shuffle_down(partial.value, offset);
        uint32_t other_idx = simd_shuffle_down(partial.index, offset);

        if (simd_lid + offset < SIMD_SIZE) {
            if (other_val > partial.value) {
                partial.value = other_val;
                partial.index = other_idx;
            }
        }
    }
    return partial;
}

/// Manual SIMD reduction for argmin
inline ValueIndexPair simd_argmin_reduce(ValueIndexPair partial, uint simd_lid) {
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = simd_shuffle_down(partial.value, offset);
        uint32_t other_idx = simd_shuffle_down(partial.index, offset);

        if (simd_lid + offset < SIMD_SIZE) {
            if (other_val < partial.value) {
                partial.value = other_val;
                partial.index = other_idx;
            }
        }
    }
    return partial;
}

kernel void argmax_fp32(
    device const float*  input      [[buffer(0)]],
    device int64_t*      output     [[buffer(1)]],
    constant uint32_t*   constants  [[buffer(2)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup ValueIndexPair* shared [[threadgroup(0)]])
{
    const uint32_t outer_size  = constants[0];
    const uint32_t reduce_size = constants[1];
    const uint32_t inner_size  = constants[2];

    const uint32_t num_output = outer_size * inner_size;
    if (gid >= num_output) return;

    const uint32_t outer_idx = gid / inner_size;
    const uint32_t inner_idx = gid % inner_size;

    // Phase 1: Strided partial reduction
    ValueIndexPair partial = {-INFINITY, 0};
    for (uint32_t r = tid; r < reduce_size; r += THREADGROUP_SIZE) {
        const uint32_t input_idx = (outer_idx * reduce_size + r) * inner_size + inner_idx;
        const float val = input[input_idx];
        if (val > partial.value) {
            partial.value = val;
            partial.index = r;
        }
    }

    // Phase 2: SIMD-group reduction
    partial = simd_argmax_reduce(partial, simd_lid);

    // Phase 3: Store to shared memory
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        ValueIndexPair val = shared[tid];
        val = simd_argmax_reduce(val, simd_lid);
        if (tid == 0) {
            output[gid] = int64_t(val.index);
        }
    }
}

kernel void argmax_fp16(
    device const half*  input      [[buffer(0)]],
    device int64_t*     output     [[buffer(1)]],
    constant uint32_t*  constants  [[buffer(2)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup ValueIndexPair* shared [[threadgroup(0)]])
{
    const uint32_t outer_size  = constants[0];
    const uint32_t reduce_size = constants[1];
    const uint32_t inner_size  = constants[2];

    const uint32_t num_output = outer_size * inner_size;
    if (gid >= num_output) return;

    const uint32_t outer_idx = gid / inner_size;
    const uint32_t inner_idx = gid % inner_size;

    // Phase 1: Strided partial reduction (convert to float for comparison)
    ValueIndexPair partial = {-INFINITY, 0};
    for (uint32_t r = tid; r < reduce_size; r += THREADGROUP_SIZE) {
        const uint32_t input_idx = (outer_idx * reduce_size + r) * inner_size + inner_idx;
        const float val = float(input[input_idx]);
        if (val > partial.value) {
            partial.value = val;
            partial.index = r;
        }
    }

    // Phase 2: SIMD-group reduction
    partial = simd_argmax_reduce(partial, simd_lid);

    // Phase 3: Store to shared memory
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        ValueIndexPair val = shared[tid];
        val = simd_argmax_reduce(val, simd_lid);
        if (tid == 0) {
            output[gid] = int64_t(val.index);
        }
    }
}

// ---------------------------------------------------------------------------
// Argmin kernels
// ---------------------------------------------------------------------------

kernel void argmin_fp32(
    device const float*  input      [[buffer(0)]],
    device int64_t*      output     [[buffer(1)]],
    constant uint32_t*   constants  [[buffer(2)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup ValueIndexPair* shared [[threadgroup(0)]])
{
    const uint32_t outer_size  = constants[0];
    const uint32_t reduce_size = constants[1];
    const uint32_t inner_size  = constants[2];

    const uint32_t num_output = outer_size * inner_size;
    if (gid >= num_output) return;

    const uint32_t outer_idx = gid / inner_size;
    const uint32_t inner_idx = gid % inner_size;

    // Phase 1: Strided partial reduction
    ValueIndexPair partial = {INFINITY, 0};
    for (uint32_t r = tid; r < reduce_size; r += THREADGROUP_SIZE) {
        const uint32_t input_idx = (outer_idx * reduce_size + r) * inner_size + inner_idx;
        const float val = input[input_idx];
        if (val < partial.value) {
            partial.value = val;
            partial.index = r;
        }
    }

    // Phase 2: SIMD-group reduction
    partial = simd_argmin_reduce(partial, simd_lid);

    // Phase 3: Store to shared memory
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        ValueIndexPair val = shared[tid];
        val = simd_argmin_reduce(val, simd_lid);
        if (tid == 0) {
            output[gid] = int64_t(val.index);
        }
    }
}

kernel void argmin_fp16(
    device const half*  input      [[buffer(0)]],
    device int64_t*     output     [[buffer(1)]],
    constant uint32_t*  constants  [[buffer(2)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup ValueIndexPair* shared [[threadgroup(0)]])
{
    const uint32_t outer_size  = constants[0];
    const uint32_t reduce_size = constants[1];
    const uint32_t inner_size  = constants[2];

    const uint32_t num_output = outer_size * inner_size;
    if (gid >= num_output) return;

    const uint32_t outer_idx = gid / inner_size;
    const uint32_t inner_idx = gid % inner_size;

    // Phase 1: Strided partial reduction (convert to float for comparison)
    ValueIndexPair partial = {INFINITY, 0};
    for (uint32_t r = tid; r < reduce_size; r += THREADGROUP_SIZE) {
        const uint32_t input_idx = (outer_idx * reduce_size + r) * inner_size + inner_idx;
        const float val = float(input[input_idx]);
        if (val < partial.value) {
            partial.value = val;
            partial.index = r;
        }
    }

    // Phase 2: SIMD-group reduction
    partial = simd_argmin_reduce(partial, simd_lid);

    // Phase 3: Store to shared memory
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        ValueIndexPair val = shared[tid];
        val = simd_argmin_reduce(val, simd_lid);
        if (tid == 0) {
            output[gid] = int64_t(val.index);
        }
    }
}

// ---------------------------------------------------------------------------
// Vectorized reduction kernels (vec4) for contiguous case (inner_size == 1)
// ---------------------------------------------------------------------------

/// Reduce sum for FP32 with vec4 loads (inner_size == 1 only).
kernel void reduce_sum_vec4_fp32(
    device const float* input      [[buffer(0)]],
    device float*       output     [[buffer(1)]],
    constant uint32_t*  constants  [[buffer(2)]],  // [outer_size, reduce_size, inner_size=1]
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    const uint reduce_size = constants[1];

    // Phase 1: Strided partial reduction with vec4 loads
    const uint base = gid * reduce_size;
    const uint vec_count = reduce_size / 4;
    const uint remainder = reduce_size % 4;

    device const float4* in4 = reinterpret_cast<device const float4*>(input + base);
    float partial = 0.0f;

    for (uint i = tid; i < vec_count; i += THREADGROUP_SIZE) {
        float4 v = in4[i];
        partial += v.x + v.y + v.z + v.w;
    }

    // Handle remainder elements
    uint rem_base = vec_count * 4;
    if (tid < remainder) {
        partial += input[base + rem_base + tid];
    }

    // Phase 2: SIMD group reduction
    partial = simd_sum(partial);

    // Phase 3: Store SIMD group results
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        float val = shared[tid];
        val = simd_sum(val);
        if (tid == 0) {
            output[gid] = val;
        }
    }
}

/// Reduce sum for FP16 with vec4 loads and FP32 accumulation (inner_size == 1 only).
kernel void reduce_sum_vec4_fp16(
    device const half* input       [[buffer(0)]],
    device float*      output      [[buffer(1)]],
    constant uint32_t* constants   [[buffer(2)]],  // [outer_size, reduce_size, inner_size=1]
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    const uint reduce_size = constants[1];

    // Phase 1: Strided partial reduction with vec4 loads
    const uint base = gid * reduce_size;
    const uint vec_count = reduce_size / 4;
    const uint remainder = reduce_size % 4;

    device const half4* in4 = reinterpret_cast<device const half4*>(input + base);
    float partial = 0.0f;

    for (uint i = tid; i < vec_count; i += THREADGROUP_SIZE) {
        half4 v = in4[i];
        partial += float(v.x) + float(v.y) + float(v.z) + float(v.w);
    }

    // Handle remainder elements
    uint rem_base = vec_count * 4;
    if (tid < remainder) {
        partial += float(input[base + rem_base + tid]);
    }

    // Phase 2: SIMD group reduction
    partial = simd_sum(partial);

    // Phase 3: Store SIMD group results
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        float val = shared[tid];
        val = simd_sum(val);
        if (tid == 0) {
            output[gid] = val;
        }
    }
}

/// Reduce mean for FP32 with vec4 loads (inner_size == 1 only).
kernel void reduce_mean_vec4_fp32(
    device const float* input      [[buffer(0)]],
    device float*       output     [[buffer(1)]],
    constant uint32_t*  constants  [[buffer(2)]],  // [outer_size, reduce_size, inner_size=1]
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    const uint reduce_size = constants[1];

    // Phase 1: Strided partial reduction with vec4 loads
    const uint base = gid * reduce_size;
    const uint vec_count = reduce_size / 4;
    const uint remainder = reduce_size % 4;

    device const float4* in4 = reinterpret_cast<device const float4*>(input + base);
    float partial = 0.0f;

    for (uint i = tid; i < vec_count; i += THREADGROUP_SIZE) {
        float4 v = in4[i];
        partial += v.x + v.y + v.z + v.w;
    }

    // Handle remainder elements
    uint rem_base = vec_count * 4;
    if (tid < remainder) {
        partial += input[base + rem_base + tid];
    }

    // Phase 2: SIMD group reduction
    partial = simd_sum(partial);

    // Phase 3: Store SIMD group results
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        float val = shared[tid];
        val = simd_sum(val);
        if (tid == 0) {
            output[gid] = val / float(reduce_size);
        }
    }
}

/// Reduce mean for FP16 with vec4 loads and FP32 accumulation (inner_size == 1 only).
kernel void reduce_mean_vec4_fp16(
    device const half* input       [[buffer(0)]],
    device float*      output      [[buffer(1)]],
    constant uint32_t* constants   [[buffer(2)]],  // [outer_size, reduce_size, inner_size=1]
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    const uint reduce_size = constants[1];

    // Phase 1: Strided partial reduction with vec4 loads
    const uint base = gid * reduce_size;
    const uint vec_count = reduce_size / 4;
    const uint remainder = reduce_size % 4;

    device const half4* in4 = reinterpret_cast<device const half4*>(input + base);
    float partial = 0.0f;

    for (uint i = tid; i < vec_count; i += THREADGROUP_SIZE) {
        half4 v = in4[i];
        partial += float(v.x) + float(v.y) + float(v.z) + float(v.w);
    }

    // Handle remainder elements
    uint rem_base = vec_count * 4;
    if (tid < remainder) {
        partial += float(input[base + rem_base + tid]);
    }

    // Phase 2: SIMD group reduction
    partial = simd_sum(partial);

    // Phase 3: Store SIMD group results
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        float val = shared[tid];
        val = simd_sum(val);
        if (tid == 0) {
            output[gid] = val / float(reduce_size);
        }
    }
}

/// Reduce max for FP32 with vec4 loads (inner_size == 1 only).
kernel void reduce_max_vec4_fp32(
    device const float* input      [[buffer(0)]],
    device float*       output     [[buffer(1)]],
    constant uint32_t*  constants  [[buffer(2)]],  // [outer_size, reduce_size, inner_size=1]
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    const uint reduce_size = constants[1];

    // Phase 1: Strided partial reduction with vec4 loads
    const uint base = gid * reduce_size;
    const uint vec_count = reduce_size / 4;
    const uint remainder = reduce_size % 4;

    device const float4* in4 = reinterpret_cast<device const float4*>(input + base);
    float partial = -INFINITY;

    for (uint i = tid; i < vec_count; i += THREADGROUP_SIZE) {
        float4 v = in4[i];
        partial = max(partial, max(max(v.x, v.y), max(v.z, v.w)));
    }

    // Handle remainder elements
    uint rem_base = vec_count * 4;
    if (tid < remainder) {
        partial = max(partial, input[base + rem_base + tid]);
    }

    // Phase 2: SIMD group reduction
    partial = simd_max(partial);

    // Phase 3: Store SIMD group results
    if (simd_lid == 0) {
        shared[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        float val = shared[tid];
        val = simd_max(val);
        if (tid == 0) {
            output[gid] = val;
        }
    }
}

/// Reduce max for FP16 with vec4 loads and half output (inner_size == 1 only).
kernel void reduce_max_vec4_fp16(
    device const half* input       [[buffer(0)]],
    device half*       output      [[buffer(1)]],
    constant uint32_t* constants   [[buffer(2)]],  // [outer_size, reduce_size, inner_size=1]
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    threadgroup half* shared [[threadgroup(0)]])
{
    const uint reduce_size = constants[1];

    // Phase 1: Strided partial reduction with vec4 loads
    const uint base = gid * reduce_size;
    const uint vec_count = reduce_size / 4;
    const uint remainder = reduce_size % 4;

    device const half4* in4 = reinterpret_cast<device const half4*>(input + base);
    float partial = -INFINITY;

    for (uint i = tid; i < vec_count; i += THREADGROUP_SIZE) {
        half4 v = in4[i];
        partial = max(partial, max(max(float(v.x), float(v.y)), max(float(v.z), float(v.w))));
    }

    // Handle remainder elements
    uint rem_base = vec_count * 4;
    if (tid < remainder) {
        partial = max(partial, float(input[base + rem_base + tid]));
    }

    // Phase 2: SIMD group reduction
    partial = simd_max(partial);

    // Phase 3: Store SIMD group results
    if (simd_lid == 0) {
        shared[simd_gid] = half(partial);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction by first SIMD group
    if (tid < NUM_SIMD_GROUPS) {
        float val = float(shared[tid]);
        val = simd_max(val);
        if (tid == 0) {
            output[gid] = half(val);
        }
    }
}
