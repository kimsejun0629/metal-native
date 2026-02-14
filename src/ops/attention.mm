/// @file attention.mm
/// @brief Objective-C++ implementation of FlashAttention.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/ops/attention.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/kernels/kernel_registry.h"
#include "metal_native/dispatch/command_pipeline.h"
#include "metal_native/memory/kv_cache.h"
#include "metal_native/memory/budget_controller.h"
#include <cstring>

namespace metal_native {

namespace {

constexpr uint32_t TILE_SIZE = 16;
constexpr uint32_t TILE_SIZE_24 = 24;

// Note: tile32 variant (if added) must only be used when head_dim <= 64
// due to threadgroup memory constraints:
//   tile32 + head_dim=128 + FP16 = 40,960 bytes (exceeds 32KB limit)
//   tile32 + head_dim=64  + FP16 = 24,576 bytes (within 32KB limit)
// The constraint is enforced in kernel selection logic below.

} // anonymous namespace

MNTensor flash_attention(const MNTensor& query,
                         const MNTensor& key,
                         const MNTensor& value,
                         const MNTensor* mask,
                         float scale) {
    // Validate input shapes: [batch, num_heads, seq_len, head_dim]
    MN_CHECK(query.ndim() == 4,
             MetalNativeError::InvalidArgument,
             "flash_attention: query must be 4D [batch, num_heads, seq_len_q, head_dim]");
    MN_CHECK(key.ndim() == 4,
             MetalNativeError::InvalidArgument,
             "flash_attention: key must be 4D [batch, num_heads, seq_len_k, head_dim]");
    MN_CHECK(value.ndim() == 4,
             MetalNativeError::InvalidArgument,
             "flash_attention: value must be 4D [batch, num_heads, seq_len_v, head_dim]");

    MN_CHECK(query.dtype() == key.dtype() && query.dtype() == value.dtype(),
             MetalNativeError::InvalidArgument,
             "flash_attention: all inputs must have the same dtype");

    const int64_t batch = query.shape()[0];
    const int64_t num_heads = query.shape()[1];
    const int64_t seq_len_q = query.shape()[2];
    const int64_t head_dim = query.shape()[3];
    const int64_t seq_len_k = key.shape()[2];
    const int64_t seq_len_v = value.shape()[2];

    MN_CHECK(seq_len_k == seq_len_v,
             MetalNativeError::InvalidArgument,
             "flash_attention: key and value sequence lengths must match");

    MN_CHECK(batch == key.shape()[0] && batch == value.shape()[0],
             MetalNativeError::InvalidArgument,
             "flash_attention: batch dimensions must match");

    MN_CHECK(num_heads == key.shape()[1] && num_heads == value.shape()[1],
             MetalNativeError::InvalidArgument,
             "flash_attention: num_heads must match across inputs");

    MN_CHECK(head_dim == key.shape()[3] && head_dim == value.shape()[3],
             MetalNativeError::InvalidArgument,
             "flash_attention: head_dim must match across inputs");

    if (mask != nullptr) {
        MN_CHECK(mask->ndim() == 4,
                 MetalNativeError::InvalidArgument,
                 "flash_attention: mask must be 4D [batch, 1, seq_len_q, seq_len_k]");
        MN_CHECK(mask->shape()[0] == batch && mask->shape()[2] == seq_len_q &&
                 mask->shape()[3] == seq_len_k,
                 MetalNativeError::InvalidArgument,
                 "flash_attention: mask shape incompatible with query/key");
    }

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();

        // Allocate output tensor (GPU-only when preferred for reduced TLB pressure)
        StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
        MNTensor output = MNTensor::empty(query.shape(), query.dtype(), device, out_mode);

        // Select kernel based on dtype (use SIMD versions for AMX acceleration)
        const char* kernel_name = nullptr;
        MTLSize threadgroup_size;

        if (query.dtype() == MNDType::Float32) {
            kernel_name = "flash_attention_simd_kernel";
            threadgroup_size = MTLSizeMake(32, 1, 1);  // 1 SIMD group
        } else if (query.dtype() == MNDType::Float16) {
            // Use larger tile when head_dim fits in 32KB threadgroup memory
            // and memory pressure is normal (budget allows performance optimization)
            float pressure_mult = MemoryBudgetController::instance().pressure_multiplier();

            // Memory pressure fallback: use smaller tile to minimize threadgroup memory
            // tile16 uses ~17KB vs tile24's ~29KB for FP16+head_dim=128
            // At Warning or Critical pressure, conserve threadgroup memory
            bool use_tile24 = (head_dim <= 128 && pressure_mult >= 1.0f);

            // Use multi-SIMD kernel for FP16 when sequence length benefits from parallelism
            // The multi-SIMD variant uses 2 SIMD groups (64 threads) for:
            // - 2x faster cooperative K/V tile loading
            // - Parallel computation of 8x16 score blocks
            // Memory footprint is identical to tile16 (17,408 bytes)
            bool use_multi_simd = (seq_len_k >= 64);  // Benefit from parallelism

            if (use_tile24) {
                kernel_name = "flash_attention_simd_kernel_fp16_tile24";
                threadgroup_size = MTLSizeMake(32, 1, 1);  // 1 SIMD group
            } else if (use_multi_simd) {
                kernel_name = "flash_attention_simd_kernel_fp16_multi";
                threadgroup_size = MTLSizeMake(64, 1, 1);  // 2 SIMD groups
            } else {
                kernel_name = "flash_attention_simd_kernel_fp16";
                threadgroup_size = MTLSizeMake(32, 1, 1);  // 1 SIMD group
            }
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "flash_attention: unsupported dtype (only Float32 and Float16)");
        }

        id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);

        // Create command buffer and encoder
        CommandPipeline& cmd_pipeline = device.command_pipeline();
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Bind buffers
        [encoder setBuffer:query.buffer()->metal_buffer() offset:query.offset() atIndex:0];
        [encoder setBuffer:key.buffer()->metal_buffer() offset:key.offset() atIndex:1];
        [encoder setBuffer:value.buffer()->metal_buffer() offset:value.offset() atIndex:2];

        if (mask != nullptr) {
            [encoder setBuffer:mask->buffer()->metal_buffer() offset:mask->offset() atIndex:3];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:3];
        }

        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:4];

        // Set parameters
        MN_CHECK(batch <= UINT32_MAX, MetalNativeError::InvalidArgument, "attention: dimension exceeds uint32_t range");
        MN_CHECK(num_heads <= UINT32_MAX, MetalNativeError::InvalidArgument, "attention: dimension exceeds uint32_t range");
        MN_CHECK(seq_len_q <= UINT32_MAX, MetalNativeError::InvalidArgument, "attention: dimension exceeds uint32_t range");
        MN_CHECK(seq_len_k <= UINT32_MAX, MetalNativeError::InvalidArgument, "attention: dimension exceeds uint32_t range");
        MN_CHECK(head_dim <= UINT32_MAX, MetalNativeError::InvalidArgument, "attention: dimension exceeds uint32_t range");
        uint32_t batch_u32 = static_cast<uint32_t>(batch);
        uint32_t num_heads_u32 = static_cast<uint32_t>(num_heads);
        uint32_t seq_len_q_u32 = static_cast<uint32_t>(seq_len_q);
        uint32_t seq_len_k_u32 = static_cast<uint32_t>(seq_len_k);
        uint32_t head_dim_u32 = static_cast<uint32_t>(head_dim);
        bool has_mask = (mask != nullptr);

        [encoder setBytes:&batch_u32 length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&num_heads_u32 length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&seq_len_q_u32 length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&seq_len_k_u32 length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&head_dim_u32 length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&scale length:sizeof(float) atIndex:10];
        [encoder setBytes:&has_mask length:sizeof(bool) atIndex:11];

        // Calculate threadgroup memory size for SIMD kernel
        // shared_K is reused for V (they're never alive simultaneously)
        size_t element_size = (query.dtype() == MNDType::Float32) ? sizeof(float) : sizeof(uint16_t);

        // Determine active tile size based on selected kernel
        uint32_t active_tile_size = TILE_SIZE;
        if (kernel_name && std::strcmp(kernel_name, "flash_attention_simd_kernel_fp16_tile24") == 0) {
            active_tile_size = TILE_SIZE_24;
        }

        // shared_Q: threadgroup(0)
        [encoder setThreadgroupMemoryLength:active_tile_size * head_dim * element_size atIndex:0];
        // shared_KV: threadgroup(1) - K then V (aliased)
        [encoder setThreadgroupMemoryLength:active_tile_size * head_dim * element_size atIndex:1];
        // shared_scores: threadgroup(2) - always FP32
        [encoder setThreadgroupMemoryLength:active_tile_size * active_tile_size * sizeof(float) atIndex:2];
        // shared_output: threadgroup(3) - always FP32 for accumulation
        [encoder setThreadgroupMemoryLength:active_tile_size * head_dim * sizeof(float) atIndex:3];

        // Dispatch threadgroups for SIMD kernel
        // Grid: [num_q_tiles, num_heads, batch]
        uint32_t num_q_tiles = (seq_len_q + active_tile_size - 1) / active_tile_size;

        MTLSize grid_size = MTLSizeMake(num_q_tiles, num_heads, batch);

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];

        [encoder endEncoding];
        // OPT-5: Use commit_and_continue to allow command buffer reuse.
        // The next operation can encode into a fresh buffer without waiting for this one.
        cmd_pipeline.commit_and_continue();

        return output;
    }
}

MNTensor flash_attention_with_kv_cache(
    const MNTensor& query,
    KVCache& kv_cache,
    size_t layer,
    size_t position,
    const MNTensor* mask,
    float scale) {

    // Get cached K and V for this layer
    MNTensor cached_key = kv_cache.get_key(layer);
    MNTensor cached_value = kv_cache.get_value(layer);

    // Delegate to standard flash_attention
    return flash_attention(query, cached_key, cached_value, mask, scale);
}

void fused_qkv_split_rope(
    const MNTensor& qkv_proj,
    const MNTensor& cos_table,
    const MNTensor& sin_table,
    MNTensor& q_out,
    MNTensor& k_out,
    MNTensor& v_out,
    int64_t batch,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t start_pos) {

    // Validate input shapes
    MN_CHECK(qkv_proj.ndim() == 3,
             MetalNativeError::InvalidArgument,
             "fused_qkv_split_rope: qkv_proj must be 3D [batch, seq_len, 3*num_heads*head_dim]");

    MN_CHECK(qkv_proj.shape()[0] == batch,
             MetalNativeError::InvalidArgument,
             "fused_qkv_split_rope: qkv_proj batch dimension mismatch");

    MN_CHECK(qkv_proj.shape()[1] == seq_len,
             MetalNativeError::InvalidArgument,
             "fused_qkv_split_rope: qkv_proj seq_len dimension mismatch");

    MN_CHECK(qkv_proj.shape()[2] == 3 * num_heads * head_dim,
             MetalNativeError::InvalidArgument,
             "fused_qkv_split_rope: qkv_proj last dimension must be 3*num_heads*head_dim");

    MN_CHECK(head_dim % 2 == 0,
             MetalNativeError::InvalidArgument,
             "fused_qkv_split_rope: head_dim must be even for RoPE");

    MN_CHECK(cos_table.ndim() == 2 && sin_table.ndim() == 2,
             MetalNativeError::InvalidArgument,
             "fused_qkv_split_rope: cos_table and sin_table must be 2D");

    MN_CHECK(cos_table.shape()[1] == head_dim / 2 && sin_table.shape()[1] == head_dim / 2,
             MetalNativeError::InvalidArgument,
             "fused_qkv_split_rope: cos_table and sin_table second dim must be head_dim/2");

    MN_CHECK(qkv_proj.dtype() == cos_table.dtype() && qkv_proj.dtype() == sin_table.dtype(),
             MetalNativeError::InvalidArgument,
             "fused_qkv_split_rope: all inputs must have the same dtype");

    // Validate output shapes
    MN_CHECK(q_out.ndim() == 4 && k_out.ndim() == 4 && v_out.ndim() == 4,
             MetalNativeError::InvalidArgument,
             "fused_qkv_split_rope: output tensors must be 4D [batch, num_heads, seq_len, head_dim]");

    MN_CHECK(q_out.shape()[0] == batch && q_out.shape()[1] == num_heads &&
             q_out.shape()[2] == seq_len && q_out.shape()[3] == head_dim,
             MetalNativeError::InvalidArgument,
             "fused_qkv_split_rope: q_out shape mismatch");

    MN_CHECK(k_out.shape()[0] == batch && k_out.shape()[1] == num_heads &&
             k_out.shape()[2] == seq_len && k_out.shape()[3] == head_dim,
             MetalNativeError::InvalidArgument,
             "fused_qkv_split_rope: k_out shape mismatch");

    MN_CHECK(v_out.shape()[0] == batch && v_out.shape()[1] == num_heads &&
             v_out.shape()[2] == seq_len && v_out.shape()[3] == head_dim,
             MetalNativeError::InvalidArgument,
             "fused_qkv_split_rope: v_out shape mismatch");

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();

        // Select kernel based on dtype and vectorization capability
        const char* kernel_name = nullptr;
        bool use_vectorized = (head_dim % 4 == 0);

        if (qkv_proj.dtype() == MNDType::Float32) {
            kernel_name = use_vectorized ? "fused_qkv_rope_fp32_vec4" : "fused_qkv_rope_fp32";
        } else if (qkv_proj.dtype() == MNDType::Float16) {
            kernel_name = use_vectorized ? "fused_qkv_rope_fp16_vec4" : "fused_qkv_rope_fp16";
        } else {
            MN_THROW(MetalNativeError::InvalidArgument,
                     "fused_qkv_split_rope: unsupported dtype (only Float32 and Float16)");
        }

        id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);

        // Create command buffer and encoder
        CommandPipeline& cmd_pipeline = device.command_pipeline();
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Bind buffers
        [encoder setBuffer:qkv_proj.buffer()->metal_buffer() offset:qkv_proj.offset() atIndex:0];
        [encoder setBuffer:cos_table.buffer()->metal_buffer() offset:cos_table.offset() atIndex:1];
        [encoder setBuffer:sin_table.buffer()->metal_buffer() offset:sin_table.offset() atIndex:2];
        [encoder setBuffer:q_out.buffer()->metal_buffer() offset:q_out.offset() atIndex:3];
        [encoder setBuffer:k_out.buffer()->metal_buffer() offset:k_out.offset() atIndex:4];
        [encoder setBuffer:v_out.buffer()->metal_buffer() offset:v_out.offset() atIndex:5];

        // Set parameters
        MN_CHECK(batch <= UINT32_MAX, MetalNativeError::InvalidArgument, "fused_qkv_split_rope: dimension exceeds uint32_t range");
        MN_CHECK(seq_len <= UINT32_MAX, MetalNativeError::InvalidArgument, "fused_qkv_split_rope: dimension exceeds uint32_t range");
        MN_CHECK(num_heads <= UINT32_MAX, MetalNativeError::InvalidArgument, "fused_qkv_split_rope: dimension exceeds uint32_t range");
        MN_CHECK(head_dim <= UINT32_MAX, MetalNativeError::InvalidArgument, "fused_qkv_split_rope: dimension exceeds uint32_t range");
        MN_CHECK(start_pos <= UINT32_MAX, MetalNativeError::InvalidArgument, "fused_qkv_split_rope: dimension exceeds uint32_t range");

        uint32_t batch_u32 = static_cast<uint32_t>(batch);
        uint32_t seq_len_u32 = static_cast<uint32_t>(seq_len);
        uint32_t num_heads_u32 = static_cast<uint32_t>(num_heads);
        uint32_t head_dim_u32 = static_cast<uint32_t>(head_dim);
        uint32_t start_pos_u32 = static_cast<uint32_t>(start_pos);

        [encoder setBytes:&batch_u32 length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&seq_len_u32 length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&num_heads_u32 length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&head_dim_u32 length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&start_pos_u32 length:sizeof(uint32_t) atIndex:10];

        // Dispatch threadgroups
        // Grid: [head_dim/2 (or head_dim/4 for vec4), seq_len, batch * num_heads]
        uint32_t grid_x = use_vectorized ? (head_dim / 4) : (head_dim / 2);
        MTLSize grid_size = MTLSizeMake(grid_x, seq_len, batch * num_heads);
        MTLSize threadgroup_size = MTLSizeMake(32, 1, 1);  // Use warp size

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];

        [encoder endEncoding];
        cmd_pipeline.commit_and_continue();
    }
}

} // namespace metal_native
