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

        // Allocate output tensor
        MNTensor output = MNTensor::empty(query.shape(), query.dtype(), device);

        // Select kernel based on dtype (use SIMD versions for AMX acceleration)
        const char* kernel_name = nullptr;
        if (query.dtype() == MNDType::Float32) {
            kernel_name = "flash_attention_simd_kernel";
        } else if (query.dtype() == MNDType::Float16) {
            // Use larger tile when head_dim fits in 32KB threadgroup memory
            // and memory pressure is normal (budget allows performance optimization)
            float pressure_mult = MemoryBudgetController::instance().pressure_multiplier();
            if (head_dim <= 128 && pressure_mult >= 1.0f) {
                kernel_name = "flash_attention_simd_kernel_fp16_tile24";
            } else {
                kernel_name = "flash_attention_simd_kernel_fp16";
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
        // SIMD kernel uses one SIMD group (32 threads) per threadgroup
        // Grid: [num_q_tiles, num_heads, batch]
        uint32_t num_q_tiles = (seq_len_q + active_tile_size - 1) / active_tile_size;

        MTLSize grid_size = MTLSizeMake(num_q_tiles, num_heads, batch);
        MTLSize threadgroup_size = MTLSizeMake(32, 1, 1);  // One SIMD group per threadgroup

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];

        [encoder endEncoding];
        cmd_pipeline.commit();

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

} // namespace metal_native
