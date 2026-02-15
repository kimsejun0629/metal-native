/// @file future_stubs.mm
/// @brief Stub implementations for future hardware features.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/future/metal4.h"
#include "metal_native/future/bfloat16.h"
#include "metal_native/future/neural_engine.h"
#include "metal_native/future/fast_ops.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/kernels/kernel_registry.h"
#include "metal_native/dispatch/command_pipeline.h"
#include "metal_native/ops/matmul.h"

namespace metal_native {
namespace future {

bool metal4_available() noexcept {
    // Metal 4 not yet available in shipping macOS
    return false;
}

Metal4Capabilities query_metal4_capabilities() noexcept {
    return Metal4Capabilities{};
}

static BF16Policy g_bf16_policy = BF16Policy::Disabled;

bool bfloat16_supported() noexcept {
    return MNDevice::instance().supports_bfloat16();
}

void set_bf16_policy(BF16Policy policy) {
    g_bf16_policy = policy;
}

BF16Policy get_bf16_policy() noexcept {
    return g_bf16_policy;
}

bool neural_engine_available() noexcept {
    // Neural Engine via Metal TensorOps not yet exposed
    return false;
}

NeuralEngineCapabilities query_neural_engine() noexcept {
    return NeuralEngineCapabilities{};
}

} // namespace future

namespace fast {

// ---------------------------------------------------------------------------
// fast::swiglu -- Fused SiLU(gate) * up
// ---------------------------------------------------------------------------

MNTensor swiglu(const MNTensor& gate, const MNTensor& up) {
    MN_CHECK(gate.dtype() == up.dtype(),
             MetalNativeError::InvalidArgument,
             "fast::swiglu: gate and up must have same dtype");
    MN_CHECK(gate.numel() == up.numel(),
             MetalNativeError::InvalidArgument,
             "fast::swiglu: gate and up must have same number of elements");

    MNDevice& device = MNDevice::instance();
    MNTensor output = MNTensor::empty(gate.shape(), gate.dtype(), device);

    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        const char* kernel_name = (gate.dtype() == MNDType::Float32)
            ? "fused_swiglu_fp32" : "fused_swiglu_fp16";

        id<MTLComputePipelineState> pipeline =
            KernelRegistry::instance().get_pipeline(kernel_name);
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:gate.buffer()->metal_buffer() offset:gate.offset() atIndex:0];
        [encoder setBuffer:up.buffer()->metal_buffer() offset:up.offset() atIndex:1];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:2];

        uint32_t num_elements = static_cast<uint32_t>(gate.numel());
        [encoder setBytes:&num_elements length:sizeof(uint32_t) atIndex:3];

        MTLSize grid_size = MTLSizeMake((num_elements + 3) / 4, 1, 1);
        MTLSize threadgroup_size = MTLSizeMake(
            std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];
        cmd_pipeline.commit_and_continue();
    }

    return output;
}

// ---------------------------------------------------------------------------
// fast::rms_norm -- Fused RMSNorm with weight
// ---------------------------------------------------------------------------

MNTensor rms_norm(const MNTensor& input, const MNTensor& weight, float eps) {
    MNDevice& device = MNDevice::instance();
    MNTensor output = MNTensor::empty(input.shape(), input.dtype(), device);

    const auto& shape = input.shape();
    uint32_t norm_size = static_cast<uint32_t>(shape[shape.ndim() - 1]);
    uint32_t batch_size = static_cast<uint32_t>(input.numel() / norm_size);

    const bool use_large = (norm_size >= 2048);

    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        const char* kernel_name = nullptr;
        if (input.dtype() == MNDType::Float32) {
            kernel_name = use_large ? "fused_rms_norm_large_fp32" : "fused_rms_norm_fp32";
        } else {
            kernel_name = use_large ? "fused_rms_norm_large_fp16" : "fused_rms_norm_fp16";
        }

        id<MTLComputePipelineState> pipeline =
            KernelRegistry::instance().get_pipeline(kernel_name);
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input.buffer()->metal_buffer() offset:input.offset() atIndex:0];
        [encoder setBuffer:weight.buffer()->metal_buffer() offset:weight.offset() atIndex:1];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:2];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&norm_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&eps length:sizeof(float) atIndex:5];

        if (use_large) {
            // 256 threads per threadgroup, one threadgroup per batch row.
            [encoder setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];

            MTLSize grid_size = MTLSizeMake(batch_size, 1, 1);
            MTLSize threadgroup_size = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];
        } else {
            MTLSize grid_size = MTLSizeMake(32, batch_size, 1);
            MTLSize threadgroup_size = MTLSizeMake(32, 1, 1);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        }

        [encoder endEncoding];
        cmd_pipeline.commit_and_continue();
    }

    return output;
}

// ---------------------------------------------------------------------------
// fast::layer_norm -- Fused LayerNorm with weight + bias
// ---------------------------------------------------------------------------

MNTensor layer_norm(const MNTensor& input, const MNTensor& weight,
                    const MNTensor& bias, float eps) {
    MNDevice& device = MNDevice::instance();
    MNTensor output = MNTensor::empty(input.shape(), input.dtype(), device);

    const auto& shape = input.shape();
    uint32_t norm_size = static_cast<uint32_t>(shape[shape.ndim() - 1]);
    uint32_t batch_size = static_cast<uint32_t>(input.numel() / norm_size);

    const bool use_large = (norm_size >= 2048);

    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        const char* kernel_name = nullptr;
        if (input.dtype() == MNDType::Float32) {
            kernel_name = use_large ? "fused_layer_norm_large_fp32" : "fused_layer_norm_fp32";
        } else {
            kernel_name = use_large ? "fused_layer_norm_large_fp16" : "fused_layer_norm_fp16";
        }

        id<MTLComputePipelineState> pipeline =
            KernelRegistry::instance().get_pipeline(kernel_name);
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input.buffer()->metal_buffer() offset:input.offset() atIndex:0];
        [encoder setBuffer:weight.buffer()->metal_buffer() offset:weight.offset() atIndex:1];
        [encoder setBuffer:bias.buffer()->metal_buffer() offset:bias.offset() atIndex:2];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:3];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&norm_size length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&eps length:sizeof(float) atIndex:6];

        if (use_large) {
            // 256 threads per threadgroup, one threadgroup per batch row.
            // 16 floats: 8 for sum + 8 for sq_sum inter-SIMD reduction.
            [encoder setThreadgroupMemoryLength:16 * sizeof(float) atIndex:0];

            MTLSize grid_size = MTLSizeMake(batch_size, 1, 1);
            MTLSize threadgroup_size = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];
        } else {
            // Threadgroup memory for caching input during 2-pass normalization
            [encoder setThreadgroupMemoryLength:norm_size * sizeof(float) atIndex:0];

            MTLSize grid_size = MTLSizeMake(32, batch_size, 1);
            MTLSize threadgroup_size = MTLSizeMake(32, 1, 1);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        }

        [encoder endEncoding];
        cmd_pipeline.commit_and_continue();
    }

    return output;
}

// ---------------------------------------------------------------------------
// fast::rope -- Fused Rotary Position Embedding for Q and K
// ---------------------------------------------------------------------------

void rope(MNTensor& q, MNTensor& k,
          const MNTensor& cos_cache, const MNTensor& sin_cache,
          int64_t start_pos) {
    MN_CHECK(q.shape().ndim() == 4,
             MetalNativeError::InvalidArgument,
             "fast::rope: Q must be 4D [batch, heads, seq_len, head_dim]");
    MN_CHECK(k.shape().ndim() == 4,
             MetalNativeError::InvalidArgument,
             "fast::rope: K must be 4D [batch, heads, seq_len, head_dim]");

    MNDevice& device = MNDevice::instance();

    const auto& shape = q.shape();
    uint32_t batch_size = static_cast<uint32_t>(shape[0]);
    uint32_t num_heads = static_cast<uint32_t>(shape[1]);
    uint32_t seq_len = static_cast<uint32_t>(shape[2]);
    uint32_t head_dim = static_cast<uint32_t>(shape[3]);
    uint32_t start_pos_u32 = static_cast<uint32_t>(start_pos);

    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        const char* kernel_name = (q.dtype() == MNDType::Float32)
            ? "fused_rope_fp32" : "fused_rope_fp16";

        id<MTLComputePipelineState> pipeline =
            KernelRegistry::instance().get_pipeline(kernel_name);
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:q.buffer()->metal_buffer() offset:q.offset() atIndex:0];
        [encoder setBuffer:k.buffer()->metal_buffer() offset:k.offset() atIndex:1];
        [encoder setBuffer:cos_cache.buffer()->metal_buffer() offset:cos_cache.offset() atIndex:2];
        [encoder setBuffer:sin_cache.buffer()->metal_buffer() offset:sin_cache.offset() atIndex:3];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&num_heads length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&seq_len length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&start_pos_u32 length:sizeof(uint32_t) atIndex:8];

        MTLSize grid_size = MTLSizeMake(head_dim / 2, seq_len, batch_size * num_heads);
        MTLSize threadgroup_size = MTLSizeMake(
            std::min<NSUInteger>(head_dim / 2, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];
        cmd_pipeline.commit_and_continue();
    }
}

// ---------------------------------------------------------------------------
// fast::fused_bias_gelu -- Fused Bias + GELU activation
// ---------------------------------------------------------------------------

MNTensor fused_bias_gelu(const MNTensor& input, const MNTensor& bias) {
    MN_CHECK(input.dtype() == bias.dtype(),
             MetalNativeError::InvalidArgument,
             "fast::fused_bias_gelu: input and bias must have same dtype");

    const auto& input_shape = input.shape();
    MN_CHECK(input_shape.ndim() >= 1,
             MetalNativeError::InvalidArgument,
             "fast::fused_bias_gelu: input must be at least 1D");

    uint32_t bias_size = static_cast<uint32_t>(bias.numel());
    uint32_t last_dim = static_cast<uint32_t>(input_shape[input_shape.ndim() - 1]);
    MN_CHECK(bias_size == last_dim,
             MetalNativeError::InvalidArgument,
             "fast::fused_bias_gelu: bias size must match input's last dimension");

    MNDevice& device = MNDevice::instance();
    MNTensor output = MNTensor::empty(input.shape(), input.dtype(), device);

    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        const char* kernel_name = (input.dtype() == MNDType::Float32)
            ? "fused_bias_gelu_fp32" : "fused_bias_gelu_fp16";

        id<MTLComputePipelineState> pipeline =
            KernelRegistry::instance().get_pipeline(kernel_name);
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input.buffer()->metal_buffer() offset:input.offset() atIndex:0];
        [encoder setBuffer:bias.buffer()->metal_buffer() offset:bias.offset() atIndex:1];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:2];

        uint32_t num_elements = static_cast<uint32_t>(input.numel());
        [encoder setBytes:&num_elements length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&bias_size length:sizeof(uint32_t) atIndex:4];

        MTLSize grid_size = MTLSizeMake((num_elements + 3) / 4, 1, 1);
        MTLSize threadgroup_size = MTLSizeMake(
            std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];
        cmd_pipeline.commit_and_continue();
    }

    return output;
}

// ---------------------------------------------------------------------------
// fast::fused_residual_norm -- Fused Residual + RMSNorm
// ---------------------------------------------------------------------------

MNTensor fused_residual_norm(const MNTensor& input, const MNTensor& residual,
                              const MNTensor& weight, float eps) {
    MN_CHECK(input.dtype() == residual.dtype(),
             MetalNativeError::InvalidArgument,
             "fast::fused_residual_norm: input and residual must have same dtype");
    MN_CHECK(input.dtype() == weight.dtype(),
             MetalNativeError::InvalidArgument,
             "fast::fused_residual_norm: input and weight must have same dtype");
    MN_CHECK(input.numel() == residual.numel(),
             MetalNativeError::InvalidArgument,
             "fast::fused_residual_norm: input and residual must have same number of elements");

    const auto& shape = input.shape();
    uint32_t norm_size = static_cast<uint32_t>(shape[shape.ndim() - 1]);
    uint32_t batch_size = static_cast<uint32_t>(input.numel() / norm_size);

    MN_CHECK(weight.numel() == norm_size,
             MetalNativeError::InvalidArgument,
             "fast::fused_residual_norm: weight size must match input's last dimension");

    MNDevice& device = MNDevice::instance();
    MNTensor output = MNTensor::empty(input.shape(), input.dtype(), device);

    const bool use_large = (norm_size >= 2048);
    const bool use_regcache = use_large && (norm_size <= 8192);  // fits in 32 registers per thread (8192/256=32)

    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        const char* kernel_name = nullptr;
        if (input.dtype() == MNDType::Float32) {
            if (use_regcache) {
                kernel_name = "fused_residual_rms_norm_regcache_fp32";
            } else if (use_large) {
                kernel_name = "fused_residual_rms_norm_large_fp32";
            } else {
                kernel_name = "fused_residual_rms_norm_fp32";
            }
        } else {
            if (use_regcache) {
                kernel_name = "fused_residual_rms_norm_regcache_fp16";
            } else if (use_large) {
                kernel_name = "fused_residual_rms_norm_large_fp16";
            } else {
                kernel_name = "fused_residual_rms_norm_fp16";
            }
        }

        id<MTLComputePipelineState> pipeline =
            KernelRegistry::instance().get_pipeline(kernel_name);
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input.buffer()->metal_buffer() offset:input.offset() atIndex:0];
        [encoder setBuffer:residual.buffer()->metal_buffer() offset:residual.offset() atIndex:1];
        [encoder setBuffer:weight.buffer()->metal_buffer() offset:weight.offset() atIndex:2];
        [encoder setBuffer:output.buffer()->metal_buffer() offset:output.offset() atIndex:3];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&norm_size length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&eps length:sizeof(float) atIndex:6];

        if (use_large) {
            // 256 threads per threadgroup, one threadgroup per batch row.
            [encoder setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];

            MTLSize grid_size = MTLSizeMake(batch_size, 1, 1);
            MTLSize threadgroup_size = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];
        } else {
            MTLSize grid_size = MTLSizeMake(32, batch_size, 1);
            MTLSize threadgroup_size = MTLSizeMake(32, 1, 1);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        }

        [encoder endEncoding];
        cmd_pipeline.commit_and_continue();
    }

    return output;
}

// ---------------------------------------------------------------------------
// fast::fused_qkv_projection -- Fused QKV Projection with single matmul
// ---------------------------------------------------------------------------

QKVResult fused_qkv_projection(const MNTensor& input,
                                const MNTensor& w_qkv,
                                int64_t num_heads,
                                int64_t head_dim) {
    // Validate input shape: [batch, seq_len, hidden_dim]
    const auto& input_shape = input.shape();
    MN_CHECK(input_shape.ndim() == 3,
             MetalNativeError::InvalidArgument,
             "fast::fused_qkv_projection: input must be 3D [batch, seq_len, hidden_dim]");

    int64_t batch_size = input_shape[0];
    int64_t seq_len = input_shape[1];
    int64_t hidden_dim = input_shape[2];

    // Validate w_qkv shape: [hidden_dim, 3*num_heads*head_dim]
    const auto& w_qkv_shape = w_qkv.shape();
    MN_CHECK(w_qkv_shape.ndim() == 2,
             MetalNativeError::InvalidArgument,
             "fast::fused_qkv_projection: w_qkv must be 2D [hidden_dim, 3*num_heads*head_dim]");
    MN_CHECK(w_qkv_shape[0] == hidden_dim,
             MetalNativeError::InvalidArgument,
             "fast::fused_qkv_projection: w_qkv first dimension must match input hidden_dim");
    MN_CHECK(w_qkv_shape[1] == 3 * num_heads * head_dim,
             MetalNativeError::InvalidArgument,
             "fast::fused_qkv_projection: w_qkv second dimension must be 3*num_heads*head_dim");

    // Step 1: Single matmul QKV = input @ w_qkv
    // Result shape: [batch, seq_len, 3*num_heads*head_dim]
    MNTensor qkv_combined = matmul(input, w_qkv);

    // Step 2: Allocate Q, K, V output tensors [batch, num_heads, seq_len, head_dim]
    MNDevice& device = MNDevice::instance();
    MNTensor q_out = MNTensor::empty(
        MNShape({batch_size, num_heads, seq_len, head_dim}),
        input.dtype(),
        device
    );
    MNTensor k_out = MNTensor::empty(
        MNShape({batch_size, num_heads, seq_len, head_dim}),
        input.dtype(),
        device
    );
    MNTensor v_out = MNTensor::empty(
        MNShape({batch_size, num_heads, seq_len, head_dim}),
        input.dtype(),
        device
    );

    // Step 3: Dispatch split+reshape kernel
    @autoreleasepool {
        CommandPipeline& cmd_pipeline = device.command_pipeline();

        const char* kernel_name = (input.dtype() == MNDType::Float32)
            ? "qkv_split_reshape_fp32" : "qkv_split_reshape_fp16";

        id<MTLComputePipelineState> pipeline =
            KernelRegistry::instance().get_pipeline(kernel_name);
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:qkv_combined.buffer()->metal_buffer() offset:qkv_combined.offset() atIndex:0];
        [encoder setBuffer:q_out.buffer()->metal_buffer() offset:q_out.offset() atIndex:1];
        [encoder setBuffer:k_out.buffer()->metal_buffer() offset:k_out.offset() atIndex:2];
        [encoder setBuffer:v_out.buffer()->metal_buffer() offset:v_out.offset() atIndex:3];

        uint32_t batch_size_u32 = static_cast<uint32_t>(batch_size);
        uint32_t seq_len_u32 = static_cast<uint32_t>(seq_len);
        uint32_t num_heads_u32 = static_cast<uint32_t>(num_heads);
        uint32_t head_dim_u32 = static_cast<uint32_t>(head_dim);

        [encoder setBytes:&batch_size_u32 length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&seq_len_u32 length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&num_heads_u32 length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&head_dim_u32 length:sizeof(uint32_t) atIndex:7];

        // Grid: [head_dim, seq_len, batch_size * num_heads]
        MTLSize grid_size = MTLSizeMake(head_dim, seq_len, batch_size * num_heads);
        MTLSize threadgroup_size = MTLSizeMake(std::min<NSUInteger>(head_dim, 256), 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];
        cmd_pipeline.commit_and_continue();
    }

    return QKVResult{q_out, k_out, v_out};
}

// ---------------------------------------------------------------------------
// fast::scaled_dot_product_attention -- Delegates to ops::attention
// ---------------------------------------------------------------------------

MNTensor scaled_dot_product_attention(const MNTensor& query,
                                       const MNTensor& key,
                                       const MNTensor& value,
                                       float scale,
                                       bool causal) {
    // Already implemented in ops/attention.mm via existing FlashAttention kernels.
    // This fast:: variant will be a thin wrapper in a future phase.
    MN_THROW(MetalNativeError::NotImplemented,
             "fast::scaled_dot_product_attention: use ops::attention() directly. "
             "Fast variant planned for Phase 4.");
}

} // namespace fast
} // namespace metal_native
