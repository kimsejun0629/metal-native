/// @file matmul.mm
/// @brief Objective-C++ implementation of matrix multiplication operators.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

#include "metal_native/ops/matmul.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/graph/graph_builder.h"
#include "metal_native/graph/graph_cache.h"
#include "metal_native/memory/budget_controller.h"
#include "metal_native/dispatch/command_pipeline.h"
#include "metal_native/kernels/kernel_registry.h"

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <mach/mach_time.h>

namespace metal_native {

namespace {

constexpr int64_t kSmallTensorThreshold = 16384; // elements (lowered: GPU dispatch overhead threshold)
constexpr int64_t kSIMDMatmulMaxDim = 512; // max dimension for custom kernels (512+ delegates to MPSGraph)

MPSDataType to_mps_datatype(MNDType dtype) {
    switch (dtype) {
        case MNDType::Float32:  return MPSDataTypeFloat32;
        case MNDType::Float16:  return MPSDataTypeFloat16;
        case MNDType::BFloat16: return MPSDataTypeBFloat16;
        case MNDType::Int32:    return MPSDataTypeInt32;
        default:
            MN_THROW(MetalNativeError::InvalidArgument,
                     "matmul: unsupported dtype");
    }
}

NSArray<NSNumber*>* to_ns_shape(const MNShape& shape) {
    NSMutableArray<NSNumber*>* ns_shape = [NSMutableArray arrayWithCapacity:shape.ndim()];
    for (size_t i = 0; i < shape.ndim(); ++i) {
        [ns_shape addObject:@(shape[static_cast<int64_t>(i)])];
    }
    return ns_shape;
}

// Cached version of to_ns_shape to avoid repeated NSArray allocations
static NSArray<NSNumber*>* cached_ns_shape(const MNShape& shape) {
    // Use thread_local to avoid locking
    thread_local std::unordered_map<uint64_t, NSArray<NSNumber*>*> shape_cache;

    // Simple hash of shape dims
    uint64_t hash = shape.ndim();
    for (size_t i = 0; i < shape.ndim(); ++i) {
        hash = hash * 31 + static_cast<uint64_t>(shape[static_cast<int64_t>(i)]);
    }

    auto it = shape_cache.find(hash);
    if (it != shape_cache.end()) {
        return it->second;
    }

    // Cache miss - create and store
    NSArray<NSNumber*>* ns_shape = to_ns_shape(shape);
    // Limit cache size
    if (shape_cache.size() > 64) shape_cache.clear();
    shape_cache[hash] = ns_shape;
    return ns_shape;
}

// Cache for MPSGraphTensorData to avoid repeated allocations
// Shared across threads with reader-writer lock for better multi-thread cache hit rates
struct TensorDataCache {
    uint64_t shape_hash = 0;
    MPSDataType dtype = MPSDataTypeFloat32;
    id<MTLBuffer> buffer = nil;
    MPSGraphTensorData* data = nil;
};

std::shared_mutex td_cache_mu;
TensorDataCache td_cache_a, td_cache_b, td_cache_result;

// Cache entry for matmul graphs
struct MatmulCacheEntry {
    MPSGraph* graph;
    MPSGraphTensor* a_placeholder;
    MPSGraphTensor* b_placeholder;
    MPSGraphTensor* result_tensor;
    uint64_t last_access;  // mach_absolute_time() for LRU
};

static uint64_t monotonic_tick() {
    return mach_absolute_time();
}

static constexpr size_t kDefaultMaxCacheEntries = 64;

std::mutex matmul_cache_mu;
std::unordered_map<GraphCacheKey, MatmulCacheEntry> matmul_graph_cache;
std::atomic<size_t> matmul_cache_max_entries{kDefaultMaxCacheEntries};

// Evict oldest entry when cache is at capacity
static void evict_lru_if_needed() {
    // Must be called with matmul_cache_mu held
    if (matmul_graph_cache.size() < matmul_cache_max_entries) return;

    // Find LRU entry
    auto oldest = matmul_graph_cache.begin();
    for (auto it = matmul_graph_cache.begin(); it != matmul_graph_cache.end(); ++it) {
        if (it->second.last_access < oldest->second.last_access) {
            oldest = it;
        }
    }
    matmul_graph_cache.erase(oldest);
}

// Compute cache key from operation parameters
GraphCacheKey make_matmul_key(const MNShape& a_shape, const MNShape& b_shape,
                               MNDType dtype, bool transpose_a, bool transpose_b) {
    GraphCacheKey key;
    // Use a simple hash for matmul topology: op_type + transpose flags
    key.topology_hash = 0x4D41544D554C0000ULL;  // "MATMUL\0\0"
    key.topology_hash ^= (transpose_a ? 1ULL : 0ULL) | (transpose_b ? 2ULL : 0ULL);

    // Shape tuple: flatten all input shapes
    for (size_t i = 0; i < a_shape.ndim(); ++i) {
        key.shape_tuple.push_back(static_cast<size_t>(a_shape[static_cast<int64_t>(i)]));
    }
    key.shape_tuple.push_back(0);  // separator
    for (size_t i = 0; i < b_shape.ndim(); ++i) {
        key.shape_tuple.push_back(static_cast<size_t>(b_shape[static_cast<int64_t>(i)]));
    }
    key.dtype = dtype;
    return key;
}

bool should_use_cpu_path(const MNTensor& a, const MNTensor& b,
                         int64_t M, int64_t N, int64_t K) {
    // Use CPU (Accelerate/CBLAS) for small FP32 matrices where GPU dispatch
    // overhead dominates. Benchmarks show:
    //   128x128: CPU 0.005ms vs GPU 0.238ms (48x faster)
    //   256x256: CPU 0.026ms vs GPU 0.312ms (12x faster)
    //   512x512: CPU 0.106ms vs GPU 0.547ms (5x faster)
    //  1024x1024: GPU wins (0.385ms vs CPU 0.726ms)
    if (a.dtype() != MNDType::Float32) return false;
    if (!a.is_contiguous() || !b.is_contiguous()) return false;

    // Dimension-based check: CPU wins when max dimension ≤ 512
    int64_t max_dim = std::max({M, N, K});
    return max_dim <= 512;
}

enum class MatmulKernelType {
    VecMat,      // M=1 or M<=8, dedicated vector-matrix kernel
    SIMD32,      // Small matrices, 32x32 tiles (existing)
    Tiled64,     // Medium matrices, 64x64 tiles (new)
    MPSGraph,    // Large matrices or non-contiguous (existing)
    CPU          // Tiny tensors (existing)
};

MatmulKernelType select_kernel(int64_t M, int64_t N, int64_t K, MNDType dtype,
                                bool is_contiguous) {
    // Non-contiguous: always MPSGraph (handles strides internally)
    if (!is_contiguous) return MatmulKernelType::MPSGraph;

    // Only FP16, FP32, and BF16 for custom kernels
    if (dtype != MNDType::Float16 && dtype != MNDType::Float32 && dtype != MNDType::BFloat16)
        return MatmulKernelType::MPSGraph;

    // Decode phase: vector-matrix (M=1..8, large N and K)
    if (M <= 8 && N >= 256 && K >= 256)
        return MatmulKernelType::VecMat;

    // Small matrices: existing 32x32 SIMD kernel (≤128 per dim)
    if (M <= 128 && N <= 128 && K <= 128)
        return MatmulKernelType::SIMD32;

    // Medium matrices: 64x64 tiled kernel (129-512 per dim)
    if (M <= kSIMDMatmulMaxDim && N <= kSIMDMatmulMaxDim && K <= kSIMDMatmulMaxDim)
        return MatmulKernelType::Tiled64;

    // Large: MPSGraph
    return MatmulKernelType::MPSGraph;
}

MNTensor matmul_vecmat(const MNTensor& a, const MNTensor& b,
                       bool transpose_a, bool transpose_b,
                       int64_t M, int64_t N, int64_t K, MNDevice& device) {
    @autoreleasepool {
        // Output shape
        std::vector<int64_t> output_dims;
        for (size_t i = 0; i < a.ndim() - 2; ++i)
            output_dims.push_back(a.shape()[static_cast<int64_t>(i)]);
        output_dims.push_back(M);
        output_dims.push_back(N);
        MNShape output_shape(output_dims);

        StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
        MNTensor result = MNTensor::empty(output_shape, a.dtype(), device, out_mode);

        // Select kernel
        const char* kernel_name = nullptr;
        int64_t batch_size = 1;
        for (size_t i = 0; i < a.ndim() - 2; ++i) {
            batch_size *= a.shape()[static_cast<int64_t>(i)];
        }

        bool use_multi2 = (M == 1 && K >= 512);  // Multi-output when K is large enough

        if (M > 1) {
            kernel_name = (a.dtype() == MNDType::Float16) ?
                "matmul_vecmat_batch_fp16" : "matmul_vecmat_batch_fp32";
        } else if (use_multi2) {
            kernel_name = (a.dtype() == MNDType::Float16) ?
                "matmul_vecmat_fp16_multi2" : "matmul_vecmat_fp32";
        } else {
            kernel_name = (a.dtype() == MNDType::Float16) ?
                "matmul_vecmat_fp16" : "matmul_vecmat_fp32";
        }

        KernelRegistry& registry = KernelRegistry::instance();
        id<MTLComputePipelineState> pipeline = registry.get_pipeline(kernel_name);

        CommandPipeline& cmd_pipeline = device.command_pipeline();
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Bind buffers
        [encoder setBuffer:a.buffer()->metal_buffer() offset:0 atIndex:0];
        [encoder setBuffer:b.buffer()->metal_buffer() offset:0 atIndex:1];
        [encoder setBuffer:result.buffer()->metal_buffer() offset:0 atIndex:2];

        // Set matrix dimensions
        uint32_t M_u32 = static_cast<uint32_t>(M);
        uint32_t N_u32 = static_cast<uint32_t>(N);
        uint32_t K_u32 = static_cast<uint32_t>(K);
        [encoder setBytes:&M_u32 length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&N_u32 length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&K_u32 length:sizeof(uint32_t) atIndex:5];

        // Set leading dimensions
        uint32_t lda = static_cast<uint32_t>(a.shape()[a.ndim() - 1]);
        uint32_t ldb = static_cast<uint32_t>(b.shape()[b.ndim() - 1]);
        uint32_t ldc = static_cast<uint32_t>(N);
        [encoder setBytes:&lda length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&ldb length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&ldc length:sizeof(uint32_t) atIndex:8];

        // Set transpose flags
        [encoder setBytes:&transpose_a length:sizeof(bool) atIndex:9];
        [encoder setBytes:&transpose_b length:sizeof(bool) atIndex:10];

        // For batch variant, set batch strides
        if (M > 1) {
            uint32_t batch_stride_a = static_cast<uint32_t>(
                a.shape()[a.ndim() - 2] * a.shape()[a.ndim() - 1]);
            uint32_t batch_stride_b = static_cast<uint32_t>(
                b.shape()[b.ndim() - 2] * b.shape()[b.ndim() - 1]);
            uint32_t batch_stride_c = static_cast<uint32_t>(M * N);
            [encoder setBytes:&batch_stride_a length:sizeof(uint32_t) atIndex:11];
            [encoder setBytes:&batch_stride_b length:sizeof(uint32_t) atIndex:12];
            [encoder setBytes:&batch_stride_c length:sizeof(uint32_t) atIndex:13];
        }

        // Dispatch: 256 threads per threadgroup (8 SIMD groups)
        // Grid: (ceil(N/8), M, batch_size) for batch variant
        //       (ceil(N/16), 1, 1) for M=1 multi2 variant (16 outputs per threadgroup)
        //       (ceil(N/8), 1, 1) for M=1 standard variant (8 outputs per threadgroup)
        uint64_t grid_x;
        if (use_multi2) {
            grid_x = (N + 15) / 16;  // 16 outputs per threadgroup in multi2
        } else {
            grid_x = (N + 7) / 8;    // 8 outputs per threadgroup in standard
        }

        MTLSize grid_size = MTLSizeMake(
            grid_x,
            M > 1 ? M : 1,
            batch_size
        );
        MTLSize threadgroup_size = MTLSizeMake(256, 1, 1);

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        cmd_pipeline.commit_and_continue();

        return result;
    }
}

MNTensor matmul_simd(const MNTensor& a, const MNTensor& b,
                     bool transpose_a, bool transpose_b,
                     int64_t M, int64_t N, int64_t K, MNDevice& device) {
    @autoreleasepool {
        // Compute output shape
        std::vector<int64_t> output_dims;
        for (size_t i = 0; i < a.ndim() - 2; ++i) {
            output_dims.push_back(a.shape()[static_cast<int64_t>(i)]);
        }
        output_dims.push_back(M);
        output_dims.push_back(N);
        MNShape output_shape(output_dims);

        // Allocate output tensor (GPU-only when preferred)
        StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
        MNTensor result = MNTensor::empty(output_shape, a.dtype(), device, out_mode);

        // Select kernel based on dtype and batching
        const char* kernel_name = nullptr;
        int64_t batch_size = 1;
        for (size_t i = 0; i < a.ndim() - 2; ++i) {
            batch_size *= a.shape()[static_cast<int64_t>(i)];
        }

        if (batch_size > 1) {
            if (a.dtype() == MNDType::Float16) {
                kernel_name = "matmul_simd_batched_fp16";
            } else if (a.dtype() == MNDType::BFloat16) {
                kernel_name = "matmul_simd_batched_bf16";
            } else {
                kernel_name = "matmul_simd_batched_fp32";
            }
        } else {
            if (a.dtype() == MNDType::Float16) {
                kernel_name = "matmul_simd_fp16";
            } else if (a.dtype() == MNDType::BFloat16) {
                kernel_name = "matmul_simd_bf16";
            } else {
                kernel_name = "matmul_simd_fp32";
            }
        }

        KernelRegistry& registry = KernelRegistry::instance();
        id<MTLComputePipelineState> pipeline = registry.get_pipeline(kernel_name);

        // Setup command encoding
        CommandPipeline& cmd_pipeline = device.command_pipeline();
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Bind buffers
        [encoder setBuffer:a.buffer()->metal_buffer() offset:0 atIndex:0];
        [encoder setBuffer:b.buffer()->metal_buffer() offset:0 atIndex:1];
        [encoder setBuffer:result.buffer()->metal_buffer() offset:0 atIndex:2];

        // Set matrix dimensions
        uint32_t M_u32 = static_cast<uint32_t>(M);
        uint32_t N_u32 = static_cast<uint32_t>(N);
        uint32_t K_u32 = static_cast<uint32_t>(K);
        [encoder setBytes:&M_u32 length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&N_u32 length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&K_u32 length:sizeof(uint32_t) atIndex:5];

        // Set leading dimensions (row-major, so lda = cols of A)
        uint32_t lda = static_cast<uint32_t>(a.shape()[a.ndim() - 1]);
        uint32_t ldb = static_cast<uint32_t>(b.shape()[b.ndim() - 1]);
        uint32_t ldc = static_cast<uint32_t>(N);
        [encoder setBytes:&lda length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&ldb length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&ldc length:sizeof(uint32_t) atIndex:8];

        // Set transpose flags
        [encoder setBytes:&transpose_a length:sizeof(bool) atIndex:9];
        [encoder setBytes:&transpose_b length:sizeof(bool) atIndex:10];

        // For batched kernels, set batch strides
        if (batch_size > 1) {
            uint32_t batch_stride_a = static_cast<uint32_t>(
                a.shape()[a.ndim() - 2] * a.shape()[a.ndim() - 1]);
            uint32_t batch_stride_b = static_cast<uint32_t>(
                b.shape()[b.ndim() - 2] * b.shape()[b.ndim() - 1]);
            uint32_t batch_stride_c = static_cast<uint32_t>(M * N);
            [encoder setBytes:&batch_stride_a length:sizeof(uint32_t) atIndex:11];
            [encoder setBytes:&batch_stride_b length:sizeof(uint32_t) atIndex:12];
            [encoder setBytes:&batch_stride_c length:sizeof(uint32_t) atIndex:13];
        }

        // Set threadgroup memory sizes
        // shared_A: TILE_M * TILE_K * element_size
        // shared_B: TILE_K * TILE_N * element_size
        // BF16 uses FP32 threadgroup memory (converted on load)
        constexpr uint32_t TILE_M = 32;
        constexpr uint32_t TILE_K = 32;
        constexpr uint32_t TILE_N = 32;
        size_t element_size = (a.dtype() == MNDType::Float16) ? 2 : 4;
        [encoder setThreadgroupMemoryLength:(TILE_M * TILE_K * element_size) atIndex:0];
        [encoder setThreadgroupMemoryLength:(TILE_K * TILE_N * element_size) atIndex:1];

        // Dispatch threadgroups
        // Grid: (ceil(N/32), ceil(M/32), batch_size)
        // Threadgroup: (128, 1, 1) = 4 SIMD groups of 32 threads
        MTLSize grid_size = MTLSizeMake(
            (N + TILE_N - 1) / TILE_N,
            (M + TILE_M - 1) / TILE_M,
            batch_size
        );
        MTLSize threadgroup_size = MTLSizeMake(128, 1, 1);

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        cmd_pipeline.commit_and_continue();

        return result;
    }
}

MNTensor matmul_tiled64(const MNTensor& a, const MNTensor& b,
                        bool transpose_a, bool transpose_b,
                        int64_t M, int64_t N, int64_t K, MNDevice& device) {
    @autoreleasepool {
        // Compute output shape
        std::vector<int64_t> output_dims;
        for (size_t i = 0; i < a.ndim() - 2; ++i) {
            output_dims.push_back(a.shape()[static_cast<int64_t>(i)]);
        }
        output_dims.push_back(M);
        output_dims.push_back(N);
        MNShape output_shape(output_dims);

        // Allocate output tensor (GPU-only when preferred)
        StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
        MNTensor result = MNTensor::empty(output_shape, a.dtype(), device, out_mode);

        // Select kernel based on dtype and batching
        // Note: Only fp16 and fp32 non-batched, and fp16 batched are currently available
        const char* kernel_name = nullptr;
        int64_t batch_size = 1;
        for (size_t i = 0; i < a.ndim() - 2; ++i) {
            batch_size *= a.shape()[static_cast<int64_t>(i)];
        }

        if (batch_size > 1) {
            kernel_name = (a.dtype() == MNDType::Float16) ?
                "matmul_tiled64_batched_fp16" : "matmul_tiled64_fp32";
        } else {
            if (a.dtype() == MNDType::Float16) {
                kernel_name = "matmul_tiled64_fp16";
            } else if (a.dtype() == MNDType::BFloat16) {
                kernel_name = "matmul_tiled64_bf16";
            } else {
                kernel_name = "matmul_tiled64_fp32";
            }
        }

        KernelRegistry& registry = KernelRegistry::instance();
        id<MTLComputePipelineState> pipeline = registry.get_pipeline(kernel_name);

        // Setup command encoding
        CommandPipeline& cmd_pipeline = device.command_pipeline();
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Bind buffers
        [encoder setBuffer:a.buffer()->metal_buffer() offset:0 atIndex:0];
        [encoder setBuffer:b.buffer()->metal_buffer() offset:0 atIndex:1];
        [encoder setBuffer:result.buffer()->metal_buffer() offset:0 atIndex:2];

        // Set matrix dimensions
        uint32_t M_u32 = static_cast<uint32_t>(M);
        uint32_t N_u32 = static_cast<uint32_t>(N);
        uint32_t K_u32 = static_cast<uint32_t>(K);
        [encoder setBytes:&M_u32 length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&N_u32 length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&K_u32 length:sizeof(uint32_t) atIndex:5];

        // Set leading dimensions (row-major, so lda = cols of A)
        uint32_t lda = static_cast<uint32_t>(a.shape()[a.ndim() - 1]);
        uint32_t ldb = static_cast<uint32_t>(b.shape()[b.ndim() - 1]);
        uint32_t ldc = static_cast<uint32_t>(N);
        [encoder setBytes:&lda length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&ldb length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&ldc length:sizeof(uint32_t) atIndex:8];

        // Set transpose flags
        [encoder setBytes:&transpose_a length:sizeof(bool) atIndex:9];
        [encoder setBytes:&transpose_b length:sizeof(bool) atIndex:10];

        // For batched kernels, set batch strides
        if (batch_size > 1) {
            uint32_t batch_stride_a = static_cast<uint32_t>(
                a.shape()[a.ndim() - 2] * a.shape()[a.ndim() - 1]);
            uint32_t batch_stride_b = static_cast<uint32_t>(
                b.shape()[b.ndim() - 2] * b.shape()[b.ndim() - 1]);
            uint32_t batch_stride_c = static_cast<uint32_t>(M * N);
            [encoder setBytes:&batch_stride_a length:sizeof(uint32_t) atIndex:11];
            [encoder setBytes:&batch_stride_b length:sizeof(uint32_t) atIndex:12];
            [encoder setBytes:&batch_stride_c length:sizeof(uint32_t) atIndex:13];
        }

        // Set threadgroup memory sizes (double-buffered)
        // shared_A: 2 * TILE_M * TILE_K * element_size
        // shared_B: 2 * TILE_K * TILE_N * element_size
        // BF16 uses FP32 threadgroup memory (converted on load)
        constexpr uint32_t TILE_M = 64;
        constexpr uint32_t TILE_K = 32;
        constexpr uint32_t TILE_N = 64;
        size_t element_size = (a.dtype() == MNDType::Float16) ? 2 : 4;
        // Double-buffered shared memory
        [encoder setThreadgroupMemoryLength:(2 * TILE_M * TILE_K * element_size) atIndex:0];
        [encoder setThreadgroupMemoryLength:(2 * TILE_K * TILE_N * element_size) atIndex:1];

        // Dispatch threadgroups
        // Grid: (ceil(N/64), ceil(M/64), batch_size)
        // Threadgroup: (256, 1, 1) = 8 SIMD groups of 32 threads
        MTLSize grid_size = MTLSizeMake(
            (N + TILE_N - 1) / TILE_N,
            (M + TILE_M - 1) / TILE_M,
            batch_size
        );
        MTLSize threadgroup_size = MTLSizeMake(256, 1, 1);

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        cmd_pipeline.commit_and_continue();

        return result;
    }
}

MNTensor matmul_cpu(const MNTensor& a, const MNTensor& b,
                    bool transpose_a, bool transpose_b, MNDevice& device) {
    // Only FP32 for CPU path (fall through for FP16/BF16)
    MN_CHECK(a.dtype() == MNDType::Float32,
             MetalNativeError::InvalidArgument,
             "CPU matmul fast-path only supports Float32");

    int64_t M = transpose_a ? a.shape()[a.ndim() - 1] : a.shape()[a.ndim() - 2];
    int64_t K = transpose_a ? a.shape()[a.ndim() - 2] : a.shape()[a.ndim() - 1];
    int64_t N = transpose_b ? b.shape()[b.ndim() - 2] : b.shape()[b.ndim() - 1];

    // Compute output shape
    std::vector<int64_t> output_dims;
    for (size_t i = 0; i < a.ndim() - 2; ++i) {
        output_dims.push_back(a.shape()[static_cast<int64_t>(i)]);
    }
    output_dims.push_back(M);
    output_dims.push_back(N);
    MNShape output_shape(output_dims);

    // Allocate output (Shared mode for CPU access)
    MNTensor result = MNTensor::empty(output_shape, MNDType::Float32, device,
                                       StorageMode::Shared);

    // For 2D matrices, use cblas_sgemm directly
    // For batched, loop over batch dims
    int64_t batch_size = 1;
    for (size_t i = 0; i < a.ndim() - 2; ++i) {
        batch_size *= a.shape()[static_cast<int64_t>(i)];
    }

    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* c_ptr = result.data_ptr<float>();

    int64_t lda = a.shape()[a.ndim() - 1];
    int64_t ldb = b.shape()[b.ndim() - 1];
    int64_t ldc = N;

    int64_t a_batch_stride = M * K;
    if (transpose_a) a_batch_stride = K * a.shape()[a.ndim() - 1];
    else a_batch_stride = a.shape()[a.ndim() - 2] * a.shape()[a.ndim() - 1];
    int64_t b_batch_stride = b.shape()[b.ndim() - 2] * b.shape()[b.ndim() - 1];
    int64_t c_batch_stride = M * N;

    CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        cblas_sgemm(CblasRowMajor,
                    trans_a, trans_b,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    1.0f, // alpha
                    a_ptr + batch * a_batch_stride, static_cast<int>(lda),
                    b_ptr + batch * b_batch_stride, static_cast<int>(ldb),
                    0.0f, // beta
                    c_ptr + batch * c_batch_stride, static_cast<int>(ldc));
    }

    return result;
}

} // anonymous namespace

MNTensor matmul(const MNTensor& a,
                const MNTensor& b,
                bool transpose_a,
                bool transpose_b) {
    MN_CHECK(a.ndim() >= 2,
             MetalNativeError::InvalidArgument,
             "matmul: input 'a' must have at least 2 dimensions");
    MN_CHECK(b.ndim() >= 2,
             MetalNativeError::InvalidArgument,
             "matmul: input 'b' must have at least 2 dimensions");
    MN_CHECK(a.dtype() == b.dtype(),
             MetalNativeError::InvalidArgument,
             "matmul: input dtypes must match");

    int64_t M = transpose_a ? a.shape()[a.ndim() - 1] : a.shape()[a.ndim() - 2];
    int64_t K_a = transpose_a ? a.shape()[a.ndim() - 2] : a.shape()[a.ndim() - 1];
    int64_t K_b = transpose_b ? b.shape()[b.ndim() - 1] : b.shape()[b.ndim() - 2];
    int64_t N = transpose_b ? b.shape()[b.ndim() - 2] : b.shape()[b.ndim() - 1];

    MN_CHECK(K_a == K_b,
             MetalNativeError::InvalidArgument,
             "matmul: incompatible dimensions for matrix multiplication");

    int64_t K = K_a; // K_a == K_b already validated

    // CPU fast-path for small FP32 tensors (GPU dispatch overhead > compute time)
    // Accelerate/CBLAS is 5-48x faster for matrices ≤ 512x512
    if (should_use_cpu_path(a, b, M, N, K)) {
        MNDevice& device = MNDevice::instance();
        return matmul_cpu(a, b, transpose_a, transpose_b, device);
    }

    // Adaptive kernel dispatch based on matrix dimensions and contiguity
    bool is_contiguous = a.is_contiguous() && b.is_contiguous();
    MatmulKernelType kernel_type = select_kernel(M, N, K, a.dtype(), is_contiguous);

    MNDevice& device = MNDevice::instance();

    switch (kernel_type) {
        case MatmulKernelType::VecMat:
            // Decode phase: vector-matrix kernel (M=1..8)
            // Note: vecmat only supports fp16/fp32, not bf16 yet
            if (a.dtype() == MNDType::Float16 || a.dtype() == MNDType::Float32) {
                return matmul_vecmat(a, b, transpose_a, transpose_b, M, N, K, device);
            }
            // Fall through to SIMD32 for BF16
            [[fallthrough]];

        case MatmulKernelType::SIMD32:
            // Small matrices: 32x32 tiled kernel (supports fp16/fp32/bf16)
            return matmul_simd(a, b, transpose_a, transpose_b, M, N, K, device);

        case MatmulKernelType::Tiled64:
            // Medium matrices: 64x64 tiled kernel (supports fp16/fp32/bf16)
            return matmul_tiled64(a, b, transpose_a, transpose_b, M, N, K, device);

        case MatmulKernelType::MPSGraph:
            // Fall through to MPSGraph for large matrices or non-contiguous tensors
            break;

        case MatmulKernelType::CPU:
            // Should not reach here (handled above)
            break;
    }

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();

        // Check cache first
        GraphCacheKey cache_key = make_matmul_key(a.shape(), b.shape(), a.dtype(), transpose_a, transpose_b);

        MPSGraph* graph = nil;
        MPSGraphTensor* a_tensor = nil;
        MPSGraphTensor* b_tensor = nil;
        MPSGraphTensor* result_tensor = nil;

        {
            std::lock_guard<std::mutex> lock(matmul_cache_mu);
            auto it = matmul_graph_cache.find(cache_key);
            if (it != matmul_graph_cache.end()) {
                // Cache hit: reuse the cached graph
                graph = it->second.graph;
                a_tensor = it->second.a_placeholder;
                b_tensor = it->second.b_placeholder;
                result_tensor = it->second.result_tensor;
                it->second.last_access = monotonic_tick();
            }
        }

        // Cache miss: build the graph
        if (graph == nil) {
            graph = [[MPSGraph alloc] init];

            // Create placeholder tensors
            a_tensor = [graph placeholderWithShape:to_ns_shape(a.shape())
                                           dataType:to_mps_datatype(a.dtype())
                                               name:@"a"];
            b_tensor = [graph placeholderWithShape:to_ns_shape(b.shape())
                                           dataType:to_mps_datatype(b.dtype())
                                               name:@"b"];

            // Apply transpose if needed
            MPSGraphTensor* a_transposed = a_tensor;
            MPSGraphTensor* b_transposed = b_tensor;

            if (transpose_a) {
                NSInteger ndim = a.ndim();
                NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:ndim];
                for (NSInteger i = 0; i < ndim - 2; ++i) {
                    [perm addObject:@(i)];
                }
                [perm addObject:@(ndim - 1)];  // swap last two dims
                [perm addObject:@(ndim - 2)];
                a_transposed = [graph transposeTensor:a_tensor permutation:perm name:nil];
            }

            if (transpose_b) {
                NSInteger ndim = b.ndim();
                NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:ndim];
                for (NSInteger i = 0; i < ndim - 2; ++i) {
                    [perm addObject:@(i)];
                }
                [perm addObject:@(ndim - 1)];  // swap last two dims
                [perm addObject:@(ndim - 2)];
                b_transposed = [graph transposeTensor:b_tensor permutation:perm name:nil];
            }

            // Matrix multiplication
            result_tensor = [graph matrixMultiplicationWithPrimaryTensor:a_transposed
                                                         secondaryTensor:b_transposed
                                                                    name:@"matmul"];

            MN_CHECK(result_tensor != nil,
                     MetalNativeError::InternalError,
                     "matmul: MPSGraph operation failed");

            // Update cache capacity from budget controller
            auto& bc = MemoryBudgetController::instance();
            float mult = bc.pressure_multiplier();
            matmul_cache_max_entries = std::max<size_t>(4, static_cast<size_t>(kDefaultMaxCacheEntries * mult));

            // Store in cache
            std::lock_guard<std::mutex> lock(matmul_cache_mu);
            evict_lru_if_needed();
            MatmulCacheEntry entry;
            entry.graph = graph;
            entry.a_placeholder = a_tensor;
            entry.b_placeholder = b_tensor;
            entry.result_tensor = result_tensor;
            entry.last_access = monotonic_tick();
            matmul_graph_cache[cache_key] = entry;
        }

        // Create MPSGraphTensorData for inputs (with shared cache)
        MPSDataType mps_dtype = to_mps_datatype(a.dtype());
        uint64_t a_hash = a.shape().ndim();
        for (size_t i = 0; i < a.shape().ndim(); ++i) {
            a_hash = a_hash * 31 + static_cast<uint64_t>(a.shape()[static_cast<int64_t>(i)]);
        }

        MPSGraphTensorData* a_data;
        {
            std::shared_lock<std::shared_mutex> rlock(td_cache_mu);
            if (td_cache_a.buffer == a.buffer()->metal_buffer() &&
                td_cache_a.shape_hash == a_hash &&
                td_cache_a.dtype == mps_dtype) {
                a_data = td_cache_a.data;
            } else {
                rlock.unlock();
                a_data = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:a.buffer()->metal_buffer()
                               shape:cached_ns_shape(a.shape())
                            dataType:mps_dtype];
                std::unique_lock<std::shared_mutex> wlock(td_cache_mu);
                td_cache_a.buffer = a.buffer()->metal_buffer();
                td_cache_a.shape_hash = a_hash;
                td_cache_a.dtype = mps_dtype;
                td_cache_a.data = a_data;
            }
        }

        uint64_t b_hash = b.shape().ndim();
        for (size_t i = 0; i < b.shape().ndim(); ++i) {
            b_hash = b_hash * 31 + static_cast<uint64_t>(b.shape()[static_cast<int64_t>(i)]);
        }

        MPSGraphTensorData* b_data;
        {
            std::shared_lock<std::shared_mutex> rlock(td_cache_mu);
            if (td_cache_b.buffer == b.buffer()->metal_buffer() &&
                td_cache_b.shape_hash == b_hash &&
                td_cache_b.dtype == mps_dtype) {
                b_data = td_cache_b.data;
            } else {
                rlock.unlock();
                b_data = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:b.buffer()->metal_buffer()
                               shape:cached_ns_shape(b.shape())
                            dataType:mps_dtype];
                std::unique_lock<std::shared_mutex> wlock(td_cache_mu);
                td_cache_b.buffer = b.buffer()->metal_buffer();
                td_cache_b.shape_hash = b_hash;
                td_cache_b.dtype = mps_dtype;
                td_cache_b.data = b_data;
            }
        }

        // Determine output shape
        std::vector<int64_t> output_dims;
        for (size_t i = 0; i < a.ndim() - 2; ++i) {
            output_dims.push_back(a.shape()[static_cast<int64_t>(i)]);
        }
        output_dims.push_back(M);
        output_dims.push_back(N);
        MNShape output_shape(output_dims);

        // Allocate output tensor (GPU-only when preferred)
        StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
        MNTensor result = MNTensor::empty(output_shape, a.dtype(), device, out_mode);

        // Create result tensor data backed by our output buffer so MPSGraph
        // writes directly into it (zero-copy output).
        uint64_t result_hash = output_shape.ndim();
        for (size_t i = 0; i < output_shape.ndim(); ++i) {
            result_hash = result_hash * 31 + static_cast<uint64_t>(output_shape[static_cast<int64_t>(i)]);
        }

        MPSGraphTensorData* result_data;
        {
            std::shared_lock<std::shared_mutex> rlock(td_cache_mu);
            if (td_cache_result.buffer == result.buffer()->metal_buffer() &&
                td_cache_result.shape_hash == result_hash &&
                td_cache_result.dtype == mps_dtype) {
                result_data = td_cache_result.data;
            } else {
                rlock.unlock();
                result_data = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:result.buffer()->metal_buffer()
                               shape:cached_ns_shape(output_shape)
                            dataType:mps_dtype];
                std::unique_lock<std::shared_mutex> wlock(td_cache_mu);
                td_cache_result.buffer = result.buffer()->metal_buffer();
                td_cache_result.shape_hash = result_hash;
                td_cache_result.dtype = mps_dtype;
                td_cache_result.data = result_data;
            }
        }

        NSDictionary* feeds = @{
            a_tensor: a_data,
            b_tensor: b_data
        };
        NSDictionary* results_dict = @{
            result_tensor: result_data
        };

        @try {
            // Encode the graph asynchronously into the command pipeline's buffer.
            // Using MPSCommandBuffer + resultsDictionary so MPSGraph writes
            // directly into our pre-allocated output buffer (zero-copy).
            CommandPipeline& cmd_pipeline = device.command_pipeline();
            id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
            MPSCommandBuffer* mps_cmd_buf =
                [MPSCommandBuffer commandBufferWithCommandBuffer:cmd_buffer];

            [graph encodeToCommandBuffer:mps_cmd_buf
                                   feeds:feeds
                        targetOperations:nil
                       resultsDictionary:results_dict
                     executionDescriptor:nil];

            // MPSGraph's encodeToCommandBuffer only encodes GPU commands; it does
            // NOT commit the underlying MTLCommandBuffer. Safe to use the normal
            // commit path, which respects lazy-commit batching.
            cmd_pipeline.commit_and_continue();
        } @catch (NSException* exception) {
            MN_THROW(MetalNativeError::InternalError,
                     "matmul: MPSGraph execution failed: " +
                     std::string([[exception reason] UTF8String]));
        }

        return result;
    }
}

MNTensor batched_matmul(const MNTensor& a,
                        const MNTensor& b,
                        bool transpose_a,
                        bool transpose_b) {
    MN_CHECK(a.ndim() >= 3,
             MetalNativeError::InvalidArgument,
             "batched_matmul: input 'a' must have at least 3 dimensions");
    MN_CHECK(b.ndim() >= 3,
             MetalNativeError::InvalidArgument,
             "batched_matmul: input 'b' must have at least 3 dimensions");
    MN_CHECK(a.dtype() == b.dtype(),
             MetalNativeError::InvalidArgument,
             "batched_matmul: input dtypes must match");

    // For batched matmul, the regular matmul with MPSGraph handles batching automatically
    // via broadcasting of the batch dimensions
    return matmul(a, b, transpose_a, transpose_b);
}

MNTensor dequant_matmul(const MNTensor& activations,
                         const MNTensor& weights_packed,
                         const MNTensor& scales,
                         const MNTensor& zeros,
                         uint32_t group_size,
                         QuantType quant_type) {
    // Validate inputs
    MN_CHECK(activations.ndim() == 2,
             MetalNativeError::InvalidArgument,
             "dequant_matmul: activations must be 2D [M, K]");
    MN_CHECK(activations.dtype() == MNDType::Float16,
             MetalNativeError::InvalidArgument,
             "dequant_matmul: activations must be Float16");
    MN_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
             MetalNativeError::InvalidArgument,
             "dequant_matmul: group_size must be 32, 64, or 128");

    const int64_t M = activations.shape()[0];
    const int64_t K = activations.shape()[1];
    const int64_t N = (quant_type == QuantType::INT4) ?
        weights_packed.shape()[0] : weights_packed.shape()[0];

    // Validate weight dimensions
    if (quant_type == QuantType::INT4) {
        MN_CHECK(weights_packed.shape()[1] == K / 2,
                 MetalNativeError::InvalidArgument,
                 "dequant_matmul: INT4 weights must be [N, K/2]");
    } else {
        MN_CHECK(weights_packed.shape()[1] == K,
                 MetalNativeError::InvalidArgument,
                 "dequant_matmul: INT8 weights must be [N, K]");
    }

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();

        // Allocate output
        MNShape output_shape({M, N});
        StorageMode out_mode = device.prefer_private_storage() ? StorageMode::Private : StorageMode::Shared;
        MNTensor result = MNTensor::empty(output_shape, MNDType::Float16, device, out_mode);

        // Select kernel based on quantization type and M dimension
        const char* kernel_name = nullptr;
        bool use_vecmat = (M <= 8);

        if (quant_type == QuantType::INT4) {
            kernel_name = use_vecmat ?
                "dequant_matmul_int4_vecmat_fp16" :
                "dequant_matmul_int4_fp16";
        } else {
            kernel_name = use_vecmat ?
                "dequant_matmul_int8_vecmat_fp16" :
                "dequant_matmul_int8_fp16";
        }

        KernelRegistry& registry = KernelRegistry::instance();
        id<MTLComputePipelineState> pipeline = registry.get_pipeline(kernel_name);

        CommandPipeline& cmd_pipeline = device.command_pipeline();
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Bind buffers (matching matmul_dequant_kernel.metal layout)
        [encoder setBuffer:activations.buffer()->metal_buffer() offset:activations.offset() atIndex:0];
        [encoder setBuffer:weights_packed.buffer()->metal_buffer() offset:weights_packed.offset() atIndex:1];
        [encoder setBuffer:result.buffer()->metal_buffer() offset:result.offset() atIndex:2];
        [encoder setBuffer:scales.buffer()->metal_buffer() offset:scales.offset() atIndex:3];
        [encoder setBuffer:zeros.buffer()->metal_buffer() offset:zeros.offset() atIndex:4];

        // Set dimension parameters
        uint32_t M_u32 = static_cast<uint32_t>(M);
        uint32_t N_u32 = static_cast<uint32_t>(N);
        uint32_t K_u32 = static_cast<uint32_t>(K);
        uint32_t lda = static_cast<uint32_t>(K);
        uint32_t ldc = static_cast<uint32_t>(N);

        if (use_vecmat) {
            // Vecmat kernel signature: N, K, group_size (no M, no lda/ldc)
            [encoder setBytes:&N_u32 length:sizeof(uint32_t) atIndex:5];
            [encoder setBytes:&K_u32 length:sizeof(uint32_t) atIndex:6];
            [encoder setBytes:&group_size length:sizeof(uint32_t) atIndex:7];
        } else {
            // Tiled matmul kernel signature: M, N, K, group_size, lda, ldc
            [encoder setBytes:&M_u32 length:sizeof(uint32_t) atIndex:5];
            [encoder setBytes:&N_u32 length:sizeof(uint32_t) atIndex:6];
            [encoder setBytes:&K_u32 length:sizeof(uint32_t) atIndex:7];
            [encoder setBytes:&group_size length:sizeof(uint32_t) atIndex:8];
            [encoder setBytes:&lda length:sizeof(uint32_t) atIndex:9];
            [encoder setBytes:&ldc length:sizeof(uint32_t) atIndex:10];
        }

        constexpr uint32_t TILE_M = 64;
        constexpr uint32_t TILE_N = 64;
        constexpr uint32_t TILE_K = 32;
        size_t element_size = sizeof(uint16_t);  // FP16

        if (use_vecmat) {
            // Vecmat dispatch: 128 threads (4 SIMD groups), grid (ceil(N/64), M, 1)
            // Vecmat uses shared_A [TILE_K] and shared_B [TILE_N, TILE_K]
            [encoder setThreadgroupMemoryLength:(TILE_K * element_size) atIndex:0];
            [encoder setThreadgroupMemoryLength:(TILE_N * TILE_K * element_size) atIndex:1];

            MTLSize grid_size = MTLSizeMake((N + TILE_N - 1) / TILE_N, M, 1);
            MTLSize threadgroup_size = MTLSizeMake(128, 1, 1);
            [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];
        } else {
            // Tiled matmul dispatch: 128 threads (4 SIMD groups), grid (ceil(N/64), ceil(M/64), 1)
            [encoder setThreadgroupMemoryLength:(TILE_M * TILE_K * element_size) atIndex:0];
            [encoder setThreadgroupMemoryLength:(TILE_K * TILE_N * element_size) atIndex:1];

            MTLSize grid_size = MTLSizeMake(
                (N + TILE_N - 1) / TILE_N,
                (M + TILE_M - 1) / TILE_M,
                1
            );
            MTLSize threadgroup_size = MTLSizeMake(128, 1, 1);
            [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];
        }

        [encoder endEncoding];
        cmd_pipeline.commit_and_continue();

        return result;
    }
}

} // namespace metal_native
