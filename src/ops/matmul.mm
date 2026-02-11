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

#include <mutex>
#include <unordered_map>
#include <mach/mach_time.h>

namespace metal_native {

namespace {

constexpr int64_t kSmallTensorThreshold = 65536; // elements

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
size_t matmul_cache_max_entries = kDefaultMaxCacheEntries;

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

bool should_use_cpu_path(const MNTensor& a, const MNTensor& b) {
    // Use CPU for small tensors where GPU dispatch overhead dominates
    return (a.numel() < kSmallTensorThreshold && b.numel() < kSmallTensorThreshold);
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

    // CPU fast-path for small tensors (GPU dispatch overhead > compute time)
    if (should_use_cpu_path(a, b) && a.dtype() == MNDType::Float32 &&
        a.is_contiguous() && b.is_contiguous()) {
        MNDevice& device = MNDevice::instance();
        return matmul_cpu(a, b, transpose_a, transpose_b, device);
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

        // Create MPSGraphTensorData for inputs
        MPSGraphTensorData* a_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:a.buffer()->metal_buffer()
                       shape:to_ns_shape(a.shape())
                    dataType:to_mps_datatype(a.dtype())];

        MPSGraphTensorData* b_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:b.buffer()->metal_buffer()
                       shape:to_ns_shape(b.shape())
                    dataType:to_mps_datatype(b.dtype())];

        // Determine output shape
        std::vector<int64_t> output_dims;
        for (size_t i = 0; i < a.ndim() - 2; ++i) {
            output_dims.push_back(a.shape()[static_cast<int64_t>(i)]);
        }
        output_dims.push_back(M);
        output_dims.push_back(N);
        MNShape output_shape(output_dims);

        // Allocate output tensor
        MNTensor result = MNTensor::empty(output_shape, a.dtype(), device);

        // Create result tensor data backed by our output buffer so MPSGraph
        // writes directly into it (zero-copy output).
        MPSGraphTensorData* result_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:result.buffer()->metal_buffer()
                       shape:to_ns_shape(output_shape)
                    dataType:to_mps_datatype(a.dtype())];

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

            // MPSCommandBuffer may internally commit the underlying buffer,
            // so always flush here regardless of lazy commit mode.
            cmd_pipeline.flush();
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

} // namespace metal_native
