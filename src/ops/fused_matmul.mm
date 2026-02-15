/// @file fused_matmul.mm
/// @brief Objective-C++ implementation of fused matmul with epilogue operations.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/ops/fused_matmul.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/dispatch/command_pipeline.h"
#include "metal_native/kernels/kernel_registry.h"

#include <unordered_map>
#include <mutex>

namespace metal_native {

namespace {

// Tile configuration (matches matmul_fused_epilogue.metal)
constexpr uint32_t TILE_M = 64;
constexpr uint32_t TILE_N = 64;
constexpr uint32_t TILE_K = 32;

// Cache for specialized pipeline states
struct FusedPipelineCacheKey {
    MNDType dtype;
    MatmulEpilogue epilogue;
    bool is_vecmat;  // true for M=1 decode path

    bool operator==(const FusedPipelineCacheKey& other) const {
        return dtype == other.dtype &&
               epilogue == other.epilogue &&
               is_vecmat == other.is_vecmat;
    }
};

struct FusedPipelineCacheKeyHash {
    size_t operator()(const FusedPipelineCacheKey& key) const {
        size_t h1 = std::hash<int>{}(static_cast<int>(key.dtype));
        size_t h2 = std::hash<int>{}(static_cast<int>(key.epilogue));
        size_t h3 = std::hash<bool>{}(key.is_vecmat);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

std::mutex pipeline_cache_mu;
std::unordered_map<FusedPipelineCacheKey, id<MTLComputePipelineState>,
                   FusedPipelineCacheKeyHash> fused_pipeline_cache;

// Get or create specialized pipeline state with function constants
id<MTLComputePipelineState> get_fused_pipeline(
    MNDType dtype,
    MatmulEpilogue epilogue,
    bool is_vecmat,
    id<MTLLibrary> library,
    id<MTLDevice> device
) {
    FusedPipelineCacheKey key{dtype, epilogue, is_vecmat};

    {
        std::lock_guard<std::mutex> lock(pipeline_cache_mu);
        auto it = fused_pipeline_cache.find(key);
        if (it != fused_pipeline_cache.end()) {
            return it->second;
        }
    }

    // Cache miss: create specialized pipeline
    @autoreleasepool {
        // Set function constants for epilogue type
        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        uint epilogue_type = static_cast<uint>(epilogue);
        [constants setConstantValue:&epilogue_type
                               type:MTLDataTypeUInt
                            atIndex:2];

        // Get specialized function
        NSError* error = nil;
        NSString* kernel_name;
        if (is_vecmat) {
            if (dtype == MNDType::Float16) {
                kernel_name = @"matmul_vecmat_fused_epilogue_fp16";
            } else {
                kernel_name = @"matmul_vecmat_fused_epilogue_fp32";
            }
        } else {
            if (dtype == MNDType::Float16) {
                kernel_name = @"matmul_fused_epilogue_fp16";
            } else {
                kernel_name = @"matmul_fused_epilogue_fp32";
            }
        }

        id<MTLFunction> function = [library newFunctionWithName:kernel_name
                                                  constantValues:constants
                                                           error:&error];
        MN_CHECK(function != nil,
                 MetalNativeError::InternalError,
                 std::string("Failed to create function '") +
                 [kernel_name UTF8String] + "': " +
                 (error ? [[error localizedDescription] UTF8String] : "unknown error"));

        // Create pipeline state
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function
                                                   error:&error];
        MN_CHECK(pipeline != nil,
                 MetalNativeError::InternalError,
                 std::string("Failed to create pipeline: ") +
                 (error ? [[error localizedDescription] UTF8String] : "unknown error"));

        // Cache it
        std::lock_guard<std::mutex> lock(pipeline_cache_mu);
        fused_pipeline_cache[key] = pipeline;
        return pipeline;
    }
}

bool is_vecmat_case(int64_t M) {
    // Use vecmat kernel for M=1 (decode phase)
    return M == 1;
}

} // anonymous namespace

MNTensor fused_matmul(const MNTensor& a,
                      const MNTensor& b,
                      MatmulEpilogue epilogue,
                      const MNTensor* bias,
                      const MNTensor* residual,
                      bool transpose_a,
                      bool transpose_b) {
    @autoreleasepool {
        // Validation
        MN_CHECK(a.ndim() >= 2,
                 MetalNativeError::InvalidArgument,
                 "fused_matmul: input 'a' must have at least 2 dimensions");
        MN_CHECK(b.ndim() >= 2,
                 MetalNativeError::InvalidArgument,
                 "fused_matmul: input 'b' must have at least 2 dimensions");
        MN_CHECK(a.dtype() == b.dtype(),
                 MetalNativeError::InvalidArgument,
                 "fused_matmul: input dtypes must match");

        // Only FP16 and FP32 supported for fused kernels
        MN_CHECK(a.dtype() == MNDType::Float16 || a.dtype() == MNDType::Float32,
                 MetalNativeError::InvalidArgument,
                 "fused_matmul: only Float16 and Float32 supported");

        int64_t M = transpose_a ? a.shape()[a.ndim() - 1] : a.shape()[a.ndim() - 2];
        int64_t K_a = transpose_a ? a.shape()[a.ndim() - 2] : a.shape()[a.ndim() - 1];
        int64_t K_b = transpose_b ? b.shape()[b.ndim() - 1] : b.shape()[b.ndim() - 2];
        int64_t N = transpose_b ? b.shape()[b.ndim() - 2] : b.shape()[b.ndim() - 1];

        MN_CHECK(K_a == K_b,
                 MetalNativeError::InvalidArgument,
                 "fused_matmul: incompatible dimensions for matrix multiplication");

        int64_t K = K_a;

        // Validate epilogue inputs
        if (epilogue == MatmulEpilogue::Bias ||
            epilogue == MatmulEpilogue::BiasReLU ||
            epilogue == MatmulEpilogue::BiasSiLU) {
            MN_CHECK(bias != nullptr,
                     MetalNativeError::InvalidArgument,
                     "fused_matmul: bias tensor required for Bias epilogue");
            MN_CHECK(bias->ndim() == 1,
                     MetalNativeError::InvalidArgument,
                     "fused_matmul: bias must be 1D tensor");
            MN_CHECK(bias->shape()[0] == N,
                     MetalNativeError::InvalidArgument,
                     "fused_matmul: bias shape must match output dimension N");
            MN_CHECK(bias->dtype() == a.dtype(),
                     MetalNativeError::InvalidArgument,
                     "fused_matmul: bias dtype must match input dtype");
        }

        if (epilogue == MatmulEpilogue::ResidualAdd) {
            MN_CHECK(residual != nullptr,
                     MetalNativeError::InvalidArgument,
                     "fused_matmul: residual tensor required for ResidualAdd epilogue");
            MN_CHECK(residual->ndim() >= 2,
                     MetalNativeError::InvalidArgument,
                     "fused_matmul: residual must have at least 2 dimensions");
            MN_CHECK(residual->shape()[residual->ndim() - 2] == M &&
                     residual->shape()[residual->ndim() - 1] == N,
                     MetalNativeError::InvalidArgument,
                     "fused_matmul: residual shape must match output shape [M, N]");
            MN_CHECK(residual->dtype() == a.dtype(),
                     MetalNativeError::InvalidArgument,
                     "fused_matmul: residual dtype must match input dtype");
        }

        MNDevice& device = MNDevice::instance();

        // Compute output shape
        std::vector<int64_t> output_dims;
        for (size_t i = 0; i < a.ndim() - 2; ++i) {
            output_dims.push_back(a.shape()[static_cast<int64_t>(i)]);
        }
        output_dims.push_back(M);
        output_dims.push_back(N);
        MNShape output_shape(output_dims);

        // Allocate output tensor
        StorageMode out_mode = device.prefer_private_storage() ?
                               StorageMode::Private : StorageMode::Shared;
        MNTensor result = MNTensor::empty(output_shape, a.dtype(), device, out_mode);

        // Get Metal library and device for pipeline creation
        // First trigger library load by getting any pipeline (ensures library is loaded)
        KernelRegistry::instance().get_pipeline("matmul_tiled64_fp16");

        id<MTLDevice> metal_device = device.metal_device();

        // Load the metal library containing our shaders
        NSError* lib_error = nil;
        id<MTLLibrary> library = [metal_device newDefaultLibrary];
        if (library == nil) {
            // Try loading from the compiled metallib path
#ifdef METAL_NATIVE_METALLIB_PATH
            NSString* metallib_path = @METAL_NATIVE_METALLIB_PATH;
            NSURL* url = [NSURL fileURLWithPath:metallib_path];
            library = [metal_device newLibraryWithURL:url error:&lib_error];
#endif
        }
        MN_CHECK(library != nil,
                 MetalNativeError::InternalError,
                 std::string("Failed to load Metal library: ") +
                 (lib_error ? [[lib_error localizedDescription] UTF8String] : "unknown error"));

        // Select appropriate kernel (vecmat for M=1, tiled for larger M)
        bool use_vecmat = is_vecmat_case(M);
        id<MTLComputePipelineState> pipeline =
            get_fused_pipeline(a.dtype(), epilogue, use_vecmat, library, metal_device);

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

        // Bind epilogue buffers (only if epilogue is not None)
        if (epilogue != MatmulEpilogue::None) {
            if (epilogue == MatmulEpilogue::Bias ||
                epilogue == MatmulEpilogue::BiasReLU ||
                epilogue == MatmulEpilogue::BiasSiLU) {
                [encoder setBuffer:bias->buffer()->metal_buffer() offset:0 atIndex:14];
            } else if (epilogue == MatmulEpilogue::ResidualAdd) {
                [encoder setBuffer:residual->buffer()->metal_buffer() offset:0 atIndex:15];
            }
        }

        // Set threadgroup memory (for tiled kernel only)
        if (!use_vecmat) {
            size_t element_size = (a.dtype() == MNDType::Float16) ? 2 : 4;
            // Double-buffered: 2 * TILE_M * TILE_K * element_size
            [encoder setThreadgroupMemoryLength:(2 * TILE_M * TILE_K * element_size) atIndex:0];
            // Double-buffered: 2 * TILE_K * TILE_N * element_size
            [encoder setThreadgroupMemoryLength:(2 * TILE_K * TILE_N * element_size) atIndex:1];
        }

        // Dispatch threadgroups
        MTLSize grid_size, threadgroup_size;

        if (use_vecmat) {
            // Vec-mat: grid (ceil(N/8), 1, 1), threadgroup (256, 1, 1)
            grid_size = MTLSizeMake((N + 7) / 8, 1, 1);
            threadgroup_size = MTLSizeMake(256, 1, 1);
        } else {
            // Tiled 64x64: grid (ceil(N/64), ceil(M/64), 1), threadgroup (256, 1, 1)
            grid_size = MTLSizeMake(
                (N + TILE_N - 1) / TILE_N,
                (M + TILE_M - 1) / TILE_M,
                1
            );
            threadgroup_size = MTLSizeMake(256, 1, 1);
        }

        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        cmd_pipeline.commit_and_continue();

        return result;
    }
}

} // namespace metal_native
