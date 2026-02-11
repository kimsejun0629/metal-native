#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/memory/quant_weight_cache.h"
#include "metal_native/core/tensor.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"
#include "metal_native/memory/budget_controller.h"
#include "metal_native/kernels/kernel_registry.h"
#include "metal_native/dispatch/command_pipeline.h"

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace metal_native {

struct QuantWeightCache::Impl {
    struct WeightEntry {
        MNTensor quantized;
        MNTensor scales;
        std::optional<MNTensor> dequantized;
        QuantFormat format;
        size_t num_elements;
        size_t group_size;
    };

    MNDevice& device;
    mutable std::mutex mu;
    std::unordered_map<std::string, WeightEntry> weights;

    explicit Impl(MNDevice& dev) : device(dev) {}
};

QuantWeightCache::QuantWeightCache(MNDevice& device)
    : impl_(std::make_unique<Impl>(device)) {}

QuantWeightCache::~QuantWeightCache() = default;

void QuantWeightCache::register_weight(const std::string& name, const MNTensor& quantized,
                                        const MNTensor& scales, QuantFormat format) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    // Calculate elements based on format
    size_t num_elements, group_size;
    if (format == QuantFormat::INT8) {
        num_elements = quantized.numel();
        group_size = quantized.numel() / scales.numel();
    } else { // INT4
        num_elements = quantized.numel() * 2; // 2 elements per byte
        group_size = num_elements / scales.numel();
    }

    impl_->weights.insert_or_assign(name, Impl::WeightEntry{
        quantized,
        scales,
        std::nullopt,  // no dequantized copy yet
        format,
        num_elements,
        group_size
    });
}

MNTensor QuantWeightCache::get_dequantized(const std::string& name) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    auto it = impl_->weights.find(name);
    MN_CHECK(it != impl_->weights.end(),
             MetalNativeError::InvalidArgument,
             "quant_weight_cache: weight not registered");

    auto& entry = it->second;

    // Return cached dequantized if available
    if (entry.dequantized.has_value()) {
        return entry.dequantized.value();
    }

    // Dequantize on-the-fly
    MNShape output_shape = entry.quantized.shape();
    if (entry.format == QuantFormat::INT4) {
        // INT4: output has 2x elements
        std::vector<int64_t> dims;
        for (size_t i = 0; i < output_shape.ndim(); ++i) {
            dims.push_back(output_shape[static_cast<int64_t>(i)]);
        }
        dims.back() *= 2;
        output_shape = MNShape(dims);
    }

    MNTensor dequantized_tensor = MNTensor::empty(output_shape, MNDType::Float16, impl_->device);

    @autoreleasepool {
        const char* kernel_name = (entry.format == QuantFormat::INT8)
            ? "dequantize_int8_to_fp16"
            : "dequantize_int4_to_fp16";

        id<MTLComputePipelineState> pipeline = KernelRegistry::instance().get_pipeline(kernel_name);

        CommandPipeline& cmd_pipeline = impl_->device.command_pipeline();
        id<MTLCommandBuffer> cmd_buffer = cmd_pipeline.current_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:entry.quantized.buffer()->metal_buffer() offset:entry.quantized.offset() atIndex:0];
        [encoder setBuffer:entry.scales.buffer()->metal_buffer() offset:entry.scales.offset() atIndex:1];
        [encoder setBuffer:dequantized_tensor.buffer()->metal_buffer() offset:dequantized_tensor.offset() atIndex:2];

        uint32_t num_elements = static_cast<uint32_t>(entry.num_elements);
        uint32_t group_size = static_cast<uint32_t>(entry.group_size);
        [encoder setBytes:&num_elements length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&group_size length:sizeof(uint32_t) atIndex:4];

        MTLSize grid_size = MTLSizeMake(entry.num_elements, 1, 1);
        MTLSize threadgroup_size = MTLSizeMake(
            std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];
        cmd_pipeline.commit();
    }

    // Cache if budget allows
    float mult = MemoryBudgetController::instance().pressure_multiplier();
    if (mult >= 0.5f) {
        entry.dequantized = dequantized_tensor;
    }

    return dequantized_tensor;
}

void QuantWeightCache::evict_dequantized() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    for (auto& [name, entry] : impl_->weights) {
        entry.dequantized = std::nullopt;
    }
}

size_t QuantWeightCache::dequantized_footprint() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    size_t total = 0;
    for (auto& [name, entry] : impl_->weights) {
        if (entry.dequantized.has_value()) {
            total += entry.num_elements * 2;  // FP16 = 2 bytes per element
        }
    }
    return total;
}

size_t QuantWeightCache::total_footprint() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    size_t total = 0;
    for (auto& [name, entry] : impl_->weights) {
        // Quantized storage
        if (entry.format == QuantFormat::INT8) {
            total += entry.num_elements; // 1 byte per element
        } else {
            total += entry.num_elements / 2; // 0.5 bytes per element
        }
        // Scales (FP16)
        total += entry.scales.numel() * 2;
        // Dequantized (FP16)
        if (entry.dequantized.has_value()) {
            total += entry.num_elements * 2;
        }
    }
    return total;
}

} // namespace metal_native
