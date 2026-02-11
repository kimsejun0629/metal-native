/// @file perf_counters.mm
/// @brief Objective-C++ implementation of PerfCounterReader.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/profiling/perf_counters.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

#include <sstream>
#include <chrono>

namespace metal_native {

// ---------------------------------------------------------------------------
// PerfCounterSnapshot
// ---------------------------------------------------------------------------

std::string PerfCounterSnapshot::summary() const {
    if (!is_valid) {
        return "PerfCounterSnapshot: Invalid (counters not supported)";
    }

    std::ostringstream oss;
    oss << "PerfCounterSnapshot:\n"
        << "  GPU Utilization:    " << gpu_utilization << "%\n"
        << "  ALU Utilization:    " << alu_utilization << "%\n"
        << "  Memory Bandwidth:   " << memory_bandwidth << " GB/s\n"
        << "  Occupancy:          " << occupancy << "%\n"
        << "  Timestamp:          " << timestamp_ns << " ns";
    return oss.str();
}

// ---------------------------------------------------------------------------
// PerfCounterReader::Impl
// ---------------------------------------------------------------------------

struct PerfCounterReader::Impl {
    id<MTLDevice>                   device = nil;
    NSArray<id<MTLCounterSet>>*     counter_sets = nil;
    id<MTLCounterSampleBuffer>      sample_buffer = nil;
    bool                            supported = false;

    Impl() {
        @autoreleasepool {
            MNDevice& mn_device = MNDevice::instance();
            device = mn_device.metal_device();

            // Check if device supports counters (iOS 14+, macOS 11+)
            if (@available(macOS 11.0, iOS 14.0, *)) {
                counter_sets = [device counterSets];
                supported = (counter_sets != nil && [counter_sets count] > 0);

                if (supported) {
                    // Create a sample buffer for reading counters
                    // Use a small buffer since we only need one sample at a time
                    MTLCounterSampleBufferDescriptor* desc =
                        [[MTLCounterSampleBufferDescriptor alloc] init];

                    // Try to find a common counter set
                    id<MTLCounterSet> primary_set = nil;
                    for (id<MTLCounterSet> set in counter_sets) {
                        // Prefer stage_utilization or timestamp sets
                        NSString* name = [set name];
                        if ([name containsString:@"timestamp"] ||
                            [name containsString:@"utilization"]) {
                            primary_set = set;
                            break;
                        }
                    }

                    // If we didn't find a preferred set, use the first available
                    if (primary_set == nil && [counter_sets count] > 0) {
                        primary_set = counter_sets[0];
                    }

                    if (primary_set != nil) {
                        desc.counterSet = primary_set;
                        desc.storageMode = MTLStorageModeShared;
                        desc.sampleCount = 1;

                        NSError* error = nil;
                        sample_buffer = [device newCounterSampleBufferWithDescriptor:desc
                                                                               error:&error];

                        if (sample_buffer == nil || error != nil) {
                            // Failed to create sample buffer - mark as unsupported
                            supported = false;
                        }
                    } else {
                        supported = false;
                    }
                }
            } else {
                supported = false;
            }
        }
    }

    ~Impl() {
        // ARC handles cleanup
    }
};

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

PerfCounterReader::PerfCounterReader()
    : impl_(std::make_unique<Impl>()) {}

PerfCounterReader::~PerfCounterReader() = default;

// ---------------------------------------------------------------------------
// Counter Discovery
// ---------------------------------------------------------------------------

std::vector<std::string> PerfCounterReader::available_counter_sets() const {
    std::vector<std::string> result;

    if (!impl_->supported || impl_->counter_sets == nil) {
        return result;
    }

    @autoreleasepool {
        for (id<MTLCounterSet> set in impl_->counter_sets) {
            NSString* name = [set name];
            if (name) {
                result.push_back([name UTF8String]);
            }
        }
    }

    return result;
}

bool PerfCounterReader::is_supported() const {
    return impl_->supported;
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

PerfCounterSnapshot PerfCounterReader::sample() {
    PerfCounterSnapshot snapshot;

    if (!impl_->supported) {
        snapshot.is_valid = false;
        return snapshot;
    }

    @autoreleasepool {
        // Note: Actual counter sampling requires command buffer integration
        // For now, we provide a basic implementation that returns valid=false
        // A full implementation would:
        // 1. Encode a sample command on a command buffer
        // 2. Commit and wait for completion
        // 3. Resolve the sample buffer
        // 4. Parse counter data

        // For this initial implementation, we demonstrate the structure
        // but mark as invalid since we'd need active GPU work to sample
        snapshot.is_valid = false;
        snapshot.timestamp_ns = std::chrono::steady_clock::now()
            .time_since_epoch()
            .count();

        // Counter sampling not implemented in v0.1.0
        // This requires:
        // - Creating a command buffer
        // - Encoding sampleCountersInBuffer:atSampleIndex:withBarrier:
        // - Committing and waiting
        // - Reading resolved data from the sample buffer
        //
        // For real-world usage, counters are typically sampled around
        // GPU work submissions, not as standalone queries.
        // Use is_supported() to check availability before calling sample().
    }

    return snapshot;
}

} // namespace metal_native
