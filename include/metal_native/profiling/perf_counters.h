#pragma once

/// @file perf_counters.h
/// @brief GPU performance counter reader using MTLCounterSampleBuffer.
///
/// PerfCounterReader provides access to GPU hardware performance counters
/// on supported devices.  Counters include GPU utilization, ALU utilization,
/// memory bandwidth, and occupancy.
///
/// Not all counters are available on all GPU models.  Older Apple Silicon
/// chips may have limited counter support.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace metal_native {

/// Snapshot of GPU performance counters at a point in time.
struct PerfCounterSnapshot {
    /// GPU utilization as a percentage (0.0 - 100.0).
    /// Represents overall GPU busy time.
    double gpu_utilization = 0.0;

    /// ALU (Arithmetic Logic Unit) utilization as a percentage (0.0 - 100.0).
    /// Represents shader core computational throughput.
    double alu_utilization = 0.0;

    /// Memory bandwidth usage in GB/s.
    /// Actual measured bandwidth to/from GPU memory.
    double memory_bandwidth = 0.0;

    /// GPU occupancy as a percentage (0.0 - 100.0).
    /// Represents how well the GPU threads are utilizing available resources.
    double occupancy = 0.0;

    /// Timestamp (in nanoseconds since an arbitrary reference point).
    uint64_t timestamp_ns = 0;

    /// Indicates whether counter data is valid.
    /// False if counters are not supported or sampling failed.
    bool is_valid = false;

    /// Human-readable summary of the snapshot.
    std::string summary() const;
};

/// Reader for GPU performance counters.
///
/// Queries the Metal device for available counter sets and provides
/// sampling capabilities.  Handles gracefully when counters are not
/// available on older hardware.
class PerfCounterReader {
public:
    /// Construct a reader for the default Metal device.
    PerfCounterReader();
    ~PerfCounterReader();

    // Non-copyable, non-movable.
    PerfCounterReader(const PerfCounterReader&) = delete;
    PerfCounterReader& operator=(const PerfCounterReader&) = delete;
    PerfCounterReader(PerfCounterReader&&) = delete;
    PerfCounterReader& operator=(PerfCounterReader&&) = delete;

    // -- Counter Discovery ---------------------------------------------------

    /// List all available MTLCounterSet names on this device.
    ///
    /// Returns an empty vector if counters are not supported.
    ///
    /// @return Vector of counter set names (e.g., "timestamp", "stage_utilization").
    std::vector<std::string> available_counter_sets() const;

    /// Check if performance counters are supported on this device.
    ///
    /// @return true if at least one counter set is available.
    bool is_supported() const;

    // -- Sampling ------------------------------------------------------------

    /// Read current GPU performance counters.
    ///
    /// Takes a snapshot of GPU hardware counters.  If counters are not
    /// supported, returns a PerfCounterSnapshot with is_valid = false.
    ///
    /// @return PerfCounterSnapshot containing current counter values.
    PerfCounterSnapshot sample();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
