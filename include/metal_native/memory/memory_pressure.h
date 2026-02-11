#pragma once
#include <cstdint>
#include <functional>
#include <atomic>

namespace metal_native {

enum class MemoryPressureLevel : uint8_t {
    Normal = 0,
    Warning = 1,   // DISPATCH_MEMORYPRESSURE_WARN
    Critical = 2   // DISPATCH_MEMORYPRESSURE_CRITICAL
};

// Callback type: receives pressure level
using MemoryPressureCallback = std::function<void(MemoryPressureLevel)>;

class MemoryPressureMonitor {
public:
    static MemoryPressureMonitor& instance();

    // Start monitoring (idempotent)
    void start();
    // Stop monitoring
    void stop();

    // Register callback (called on pressure change)
    void set_callback(MemoryPressureCallback cb);

    // Query current level
    MemoryPressureLevel current_level() const;

    // Query available system memory
    size_t available_memory() const;

    // Query recommended memory limit (from os_proc_available_memory)
    size_t recommended_limit() const;

private:
    MemoryPressureMonitor();
    ~MemoryPressureMonitor();

    MemoryPressureMonitor(const MemoryPressureMonitor&) = delete;
    MemoryPressureMonitor& operator=(const MemoryPressureMonitor&) = delete;

    // Platform-specific implementation details
    void* dispatch_source_;  // dispatch_source_t
    std::atomic<MemoryPressureLevel> current_level_{MemoryPressureLevel::Normal};
    MemoryPressureCallback callback_;
    std::atomic<bool> is_monitoring_{false};
};

} // namespace metal_native
