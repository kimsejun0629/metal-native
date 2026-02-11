#pragma once

#include "metal_native/memory/memory_pressure.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace metal_native {

class MNDevice;

enum class BudgetStrategy : uint8_t {
    GraphCache = 0,
    KVCache,
    ActivationCache,
    QuantWeightCache,
    SpeculativeBuffers,
    Count  // sentinel
};

struct GrantedBudget {
    size_t bytes;                    // Granted allocation budget
    MemoryPressureLevel pressure;   // Current pressure at grant time
    bool approved;                   // false if request denied
};

struct BudgetSnapshot {
    size_t total_capacity;          // device.recommendedMaxWorkingSetSize
    size_t model_memory;            // [device currentAllocatedSize]
    size_t headroom;                // total - model
    size_t distributable;           // headroom * (1 - safety_margin)
    MemoryPressureLevel pressure;
    float pressure_multiplier;      // Normal=1.0, Warning=0.5, Critical=0.1
};

class MemoryBudgetController {
public:
    static MemoryBudgetController& instance();

    GrantedBudget request_budget(BudgetStrategy strategy, size_t desired_bytes);
    BudgetSnapshot snapshot() const;
    void refresh();
    float pressure_multiplier() const;

    MemoryBudgetController(const MemoryBudgetController&) = delete;
    MemoryBudgetController& operator=(const MemoryBudgetController&) = delete;

private:
    MemoryBudgetController();
    ~MemoryBudgetController();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
