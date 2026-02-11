#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include "metal_native/memory/budget_controller.h"
#include "metal_native/memory/memory_pressure.h"
#include "metal_native/core/device.h"

#include <algorithm>
#include <atomic>
#include <mutex>

namespace metal_native {

struct MemoryBudgetController::Impl {
    mutable std::mutex mu;  // Level 0 lock

    size_t total_capacity = 0;
    std::atomic<MemoryPressureLevel> current_pressure{MemoryPressureLevel::Normal};
    std::atomic<size_t> last_sampled_allocated{0};

    // Track which strategies have requested budget
    bool strategy_active[static_cast<size_t>(BudgetStrategy::Count)] = {};

    // GCD timer for periodic sampling
    dispatch_source_t sample_timer = nil;
    dispatch_queue_t timer_queue = nil;

    // Weak ref to the Metal device for currentAllocatedSize queries
    id<MTLDevice> metal_device = nil;
};

MemoryBudgetController::MemoryBudgetController() : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        MNDevice& dev = MNDevice::instance();
        impl_->total_capacity = dev.recommended_max_working_set_size();
        impl_->metal_device = dev.metal_device();

        // Subscribe to pressure changes
        MemoryPressureMonitor::instance().set_callback(
            [this](MemoryPressureLevel level) {
                impl_->current_pressure.store(level);
            });

        // Start sampling timer (100ms)
        impl_->timer_queue = dispatch_queue_create(
            "com.metal_native.budget_controller.sampling",
            DISPATCH_QUEUE_SERIAL);
        impl_->sample_timer = dispatch_source_create(
            DISPATCH_SOURCE_TYPE_TIMER, 0, 0, impl_->timer_queue);
        dispatch_source_set_timer(impl_->sample_timer,
            dispatch_time(DISPATCH_TIME_NOW, 0),
            100 * NSEC_PER_MSEC, 10 * NSEC_PER_MSEC);

        auto impl_ptr = impl_.get();
        dispatch_source_set_event_handler(impl_->sample_timer, ^{
            impl_ptr->last_sampled_allocated.store(
                [impl_ptr->metal_device currentAllocatedSize]);
        });
        dispatch_resume(impl_->sample_timer);
    }
}

MemoryBudgetController::~MemoryBudgetController() {
    if (impl_->sample_timer != nil) {
        dispatch_source_cancel(impl_->sample_timer);
    }
}

MemoryBudgetController& MemoryBudgetController::instance() {
    static std::once_flag flag;
    static MemoryBudgetController* singleton = nullptr;
    std::call_once(flag, [] {
        singleton = new MemoryBudgetController();
    });
    return *singleton;
}

float MemoryBudgetController::pressure_multiplier() const {
    switch (impl_->current_pressure.load()) {
        case MemoryPressureLevel::Normal:   return 1.0f;
        case MemoryPressureLevel::Warning:  return 0.5f;
        case MemoryPressureLevel::Critical: return 0.1f;
    }
    return 0.1f;
}

BudgetSnapshot MemoryBudgetController::snapshot() const {
    BudgetSnapshot snap;
    snap.total_capacity = impl_->total_capacity;
    snap.model_memory = impl_->last_sampled_allocated.load();
    snap.headroom = (snap.total_capacity > snap.model_memory)
                    ? (snap.total_capacity - snap.model_memory) : 0;
    size_t safety = snap.headroom / 10;  // 10% safety margin
    snap.distributable = snap.headroom - safety;
    snap.pressure = impl_->current_pressure.load();
    snap.pressure_multiplier = pressure_multiplier();
    return snap;
}

GrantedBudget MemoryBudgetController::request_budget(BudgetStrategy strategy, size_t desired_bytes) {
    GrantedBudget result;
    result.pressure = impl_->current_pressure.load();

    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        impl_->strategy_active[static_cast<size_t>(strategy)] = true;
    }

    BudgetSnapshot snap = snapshot();

    // Count active strategies
    size_t active_count = 0;
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        for (size_t i = 0; i < static_cast<size_t>(BudgetStrategy::Count); ++i) {
            if (impl_->strategy_active[i]) ++active_count;
        }
    }
    if (active_count == 0) active_count = 1;

    size_t per_strategy = static_cast<size_t>(
        snap.distributable * snap.pressure_multiplier / active_count);

    result.bytes = std::min(desired_bytes, per_strategy);
    result.approved = (result.bytes > 0);

    return result;
}

void MemoryBudgetController::refresh() {
    @autoreleasepool {
        impl_->last_sampled_allocated.store(
            [impl_->metal_device currentAllocatedSize]);
    }
}

} // namespace metal_native
