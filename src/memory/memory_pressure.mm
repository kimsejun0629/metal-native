#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include "metal_native/memory/memory_pressure.h"
#include "metal_native/core/error.h"

namespace metal_native {

MemoryPressureMonitor::MemoryPressureMonitor() : dispatch_source_(nullptr) {
    // Auto-start on first use
    start();
}

MemoryPressureMonitor::~MemoryPressureMonitor() {
    stop();
}

MemoryPressureMonitor& MemoryPressureMonitor::instance() {
    static MemoryPressureMonitor instance;
    return instance;
}

void MemoryPressureMonitor::start() {
    // Idempotent: only start once
    bool expected = false;
    if (!is_monitoring_.compare_exchange_strong(expected, true)) {
        return; // Already monitoring
    }

    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    dispatch_source_t source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE,
        0,
        DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
        queue
    );

    MN_CHECK(source != nullptr, MetalNativeError::InternalError,
             "Failed to create memory pressure dispatch source");

    dispatch_source_ = (__bridge_retained void*)source;

    // Capture 'this' for event handler
    dispatch_source_set_event_handler(source, ^{
        unsigned long flags = dispatch_source_get_data(source);
        MemoryPressureLevel new_level = MemoryPressureLevel::Normal;

        if (flags & DISPATCH_MEMORYPRESSURE_CRITICAL) {
            new_level = MemoryPressureLevel::Critical;
        } else if (flags & DISPATCH_MEMORYPRESSURE_WARN) {
            new_level = MemoryPressureLevel::Warning;
        }

        MemoryPressureLevel old_level = current_level_.exchange(new_level);

        // Only invoke callback if level actually changed
        if (new_level != old_level && callback_) {
            callback_(new_level);
        }
    });

    dispatch_resume(source);
}

void MemoryPressureMonitor::stop() {
    bool expected = true;
    if (!is_monitoring_.compare_exchange_strong(expected, false)) {
        return; // Not monitoring
    }

    if (dispatch_source_) {
        dispatch_source_t source = (__bridge_transfer dispatch_source_t)dispatch_source_;
        dispatch_source_cancel(source);
        // No need for dispatch_release under ARC with __bridge_transfer
        dispatch_source_ = nullptr;
    }
}

void MemoryPressureMonitor::set_callback(MemoryPressureCallback cb) {
    callback_ = std::move(cb);
}

MemoryPressureLevel MemoryPressureMonitor::current_level() const {
    return current_level_.load();
}

size_t MemoryPressureMonitor::available_memory() const {
    // Get actual available (free + inactive) memory from mach
    vm_size_t page_size;
    mach_port_t host_port = mach_host_self();
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t host_size = sizeof(vm_statistics64_data_t) / sizeof(integer_t);

    if (host_page_size(host_port, &page_size) != KERN_SUCCESS) {
        return 0;
    }

    if (host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size) != KERN_SUCCESS) {
        return 0;
    }

    // Available memory = free + inactive pages
    return (vm_stat.free_count + vm_stat.inactive_count) * page_size;
}

size_t MemoryPressureMonitor::recommended_limit() const {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        size_t device_allocated = [device currentAllocatedSize];
        size_t max_working_set = [device recommendedMaxWorkingSetSize];
        if (device_allocated >= max_working_set) return 0;
        return max_working_set - device_allocated;
    }
}

} // namespace metal_native
