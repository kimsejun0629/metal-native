/// @file trace.mm
/// @brief Objective-C++ implementation of TraceManager.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <os/signpost.h>

#include "metal_native/profiling/trace.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

#include <atomic>
#include <mutex>

namespace metal_native {

// ---------------------------------------------------------------------------
// TraceScope
// ---------------------------------------------------------------------------

TraceScope::TraceScope(const std::string& name) : name_(name) {
    TraceManager::instance().begin_signpost(name_);
}

TraceScope::~TraceScope() {
    TraceManager::instance().end_signpost(name_);
}

// ---------------------------------------------------------------------------
// TraceManager::Impl
// ---------------------------------------------------------------------------

struct TraceManager::Impl {
    MTLCaptureManager*      capture_manager = nil;
    MTLCaptureDescriptor*   capture_desc    = nil;
    os_log_t                log_handle      = nullptr;

    std::atomic<bool>       capturing{false};
    mutable std::mutex      mu;

    std::string             capture_path;

    Impl() {
        @autoreleasepool {
            capture_manager = [MTLCaptureManager sharedCaptureManager];

            // Create os_log handle for signpost markers
            log_handle = os_log_create("com.metal_native.profiling", "gpu_ops");
        }
    }

    ~Impl() {
        // Note: os_log_t is not explicitly released in modern macOS
    }
};

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

TraceManager& TraceManager::instance() {
    static TraceManager instance;
    return instance;
}

TraceManager::TraceManager() : impl_(std::make_unique<Impl>()) {}

TraceManager::~TraceManager() = default;

// ---------------------------------------------------------------------------
// GPU Capture
// ---------------------------------------------------------------------------

void TraceManager::begin_capture() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    if (impl_->capturing.load()) {
        MN_THROW(MetalNativeError::InternalError,
                 "TraceManager::begin_capture: capture already in progress");
    }

    @autoreleasepool {
        MNDevice& device = MNDevice::instance();
        id<MTLDevice> mtl_device = device.metal_device();

        MTLCaptureDescriptor* desc = [[MTLCaptureDescriptor alloc] init];
        desc.captureObject = mtl_device;

        // Set destination if provided
        if (!impl_->capture_path.empty()) {
            NSURL* url = [NSURL fileURLWithPath:
                [NSString stringWithUTF8String:impl_->capture_path.c_str()]];
            desc.destination = MTLCaptureDestinationGPUTraceDocument;
            desc.outputURL = url;
        } else {
            // Default to developer tools
            desc.destination = MTLCaptureDestinationDeveloperTools;
        }

        NSError* error = nil;
        BOOL success = [impl_->capture_manager startCaptureWithDescriptor:desc
                                                                     error:&error];

        if (!success || error != nil) {
            std::string error_msg = "TraceManager::begin_capture: failed to start capture";
            if (error) {
                NSString* desc = [error localizedDescription];
                error_msg += " - " + std::string([desc UTF8String]);
            }
            MN_THROW(MetalNativeError::InternalError, error_msg);
        }

        impl_->capture_desc = desc;
        impl_->capturing.store(true);
    }
}

void TraceManager::end_capture() {
    std::lock_guard<std::mutex> lock(impl_->mu);

    if (!impl_->capturing.load()) {
        MN_THROW(MetalNativeError::InternalError,
                 "TraceManager::end_capture: no capture in progress");
    }

    @autoreleasepool {
        [impl_->capture_manager stopCapture];
        impl_->capturing.store(false);
        impl_->capture_desc = nil;
    }
}

bool TraceManager::is_capturing() const {
    return impl_->capturing.load();
}

void TraceManager::set_capture_destination(const std::string& path) {
    std::lock_guard<std::mutex> lock(impl_->mu);

    MN_CHECK(!impl_->capturing.load(),
             MetalNativeError::InternalError,
             "TraceManager::set_capture_destination: cannot change destination "
             "while capture is in progress");

    impl_->capture_path = path;
}

// ---------------------------------------------------------------------------
// Signpost Markers
// ---------------------------------------------------------------------------

void TraceManager::begin_signpost(const std::string& name) {
    if (impl_->log_handle) {
        os_signpost_interval_begin(impl_->log_handle,
                                   OS_SIGNPOST_ID_EXCLUSIVE,
                                   "gpu_operation",
                                   "%s", name.c_str());
    }
}

void TraceManager::end_signpost(const std::string& name) {
    if (impl_->log_handle) {
        os_signpost_interval_end(impl_->log_handle,
                                 OS_SIGNPOST_ID_EXCLUSIVE,
                                 "gpu_operation",
                                 "%s", name.c_str());
    }
}

} // namespace metal_native
