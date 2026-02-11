/// @file kernel_registry.mm
/// @brief Objective-C++ implementation of KernelRegistry.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/kernels/kernel_registry.h"
#include "metal_native/core/device.h"
#include "metal_native/core/error.h"

#include <mutex>
#include <string>
#include <unordered_map>

namespace metal_native {

// ---------------------------------------------------------------------------
// Impl (pimpl -- hides Objective-C types from the header)
// ---------------------------------------------------------------------------

struct KernelRegistry::Impl {
    /// Metal library loaded from .metallib files.
    id<MTLLibrary> library = nil;

    /// Mapping from user-facing kernel name -> Metal function name.
    std::unordered_map<std::string, std::string> name_to_function;

    /// Cached pipeline states (created lazily).
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    /// Protects all mutable state.
    mutable std::mutex mutex;
};

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

KernelRegistry& KernelRegistry::instance() {
    static std::once_flag flag;
    static KernelRegistry* singleton = nullptr;

    std::call_once(flag, [] {
        singleton = new KernelRegistry();
    });

    return *singleton;
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

KernelRegistry::KernelRegistry() : impl_(std::make_unique<Impl>()) {}

KernelRegistry::~KernelRegistry() = default;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void KernelRegistry::register_kernel(const std::string& name,
                                     const std::string& function_name) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->name_to_function[name] = function_name;
}

// ---------------------------------------------------------------------------
// Pipeline access
// ---------------------------------------------------------------------------

id<MTLComputePipelineState> KernelRegistry::get_pipeline(const std::string& name) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    // Check cached pipelines first.
    auto cached = impl_->pipelines.find(name);
    if (cached != impl_->pipelines.end()) {
        return cached->second;
    }

    // Look up the Metal function name.
    std::string func_name = name; // default: name == function name
    auto it = impl_->name_to_function.find(name);
    if (it != impl_->name_to_function.end()) {
        func_name = it->second;
    }

    MN_CHECK(impl_->library != nil,
             MetalNativeError::KernelCompilationFailed,
             "KernelRegistry: no Metal library loaded -- "
             "call load_library() or load_default_library() first");

    // Create the Metal function.
    NSString* ns_name = [NSString stringWithUTF8String:func_name.c_str()];
    id<MTLFunction> function = [impl_->library newFunctionWithName:ns_name];
    MN_CHECK(function != nil,
             MetalNativeError::KernelCompilationFailed,
             "KernelRegistry: function '" + func_name +
             "' not found in loaded Metal library");

    // Create the compute pipeline state.
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [MNDevice::instance().metal_device()
            newComputePipelineStateWithFunction:function
                                         error:&error];
    MN_CHECK(pipeline != nil,
             MetalNativeError::KernelCompilationFailed,
             "KernelRegistry: failed to create pipeline state for '" +
             func_name + "': " +
             (error ? std::string([[error localizedDescription] UTF8String])
                    : "unknown error"));

    // Cache and return.
    impl_->pipelines[name] = pipeline;
    return pipeline;
}

// ---------------------------------------------------------------------------
// Library loading
// ---------------------------------------------------------------------------

void KernelRegistry::load_library(const std::string& path) {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);

        @autoreleasepool {
            NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
            NSURL* url = [NSURL fileURLWithPath:ns_path];
            NSError* error = nil;

            impl_->library = [MNDevice::instance().metal_device()
                newLibraryWithURL:url
                            error:&error];

            MN_CHECK(impl_->library != nil,
                     MetalNativeError::KernelCompilationFailed,
                     "KernelRegistry: failed to load Metal library from '" +
                     path + "': " +
                     (error ? std::string([[error localizedDescription] UTF8String])
                            : "file not found"));
        }

        // Clear cached pipelines when loading a new library.
        impl_->pipelines.clear();
    }

    // Precompile all registered kernels asynchronously
    precompile_pipelines();
}

void KernelRegistry::load_default_library() {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);

        @autoreleasepool {
            NSError* error = nil;
            impl_->library = [MNDevice::instance().metal_device()
                newDefaultLibraryWithBundle:[NSBundle mainBundle]
                                     error:&error];

            MN_CHECK(impl_->library != nil,
                     MetalNativeError::KernelCompilationFailed,
                     "KernelRegistry: no default Metal library found in main bundle"
                     + (error ? std::string(": ") +
                        std::string([[error localizedDescription] UTF8String])
                              : std::string()));
        }

        impl_->pipelines.clear();
    }

    // Precompile all registered kernels asynchronously
    precompile_pipelines();
}

// ---------------------------------------------------------------------------
// Precompilation
// ---------------------------------------------------------------------------

void KernelRegistry::precompile_pipelines() {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (impl_->library == nil) {
        // No library loaded yet, nothing to precompile
        return;
    }

    @autoreleasepool {
        id<MTLDevice> device = MNDevice::instance().metal_device();

        // Iterate over all registered kernels
        for (const auto& entry : impl_->name_to_function) {
            const std::string& entry_name = entry.first;
            const std::string& entry_func = entry.second;

            // Skip if already cached
            if (impl_->pipelines.find(entry_name) != impl_->pipelines.end()) {
                continue;
            }

            // Get the Metal function
            NSString* ns_name = [NSString stringWithUTF8String:entry_func.c_str()];
            id<MTLFunction> function = [impl_->library newFunctionWithName:ns_name];

            if (function == nil) {
                // Log warning but continue with other kernels
                NSLog(@"KernelRegistry: warning: function '%s' not found during precompilation, skipping",
                      entry_func.c_str());
                continue;
            }

            // Copy for block capture (structured bindings cannot be captured in ObjC blocks)
            std::string captured_func = entry_func;
            std::string captured_name = entry_name;

            // Compile asynchronously
            [device newComputePipelineStateWithFunction:function
                                      completionHandler:^(id<MTLComputePipelineState> pipeline, NSError* error) {
                if (error != nil) {
                    NSLog(@"KernelRegistry: warning: failed to precompile pipeline for '%s': %@",
                          captured_func.c_str(), [error localizedDescription]);
                } else if (pipeline != nil) {
                    // Cache the compiled pipeline
                    std::lock_guard<std::mutex> cache_lock(impl_->mutex);
                    impl_->pipelines[captured_name] = pipeline;
                }
            }];
        }
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

size_t KernelRegistry::size() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->name_to_function.size();
}

} // namespace metal_native
