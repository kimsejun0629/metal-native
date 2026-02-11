/// @file graph_serializer.mm
/// @brief Objective-C++ implementation of GraphSerializer.

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_native/graph/graph_serializer.h"
#include "metal_native/core/error.h"

#include <fstream>
#include <sstream>

namespace metal_native {

// ---------------------------------------------------------------------------
// Helper: JSON metadata read/write
// ---------------------------------------------------------------------------

namespace {

struct CacheMetadata {
    std::string macos_version;
    std::string metal_driver_version;
};

// Write metadata to JSON file
void write_metadata(const std::string& path, const CacheMetadata& meta) {
    std::ofstream out(path);
    MN_CHECK(out.is_open(),
             MetalNativeError::InternalError,
             "GraphSerializer: failed to open metadata file for writing: " + path);

    out << "{\n";
    out << "  \"macos_version\": \"" << meta.macos_version << "\",\n";
    out << "  \"metal_driver_version\": \"" << meta.metal_driver_version << "\"\n";
    out << "}\n";
}

// Read metadata from JSON file
bool read_metadata(const std::string& path, CacheMetadata& meta) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }

    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string content = buffer.str();

    // Simple JSON parsing (no external dependencies)
    size_t macos_pos = content.find("\"macos_version\"");
    size_t metal_pos = content.find("\"metal_driver_version\"");

    if (macos_pos == std::string::npos || metal_pos == std::string::npos) {
        return false;
    }

    // Extract macos_version value
    size_t macos_start = content.find("\"", macos_pos + 16);
    if (macos_start == std::string::npos) return false;
    size_t macos_end = content.find("\"", macos_start + 1);
    if (macos_end == std::string::npos) return false;
    meta.macos_version = content.substr(macos_start + 1, macos_end - macos_start - 1);

    // Extract metal_driver_version value
    size_t metal_start = content.find("\"", metal_pos + 23);
    if (metal_start == std::string::npos) return false;
    size_t metal_end = content.find("\"", metal_start + 1);
    if (metal_end == std::string::npos) return false;
    meta.metal_driver_version = content.substr(metal_start + 1, metal_end - metal_start - 1);

    return true;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

void GraphSerializer::serialize(MPSGraphExecutable* executable, const std::string& path) {
    MN_CHECK(executable != nil,
             MetalNativeError::InvalidArgument,
             "GraphSerializer::serialize: executable must not be nil");

    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
        MPSGraphExecutableSerializationDescriptor* desc = [MPSGraphExecutableSerializationDescriptor new];

        @try {
            [executable serializeToMPSGraphPackageAtURL:url descriptor:desc];
        }
        @catch (NSException* exception) {
            std::string error_msg = "GraphSerializer::serialize failed";
            if (exception != nil) {
                error_msg += ": " + std::string([[exception reason] UTF8String]);
            }
            MN_THROW(MetalNativeError::InternalError, error_msg);
        }

        // Write metadata
        CacheMetadata meta;
        meta.macos_version = get_macos_version();
        // Get device from system default since executable doesn't expose device property
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        meta.metal_driver_version = get_metal_driver_version(device);
        write_metadata(metadata_path(path), meta);
    }
}

// ---------------------------------------------------------------------------
// Deserialization
// ---------------------------------------------------------------------------

MPSGraphExecutable* GraphSerializer::deserialize(const std::string& path, id<MTLDevice> device) {
    MN_CHECK(device != nil,
             MetalNativeError::InvalidArgument,
             "GraphSerializer::deserialize: device must not be nil");

    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];

        MPSGraphExecutable* executable = nil;
        @try {
            executable = [[MPSGraphExecutable alloc] initWithMPSGraphPackageAtURL:url
                                                           compilationDescriptor:nil];
        }
        @catch (NSException* exception) {
            // Deserialization failed - return nil
            return nil;
        }

        return executable;
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

bool GraphSerializer::validate_cache_entry(const std::string& path) {
    @autoreleasepool {
        // Check if package file exists
        NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
        if (![[NSFileManager defaultManager] fileExistsAtPath:ns_path]) {
            return false;
        }

        // Read metadata
        CacheMetadata stored_meta;
        if (!read_metadata(metadata_path(path), stored_meta)) {
            return false; // No metadata or corrupt metadata
        }

        // Compare with current system
        std::string current_macos = get_macos_version();
        if (stored_meta.macos_version != current_macos) {
            return false; // macOS version mismatch
        }

        // Get current Metal driver version
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return false;
        }

        std::string current_metal = get_metal_driver_version(device);
        if (stored_meta.metal_driver_version != current_metal) {
            return false; // Metal driver version mismatch
        }

        return true;
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

std::string GraphSerializer::get_macos_version() {
    @autoreleasepool {
        NSOperatingSystemVersion version = [[NSProcessInfo processInfo] operatingSystemVersion];
        std::ostringstream oss;
        oss << version.majorVersion << "."
            << version.minorVersion << "."
            << version.patchVersion;
        return oss.str();
    }
}

std::string GraphSerializer::get_metal_driver_version(id<MTLDevice> device) {
    @autoreleasepool {
        if (device == nil) {
            return "unknown";
        }

        // Use device name and registry ID as a proxy for driver version.
        // Metal doesn't expose a direct driver version API, so we use
        // device-specific identifiers that change with driver updates.
        NSString* name = [device name];
        uint64_t registry_id = [device registryID];

        std::ostringstream oss;
        oss << [name UTF8String] << "-" << registry_id;
        return oss.str();
    }
}

std::string GraphSerializer::metadata_path(const std::string& package_path) {
    return package_path + ".json";
}

} // namespace metal_native
