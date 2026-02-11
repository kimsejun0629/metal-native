#pragma once

/// @file graph_serializer.h
/// @brief Serialization and deserialization of MPSGraphExecutable objects.
///
/// The GraphSerializer provides static methods for persisting compiled
/// MPSGraph executables to disk as .mpsgraphpackage files and reloading
/// them in subsequent sessions. Version validation ensures cache entries
/// are invalidated when the OS or Metal driver changes.

#include <string>

#ifdef __OBJC__
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Metal/Metal.h>
#endif

namespace metal_native {

// ---------------------------------------------------------------------------
// GraphSerializer
// ---------------------------------------------------------------------------

/// Static utility class for MPSGraphExecutable serialization.
class GraphSerializer {
public:
    // -- Serialization -------------------------------------------------------

#ifdef __OBJC__
    /// Serialize an MPSGraphExecutable to a .mpsgraphpackage file.
    /// Also writes a companion .json metadata file with version information.
    /// @param executable  The executable to serialize.
    /// @param path        Output file path (should end with .mpsgraphpackage).
    /// @throws MNException(InternalError) if serialization fails.
    static void serialize(MPSGraphExecutable* executable, const std::string& path);
#else
    /// Opaque handle variant.
    static void serialize(void* executable, const std::string& path);
#endif

    // -- Deserialization -----------------------------------------------------

#ifdef __OBJC__
    /// Deserialize an MPSGraphExecutable from a .mpsgraphpackage file.
    /// Validates version information from the companion .json file.
    /// @param path    Input file path (should end with .mpsgraphpackage).
    /// @param device  The Metal device to use for deserialization.
    /// @return The deserialized executable, or nil if the file is corrupt or
    ///         the version check fails.
    static MPSGraphExecutable* deserialize(const std::string& path, id<MTLDevice> device);
#else
    /// Opaque handle variant.
    static void* deserialize(const std::string& path, void* device);
#endif

    // -- Validation ----------------------------------------------------------

    /// Check if a cache entry is valid for the current system.
    /// Compares the macOS version and Metal driver version stored in the
    /// .json metadata file against the current system.
    /// @param path  Path to the .mpsgraphpackage file (the .json path is derived).
    /// @return true if the cache entry is valid, false otherwise.
    static bool validate_cache_entry(const std::string& path);

    // -- Utilities -----------------------------------------------------------

    /// Get the current macOS version string (e.g., "14.2.1").
    static std::string get_macos_version();

#ifdef __OBJC__
    /// Get the Metal driver version string from a device.
    static std::string get_metal_driver_version(id<MTLDevice> device);
#else
    /// Opaque handle variant.
    static std::string get_metal_driver_version(void* device);
#endif

    /// Derive the metadata file path from a .mpsgraphpackage path.
    /// E.g., "foo.mpsgraphpackage" -> "foo.mpsgraphpackage.json"
    static std::string metadata_path(const std::string& package_path);

private:
    // Static-only class.
    GraphSerializer() = delete;
};

} // namespace metal_native
