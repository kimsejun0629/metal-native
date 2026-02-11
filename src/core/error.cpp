/// @file error.cpp
/// @brief Implementation of error handling utilities.

#include "metal_native/core/error.h"

#include <sstream>

namespace metal_native {

// ---------------------------------------------------------------------------
// error_to_string
// ---------------------------------------------------------------------------

const char* error_to_string(MetalNativeError code) noexcept {
    switch (code) {
        case MetalNativeError::Success:                  return "Success";
        case MetalNativeError::DeviceNotFound:           return "DeviceNotFound";
        case MetalNativeError::AllocationFailed:         return "AllocationFailed";
        case MetalNativeError::KernelCompilationFailed:  return "KernelCompilationFailed";
        case MetalNativeError::InvalidArgument:          return "InvalidArgument";
        case MetalNativeError::OutOfMemory:              return "OutOfMemory";
        case MetalNativeError::TimeoutError:             return "TimeoutError";
        case MetalNativeError::InternalError:            return "InternalError";
        case MetalNativeError::NotImplemented:           return "NotImplemented";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// metal_error_to_string
// ---------------------------------------------------------------------------

std::string metal_error_to_string(int64_t metal_error_code) {
    // Metal command-buffer error codes (MTLCommandBufferError).
    // These are the most commonly encountered Metal API errors.
    switch (metal_error_code) {
        case 0:    return "MTLCommandBufferErrorNone";
        case 1:    return "MTLCommandBufferErrorInternal (unexpected Metal internal error)";
        case 2:    return "MTLCommandBufferErrorTimeout (GPU command timed out)";
        case 3:    return "MTLCommandBufferErrorPageFault (GPU page fault)";
        case 4:    return "MTLCommandBufferErrorBlacklisted (process is no longer permitted to use Metal)";
        case 5:    return "MTLCommandBufferErrorNotPermitted (command not permitted)";
        case 6:    return "MTLCommandBufferErrorOutOfMemory (GPU ran out of memory)";
        case 7:    return "MTLCommandBufferErrorInvalidResource (invalid resource referenced)";
        case 8:    return "MTLCommandBufferErrorMemoryless (memoryless resource accessed outside tile)";
        case 9:    return "MTLCommandBufferErrorDeviceRemoved (external GPU removed)";
        case 10:   return "MTLCommandBufferErrorStackOverflow (GPU stack overflow)";
        default: {
            std::ostringstream oss;
            oss << "Unknown Metal error (code " << metal_error_code << ")";
            return oss.str();
        }
    }
}

// ---------------------------------------------------------------------------
// MNException
// ---------------------------------------------------------------------------

MNException::MNException(MetalNativeError code, const std::string& message)
    : std::runtime_error(
          std::string(error_to_string(code)) + ": " + message),
      code_(code) {}

MNException::MNException(MetalNativeError code, const char* message)
    : std::runtime_error(
          std::string(error_to_string(code)) + ": " + message),
      code_(code) {}

} // namespace metal_native
