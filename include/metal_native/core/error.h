#pragma once

/// @file error.h
/// @brief Error handling primitives for metal_native.
///
/// Provides a typed error enum, an exception class derived from
/// std::runtime_error, and convenience macros for condition checking
/// and error throwing.

#include <cstdint>
#include <stdexcept>
#include <string>

namespace metal_native {

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

/// Exhaustive error codes for every failure mode in the library.
enum class MetalNativeError : uint32_t {
    Success = 0,
    DeviceNotFound,
    AllocationFailed,
    KernelCompilationFailed,
    InvalidArgument,
    OutOfMemory,
    TimeoutError,
    InternalError,
    NotImplemented,
};

/// Convert a MetalNativeError to a human-readable C string.
const char* error_to_string(MetalNativeError code) noexcept;

/// Convert a Metal API NSError code (NSInteger) to a descriptive string.
/// Useful for translating MTLCommandBuffer error codes, compilation errors, etc.
std::string metal_error_to_string(int64_t metal_error_code);

// ---------------------------------------------------------------------------
// Exception
// ---------------------------------------------------------------------------

/// Exception type carrying both a MetalNativeError code and a message.
///
/// Usage:
/// @code
///   throw MNException(MetalNativeError::AllocationFailed,
///                     "could not allocate 1 GiB buffer");
/// @endcode
class MNException : public std::runtime_error {
public:
    MNException(MetalNativeError code, const std::string& message);
    MNException(MetalNativeError code, const char* message);

    /// The structured error code associated with this exception.
    MetalNativeError code() const noexcept { return code_; }

private:
    MetalNativeError code_;
};

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------

/// Check a boolean condition; if false, throw MNException with the given
/// error code and message.
///
/// @param cond   Expression convertible to bool.
/// @param code   A MetalNativeError enumerator.
/// @param msg    A string literal or std::string describing the failure.
#define MN_CHECK(cond, code, msg)                                          \
    do {                                                                   \
        if (!(cond)) {                                                     \
            throw ::metal_native::MNException((code), (msg));              \
        }                                                                  \
    } while (false)

/// Unconditionally throw an MNException.
///
/// @param code   A MetalNativeError enumerator.
/// @param msg    A string literal or std::string describing the failure.
#define MN_THROW(code, msg)                                                \
    throw ::metal_native::MNException((code), (msg))

} // namespace metal_native
