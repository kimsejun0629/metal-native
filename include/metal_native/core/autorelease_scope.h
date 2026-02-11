#pragma once

/// @file autorelease_scope.h
/// @brief RAII wrapper for Objective-C autorelease pools.
///
/// On Apple platforms, many Metal and Foundation APIs return autoreleased
/// objects.  In tight loops (e.g. per-dispatch) those objects accumulate
/// until the enclosing pool drains.  AutoreleaseScope creates a nested pool
/// that drains when the scope exits, bounding peak memory.
///
/// Implementation uses the C runtime functions objc_autoreleasePoolPush() /
/// objc_autoreleasePoolPop() which are compatible with ARC (-fobjc-arc).
/// Do NOT use NSAutoreleasePool -- it is forbidden under ARC.

namespace metal_native {

/// RAII autorelease pool scope.
///
/// Usage:
/// @code
///   {
///       AutoreleaseScope pool;
///       // ... Metal API calls that produce autoreleased objects ...
///   } // pool drains here
/// @endcode
class AutoreleaseScope {
public:
    AutoreleaseScope();
    ~AutoreleaseScope();

    // Non-copyable, non-movable.  Pools must be strictly nested.
    AutoreleaseScope(const AutoreleaseScope&) = delete;
    AutoreleaseScope& operator=(const AutoreleaseScope&) = delete;
    AutoreleaseScope(AutoreleaseScope&&) = delete;
    AutoreleaseScope& operator=(AutoreleaseScope&&) = delete;

private:
    void* pool_;
};

} // namespace metal_native
