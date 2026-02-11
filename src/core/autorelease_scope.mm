/// @file autorelease_scope.mm
/// @brief Objective-C++ implementation of AutoreleaseScope.
///
/// Uses the C-level autorelease-pool functions that are compatible with ARC
/// (-fobjc-arc).  NSAutoreleasePool is NOT used because it is forbidden
/// under ARC.

#include "metal_native/core/autorelease_scope.h"

// These are the low-level ObjC runtime functions for autorelease pool
// management.  They are declared in <objc/objc-arc.h> on some toolchains,
// but we provide our own declarations to stay portable.
extern "C" {
    void* objc_autoreleasePoolPush(void);
    void  objc_autoreleasePoolPop(void* pool);
}

namespace metal_native {

AutoreleaseScope::AutoreleaseScope()
    : pool_(objc_autoreleasePoolPush()) {}

AutoreleaseScope::~AutoreleaseScope() {
    objc_autoreleasePoolPop(pool_);
}

} // namespace metal_native
