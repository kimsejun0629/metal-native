// Shared MSL type aliases
#ifndef METAL_TYPES_H
#define METAL_TYPES_H

#include <metal_stdlib>
using namespace metal;

// Type aliases for kernel templates
typedef float    float32_t;
typedef half     float16_t;

// BFloat16 not natively supported in MSL - use float for accumulation
// and convert at memory boundary

#endif
