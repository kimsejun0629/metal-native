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

// BFloat16 conversion utilities (no native MSL support in Metal 3.0)
// BFloat16 is stored as ushort in Metal buffers
inline float bf16_to_float(ushort val) {
    uint bits = uint(val) << 16;
    return as_type<float>(bits);
}

inline ushort float_to_bf16(float val) {
    uint bits = as_type<uint>(val);
    bits += 0x7FFF + ((bits >> 16) & 1); // round to nearest even
    return ushort(bits >> 16);
}

// Vectorized BF16 conversion (4 elements)
inline float4 bf16x4_to_float4(ushort4 val) {
    return float4(bf16_to_float(val.x), bf16_to_float(val.y),
                  bf16_to_float(val.z), bf16_to_float(val.w));
}

inline ushort4 float4_to_bf16x4(float4 val) {
    return ushort4(float_to_bf16(val.x), float_to_bf16(val.y),
                   float_to_bf16(val.z), float_to_bf16(val.w));
}

#endif
