// SIMD-group reduction helpers
#ifndef SIMD_UTILS_H
#define SIMD_UTILS_H

#include <metal_stdlib>
using namespace metal;

// Warp-level (SIMD-group) sum reduction
template<typename T>
inline T simd_sum_reduce(T value) {
    return simd_sum(value); // Built-in MSL function
}

// SIMD-group max reduction
template<typename T>
inline T simd_max_reduce(T value) {
    return simd_max(value);
}

// SIMD-group min reduction
template<typename T>
inline T simd_min_reduce(T value) {
    return simd_min(value);
}

#endif
