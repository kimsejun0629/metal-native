// Math helper functions
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <metal_stdlib>
using namespace metal;

// Fast GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline float gelu_fast(float x) {
    const float c = 0.7978845608f; // sqrt(2/pi)
    return x * 0.5f * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}

inline half gelu_fast(half x) {
    return (half)gelu_fast((float)x);
}

// SiLU (Swish): x * sigmoid(x)
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

inline half silu(half x) {
    return (half)silu((float)x);
}

#endif
