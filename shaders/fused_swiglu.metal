#include <metal_stdlib>
using namespace metal;

/**
 * Fused SwiGLU Activation Kernel
 *
 * Implements the SwiGLU activation used in Llama and Mistral FFN layers:
 *   output = SiLU(gate) * up
 * where SiLU(x) = x / (1 + exp(-x))
 *
 * This kernel fuses the SiLU activation and element-wise multiplication
 * into a single operation for better memory efficiency and performance.
 */

kernel void fused_swiglu_fp32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    // Each thread processes 4 elements using float4 vectorization
    const uint idx = id * 4;

    if (idx + 3 < num_elements) {
        // Full vector path: process 4 elements at once
        float4 g = *reinterpret_cast<device const float4*>(gate + idx);
        float4 u = *reinterpret_cast<device const float4*>(up + idx);

        // Compute SiLU(gate): g / (1 + exp(-g))
        float4 silu_g = g / (1.0f + exp(-g));

        // Multiply by up vector
        float4 result = silu_g * u;

        // Write results back
        *reinterpret_cast<device float4*>(output + idx) = result;
    } else {
        // Tail handling: process remaining elements individually
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            float g = gate[i];
            float silu_g = g / (1.0f + exp(-g));
            output[i] = silu_g * up[i];
        }
    }
}

kernel void fused_swiglu_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& num_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    // Each thread processes 4 elements using half4 vectorization
    const uint idx = id * 4;

    if (idx + 3 < num_elements) {
        // Full vector path: process 4 elements at once
        half4 g_h = *reinterpret_cast<device const half4*>(gate + idx);
        half4 u_h = *reinterpret_cast<device const half4*>(up + idx);

        // Promote to FP32 for numerical stability during computation
        float4 g = float4(g_h);
        float4 u = float4(u_h);

        // Compute SiLU(gate): g / (1 + exp(-g))
        float4 silu_g = g / (1.0f + exp(-g));

        // Multiply by up vector
        float4 result = silu_g * u;

        // Convert back to FP16 and write results
        half4 result_h = half4(result);
        *reinterpret_cast<device half4*>(output + idx) = result_h;
    } else {
        // Tail handling: process remaining elements individually
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            float g = float(gate[i]);
            float silu_g = g / (1.0f + exp(-g));
            output[i] = half(silu_g * float(up[i]));
        }
    }
}
