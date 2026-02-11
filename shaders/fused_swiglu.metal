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
        float4 g = float4(gate[idx], gate[idx+1], gate[idx+2], gate[idx+3]);
        float4 u = float4(up[idx], up[idx+1], up[idx+2], up[idx+3]);

        // Compute SiLU(gate): g / (1 + exp(-g))
        float4 silu_g = g / (1.0f + exp(-g));

        // Multiply by up vector
        float4 result = silu_g * u;

        // Write results back
        output[idx] = result.x;
        output[idx+1] = result.y;
        output[idx+2] = result.z;
        output[idx+3] = result.w;
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
        half4 g_h = half4(gate[idx], gate[idx+1], gate[idx+2], gate[idx+3]);
        half4 u_h = half4(up[idx], up[idx+1], up[idx+2], up[idx+3]);

        // Promote to FP32 for numerical stability during computation
        float4 g = float4(g_h);
        float4 u = float4(u_h);

        // Compute SiLU(gate): g / (1 + exp(-g))
        float4 silu_g = g / (1.0f + exp(-g));

        // Multiply by up vector
        float4 result = silu_g * u;

        // Convert back to FP16 and write results
        half4 result_h = half4(result);
        output[idx] = result_h.x;
        output[idx+1] = result_h.y;
        output[idx+2] = result_h.z;
        output[idx+3] = result_h.w;
    } else {
        // Tail handling: process remaining elements individually
        for (uint i = idx; i < min(idx + 4, num_elements); i++) {
            float g = float(gate[i]);
            float silu_g = g / (1.0f + exp(-g));
            output[i] = half(silu_g * float(up[i]));
        }
    }
}
