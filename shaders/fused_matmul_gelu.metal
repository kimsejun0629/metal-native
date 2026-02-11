#include <metal_stdlib>
using namespace metal;

// Fused Bias + GELU: output = GELU(input + bias)
// This is the post-matmul fusion - matmul itself stays on MPSGraph/AMX
// Uses float4 vectorization

kernel void fused_bias_gelu_fp32(
    device const float* input       [[buffer(0)]],
    device const float* bias        [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant uint&      num_elements [[buffer(3)]],
    constant uint&      bias_size    [[buffer(4)]],  // hidden dim for bias wrapping
    uint id [[thread_position_in_grid]])
{
    uint idx = id * 4;
    if (idx >= num_elements) return;

    // Vectorized load
    uint remaining = min(num_elements - idx, 4u);

    float4 val;
    float4 b;

    if (remaining == 4) {
        val = *reinterpret_cast<device const float4*>(input + idx);
        // Bias wraps around hidden_dim
        b = float4(bias[idx % bias_size], bias[(idx+1) % bias_size],
                   bias[(idx+2) % bias_size], bias[(idx+3) % bias_size]);
    } else {
        val = float4(0);
        b = float4(0);
        for (uint i = 0; i < remaining; i++) {
            val[i] = input[idx + i];
            b[i] = bias[(idx + i) % bias_size];
        }
    }

    // Add bias
    val = val + b;

    // GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    float4 x3 = val * val * val;
    float4 inner = sqrt_2_over_pi * (val + 0.044715f * x3);
    float4 result = 0.5f * val * (1.0f + tanh(inner));

    if (remaining == 4) {
        *reinterpret_cast<device float4*>(output + idx) = result;
    } else {
        for (uint i = 0; i < remaining; i++) {
            output[idx + i] = result[i];
        }
    }
}

kernel void fused_bias_gelu_fp16(
    device const half*  input       [[buffer(0)]],
    device const half*  bias        [[buffer(1)]],
    device half*        output      [[buffer(2)]],
    constant uint&      num_elements [[buffer(3)]],
    constant uint&      bias_size    [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = id * 4;
    if (idx >= num_elements) return;

    uint remaining = min(num_elements - idx, 4u);

    half4 val;
    half4 b;

    if (remaining == 4) {
        val = *reinterpret_cast<device const half4*>(input + idx);
        b = half4(bias[idx % bias_size], bias[(idx+1) % bias_size],
                  bias[(idx+2) % bias_size], bias[(idx+3) % bias_size]);
    } else {
        val = half4(0);
        b = half4(0);
        for (uint i = 0; i < remaining; i++) {
            val[i] = input[idx + i];
            b[i] = bias[(idx + i) % bias_size];
        }
    }

    val = val + b;

    // Compute GELU in FP32 for stability
    const float sqrt_2_over_pi = 0.7978845608f;
    half4 result;
    for (uint i = 0; i < 4; i++) {
        float x = float(val[i]);
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
        result[i] = half(0.5f * x * (1.0f + tanh(inner)));
    }

    if (remaining == 4) {
        *reinterpret_cast<device half4*>(output + idx) = result;
    } else {
        for (uint i = 0; i < remaining; i++) {
            output[idx + i] = result[i];
        }
    }
}
