<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# common

## Purpose
Shared Metal Shading Language (MSL) header files providing type aliases, math utilities, and SIMD reduction helpers for GPU kernels.

## Key Files
| File | Description |
|------|-------------|
| `metal_types.h` | Type aliases for float32_t, float16_t; BFloat16 handling note |
| `math_utils.h` | Fast GELU approximation (tanh-based) and SiLU activation functions |
| `simd_utils.h` | SIMD-group (warp-level) reduction templates (sum, max, min) |

## For AI Agents

### Working In This Directory
- **MSL**: Metal Shading Language (C++14-based GPU language)
- **Include Guards**: All headers use `#ifndef`/`#define`/`#endif` pattern
- **Namespace**: Uses `using namespace metal;` for Metal standard library
- **BFloat16**: Not natively supported in MSL - use float32 for accumulation, convert at memory boundaries

### Testing Requirements
- Test via C++ unit tests that compile and execute kernels
- Verify GELU/SiLU match reference implementations numerically
- Test SIMD reductions on full warps (32 threads on Apple GPUs)

### Common Patterns
- **Activation Functions**: Provide both `float` and `half` overloads for type flexibility
- **SIMD Reductions**: Use Metal's built-in `simd_sum()`, `simd_max()`, `simd_min()` functions
- **Fast Math**: GELU uses polynomial approximation for performance (less accurate than erf-based)
- **Template Functions**: `inline` templates for zero-overhead abstraction

## Dependencies

### Internal
- Included by kernel files in `../kernels/` directory
- Used by all elementwise and reduction operations

### External
- **Required**: Metal Shading Language compiler (part of Xcode)
- **Metal Standard Library**: `<metal_stdlib>` for SIMD primitives

<!-- MANUAL: -->
