<!-- Parent: ../../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# metal_native
## Purpose
Root namespace directory for all public metal_native APIs. Contains umbrella header and version template, plus subdirectories for each API module.
## Key Files
| File | Description |
|------|-------------|
| metal_native.h | Umbrella header including all public headers |
| version.h.in | CMake template for version macros |
## For AI Agents
### Working In This Directory
- `metal_native.h` is the single-include convenience header for library consumers
- `version.h.in` is configured by CMake to generate `version.h` with actual version numbers
- All functionality is organized into subdirectories: core/, dispatch/, memory/, kernels/, ops/, graph/, interop/, profiling/, future/
### Common Patterns
- Include umbrella header: `#include <metal_native/metal_native.h>`
- Or include specific modules: `#include <metal_native/core/tensor.h>`
- Version macros: `METAL_NATIVE_VERSION_MAJOR`, `METAL_NATIVE_VERSION_MINOR`, `METAL_NATIVE_VERSION_PATCH`
## Dependencies
### Internal
- All subdirectories (core/, dispatch/, memory/, kernels/, ops/, graph/, interop/, profiling/, future/)
### External
- Metal framework
- MetalPerformanceShadersGraph framework
<!-- MANUAL: -->
