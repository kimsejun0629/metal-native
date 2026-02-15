<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# src
## Purpose
Container directory for all implementation files (.mm, .cpp, .metal). Holds implementation subdirectories matching the include/ hierarchy.
## Key Files
| File | Description |
|------|-------------|
| (none) | Empty container directory |
## For AI Agents
### Working In This Directory
This directory contains no direct files - all implementations are under subdirectories. Each subdirectory corresponds to an `include/metal_native/` subdirectory.
### Common Patterns
- Objective-C++ files use `.mm` extension (Metal/Foundation interop)
- Pure C++ files use `.cpp` extension
- Each subdirectory has a `CMakeLists.txt` for build configuration
- Implementation files include their corresponding public header
## Dependencies
### Internal
- Subdirectories: core/, memory/, dispatch/, kernels/, graph/, ops/, interop/, profiling/, future/
### External
- Metal framework (Objective-C++ files)
- MetalPerformanceShadersGraph framework
<!-- MANUAL: -->
