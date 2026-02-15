<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# include
## Purpose
Container directory for all public C++ header files. Holds the `metal_native/` subdirectory tree.
## Key Files
| File | Description |
|------|-------------|
| (none) | Empty container directory |
## For AI Agents
### Working In This Directory
This directory contains no direct files - all headers are under `metal_native/`. Navigate to subdirectories for actual API surfaces.
### Common Patterns
- All public APIs are under `include/metal_native/`
- Headers use `#pragma once` for include guards
- Objective-C types are hidden behind `#ifdef __OBJC__` for C++ compatibility
## Dependencies
### Internal
- metal_native/ (subdirectory)
### External
- Metal framework (Apple)
- MetalPerformanceShadersGraph framework (Apple)
<!-- MANUAL: -->
