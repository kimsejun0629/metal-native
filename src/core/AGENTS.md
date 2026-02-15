<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# core
## Purpose
Implementation files for core tensor and device abstractions. Implements MNTensor, MNBuffer, MNDevice, dtype utilities, shape operations, error handling, and autorelease pool management.
## Key Files
| File | Description |
|------|-------------|
| CMakeLists.txt | Build configuration for core module |
| tensor.cpp | MNTensor implementation (view operations, fills, cloning) |
| buffer.mm | MNBuffer implementation (MTLBuffer allocation, pooling, zero-copy) |
| device.mm | MNDevice singleton implementation (MTLDevice wrapper, command queue) |
| dtype.cpp | Data type utilities (size, name, type queries, MTL format mapping) |
| shape.cpp | Shape operations (contiguous strides, broadcasting, contiguity checks) |
| error.cpp | Error code string conversion and exception implementation |
| autorelease_scope.mm | AutoreleaseScope RAII implementation (objc_autoreleasePool* calls) |
## For AI Agents
### Working In This Directory
- `.mm` files use Objective-C++ for Metal/Foundation interop
- `.cpp` files are pure C++ (no Metal API direct usage)
- MNBuffer manages MTLBuffer lifetime and provides CPU/GPU pointer access
- MNDevice is a singleton with std::call_once initialization
- AutoreleaseScope uses C runtime functions (compatible with ARC)
### Common Patterns
- Objective-C++ files cast opaque `void*` handles to `id<MTLDevice>`, `id<MTLBuffer>`, etc.
- Pure C++ files work only with abstracted types (no Metal headers)
- Error handling uses `MN_CHECK` and `MN_THROW` macros
- All implementations include their corresponding public header first
## Dependencies
### Internal
- ../include/metal_native/core/ (public headers)
### External
- Metal framework (buffer.mm, device.mm, autorelease_scope.mm)
- Foundation framework (autorelease pool)
<!-- MANUAL: -->
