<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# core
## Purpose
Core tensor and device abstraction layer. Provides MNTensor (multi-dimensional arrays), MNBuffer (GPU memory), MNDevice (singleton GPU handle), data types, shapes, error handling, and autorelease pool management.
## Key Files
| File | Description |
|------|-------------|
| tensor.h | MNTensor: multi-dimensional GPU tensor with shape, strides, dtype, and view operations |
| buffer.h | MNBuffer: MTLBuffer wrapper with pooled allocation and zero-copy support |
| device.h | MNDevice: singleton wrapper around MTLDevice and command queue |
| dtype.h | MNDType enum and utilities (Float32, Float16, BFloat16, Int*, UInt8, Bool) |
| shape.h | MNShape: tensor shape with stride computation and broadcasting |
| error.h | MetalNativeError enum and MNException with typed error codes |
| autorelease_scope.h | AutoreleaseScope: RAII autorelease pool for Metal API objects |
## For AI Agents
### Working In This Directory
- **MNTensor** is the primary user-facing data container - all ops consume and produce tensors
- **MNDevice** is a process-wide singleton accessed via `MNDevice::instance()`
- Use **AutoreleaseScope** in tight loops to drain autoreleased Metal objects
- Buffers support three allocation modes: direct allocation, pooled (via allocator), zero-copy (wrap external memory)
- Tensors support views (reshape, slice) that share storage via `shared_ptr<MNBuffer>`
### Common Patterns
```cpp
// Device access
MNDevice& device = MNDevice::instance();

// Allocate tensor
MNTensor t = MNTensor::zeros({2, 3}, MNDType::Float32, device);

// View operations (zero-copy)
MNTensor reshaped = t.reshape({6});
MNTensor sliced = t.slice(0, 0, 1);  // first row

// Error handling
MN_CHECK(condition, MetalNativeError::InvalidArgument, "message");
```
## Dependencies
### Internal
- None (core has no internal dependencies)
### External
- Metal framework (MTLDevice, MTLBuffer, MTLCommandQueue)
- Foundation framework (autorelease pool APIs)
<!-- MANUAL: -->
