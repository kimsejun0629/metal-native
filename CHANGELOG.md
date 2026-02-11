# Changelog

All notable changes to MetalNative will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Full Python tensor operation bindings (zeros, ones, empty, tensor creation)
- Element-wise arithmetic operations (add, sub, mul, div)
- Matrix multiplication via Metal GPU
- NumPy interoperability (zero-copy on UMA)
- DLPack tensor exchange protocol
- Neural network primitives (activation functions, loss functions)
- Optimizer implementations (SGD, Adam, AdamW)
- Profiling Python API
- Autograd / automatic differentiation (v0.2.0)
- API reference documentation (Sphinx)
- Pre-built wheels for PyPI

## [0.1.0-alpha] - 2024-02-10

### Added

#### C++ Core Infrastructure
- Modern C++17 Metal compute framework for Apple Silicon
- CMake-based build system with modular architecture (core, memory, dispatch, kernels, graph, ops, interop, profiling)
- Metal shader compilation pipeline (.metal -> .air -> .metallib)
- Comprehensive error handling with MNException hierarchy

#### Buffer and Memory Management (C++)
- High-performance Metal buffer abstraction (MNBuffer class)
- MTLHeap-based memory allocation with smart caching (HeapManager)
- Memory pressure monitoring and automatic cleanup
- Support for managed, shared, and private storage modes

#### Tensor Core (C++)
- N-dimensional tensor with dynamic shape and stride management
- Type-safe dtype system (float32, float16, bfloat16, int32, int64, int8, uint8, bool)
- MNShape and MNDType utility classes

#### Metal Compute Pipeline (C++)
- Optimized Metal compute kernels for element-wise, reduction, and matrix operations
- Kernel registry and caching system
- Threadgroup-based parallel execution
- SIMD optimizations using Metal vector types

#### Device Management (C++)
- Unified device abstraction for Metal GPUs
- Automatic device discovery and capability querying
- Memory and compute performance statistics

#### Graph Engine (C++)
- MPSGraph integration for graph-level optimization
- Fusion pattern matching
- Shape bucketing for compilation caching

#### Python Package Structure
- Python 3.10+ package with pybind11 bindings scaffold
- Lazy-loading module initialization
- DType enum bindings (functional)
- Device query stubs (return placeholder values)
- PyTorch-compatible API design (tensor.py, nn/, optim/)

#### Build and CI
- CMake 3.25+ build system with Ninja support
- GitHub Actions CI for macOS ARM64
- Code formatting checks (clang-format)
- Python test infrastructure (pytest)

#### Community Infrastructure
- Apache 2.0 license
- Contributing guidelines
- Security policy
- Code of Conduct

### Known Limitations
- Python tensor operations are not yet connected to C++ implementations
- All tensor creation/arithmetic functions raise RuntimeError
- DLPack and NumPy interop are stub implementations
- No autograd / backward pass support
- Limited to macOS with Metal-capable GPU
- Single-GPU execution only

### Development Status
> **Note:** This is an alpha release. The C++ backend is substantially implemented,
> but the Python bindings that connect to it are under active development.
> v0.1.0 should be used for evaluating the architecture and contributing,
> not for production workloads.

## [0.0.1] - 2024-01-15

### Added
- Project structure and build system
- Basic Metal device enumeration
- Initial Python bindings setup

---

[Unreleased]: https://github.com/kimsejun/metal-native/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/kimsejun/metal-native/releases/tag/v0.1.0-alpha
[0.0.1]: https://github.com/kimsejun/metal-native/releases/tag/v0.0.1
