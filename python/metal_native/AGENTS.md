<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# metal_native

## Purpose
Python package providing PyTorch-compatible tensor operations on Apple Silicon GPUs via Metal and MPSGraph. Core user-facing API with zero-copy NumPy/PyTorch interop, automatic mixed precision, unified memory architecture optimization, and DLPack protocol support.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package entry point with lazy imports, C extension loader, and public API exports |
| `_version.py` | Version metadata (0.1.0) |
| `_C.cpython-312-darwin.so` | Compiled C++/Metal extension providing core GPU operations |
| `tensor.py` | Tensor class with arithmetic operators, DLPack/NumPy/PyTorch interop, and autograd hooks |
| `device.py` | Metal device query API (name, properties, bfloat16 support, UMA detection) |
| `dtypes.py` | Data type constants and NumPy conversion utilities (float32/16/bf16, int64/32/16/8, uint8, bool) |
| `config.py` | Runtime configuration dataclasses (allocator, backpressure, dispatch, kernel cache, profiling) |
| `utils.py` | GPU synchronization, memory management, and random seed utilities |
| `interop.py` | PyTorch tensor conversion with zero-copy via unified memory or DLPack |
| `dlpack_bridge.py` | DLPack protocol implementation for cross-framework tensor exchange |
| `profiling.py` | GPU trace capture, performance counters, and memory snapshots for Xcode Instruments |
| `accelerate_plugin.py` | HuggingFace Accelerate backend registration for seamless Transformers integration |
| `py.typed` | PEP 561 marker for type hint distribution |

## For AI Agents

### Working In This Directory
- **C Extension**: All GPU operations require `_C` module. Check `_C is not None` before calling native methods
- **Lazy Loading**: Use `__getattr__` pattern in `__init__.py` for submodule imports to avoid circular dependencies
- **Zero-Copy**: On UMA systems, tensors share memory with NumPy/PyTorch via `data_ptr()` or DLPack
- **DLPack Device**: Metal uses device type `8` (kDLMetal) with device ID `0`
- **Error Handling**: `_ensure_initialized()` raises RuntimeError with macOS/Apple Silicon requirements

### Testing Requirements
- Unit tests: Mock `_C` extension for tests that don't need GPU (see `tests/python/`)
- Integration tests: Mark with `@requires_metal` decorator and verify Metal availability
- Test both CPU and MPS tensor paths in `interop.py` (DLPack vs copy fallback)
- Verify memory tracking via `memory_allocated()`, `max_memory_allocated()`, `empty_cache()`

### Common Patterns
- **Tensor Construction**: Factory functions (`zeros`, `ones`, `randn`) delegate to `_C` then wrap in Python `Tensor`
- **Property Caching**: Tensor shape/dtype lazy-loaded from C++ on first access, cached in `_shape`/`_dtype`
- **Config Merging**: `MetalNativeConfig.from_dict()` uses partial updates over defaults
- **Type Conversion**: `dtypes.get_dtype()` accepts DType, str, or NumPy dtype; use for flexible inputs

## Dependencies

### Internal
- `nn/`: Neural network modules (placeholder for Phase 4)
- `optim/`: Optimizers (Adam, AdamW, SGD)
- C++ core: `metal_native/core/`, `metal_native/memory/`, `metal_native/kernels/`

### External
- **Required**: NumPy (array interop), pybind11 (C++ bindings)
- **Optional**: PyTorch (for `from_torch`/`to_torch`), PyYAML (for YAML config), HuggingFace Accelerate
- **Runtime**: macOS 14+, Apple Silicon (M1/M2/M3/M4), Xcode Command Line Tools

<!-- MANUAL: -->
