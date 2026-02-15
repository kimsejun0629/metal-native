<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# bindings
## Purpose
Pybind11 bindings exposing C++ MetalNative functionality to Python. Provides the `_C` extension module with device management, tensor operations, fused kernels (fast::), profiling, and PyTorch/NumPy interop. Main entry point for Python → Metal bridge.

## Key Files
| File | Description |
|------|-------------|
| bind_core.cpp | Core bindings: MNTensor class, MNDType enum, device queries (name, is_available, supports_bfloat16), initialization (initialize, load_library), synchronization, memory management (empty_cache, memory_allocated), RNG (set_seed). Defines PYBIND11_MODULE(_C). Includes placeholder stubs for tensor operations to be implemented. |
| bind_ops.cpp | Operation bindings: matmul, batched_matmul, softmax, log_softmax, flash_attention. fast:: submodule with rms_norm, layer_norm, rope, swiglu, fused_bias_gelu, fused_residual_norm, fused_qkv_projection, scaled_dot_product_attention. All with Python docstrings. |
| bind_profiling.cpp | Profiling bindings: TraceContext, performance counters, memory snapshots, operation timing. Exposes profiling:: namespace to Python. |
| bind_interop.cpp | Interoperability bindings: DLPack export/import (tensor_to_dlpack, tensor_from_dlpack), PyTorch MPS zero-copy (tensor_from_mps_ptr), NumPy conversion. Enables seamless integration with PyTorch and NumPy. |
| CMakeLists.txt | Build configuration for pybind11 extension. Links metal_native_core, adds pybind11 include paths, sets target properties for Python module. |

## For AI Agents
### Working In This Directory
- Each bind_*.cpp file implements `void bind_*(py::module_& m)` called from bind_core.cpp
- Use pybind11 conventions: `py::arg("name")` for named arguments, docstrings as final string param
- MNTensor is exposed as shared_ptr: `py::class_<MNTensor, std::shared_ptr<MNTensor>>`
- Lambda wrappers handle Python object → C++ conversions (e.g., py::none() → nullptr)
- Register exception translation: `py::register_exception<MNException>(m, "MetalNativeError")`
- Stub implementations throw `std::runtime_error("not yet implemented")` with phase marker

### Testing Requirements
- Build: `cd ../build && cmake --build . --target _C`
- Test import: `python3 -c "import metal_native._C as C; print(C.__version__)"`
- Test device: `python3 -c "import metal_native._C as C; print(C.device.is_available())"`
- Full test: `pytest ../tests/python/test_imports.py`

### Common Patterns
- Submodule creation: `auto fast_m = m.def_submodule("fast", "Fused Metal operations");`
- Optional parameters: `py::arg("mask") = py::none()`
- Enum binding: `py::enum_<MNDType>(m, "DType").value("Float32", MNDType::Float32).export_values();`
- Lambda for type conversion: `[](const MNTensor& t) { return dtype_name(t.dtype()); }`
- Docstrings use triple-quoted format with Args/Returns sections

## Dependencies
### Internal
- `../include/metal_native/core/`: dtype.h, error.h, device.h, tensor.h
- `../include/metal_native/ops/`: matmul.h, attention.h, softmax.h
- `../include/metal_native/future/`: fast_ops.h
- `../include/metal_native/kernels/`: kernel_registry.h
- `../include/metal_native/dispatch/`: command_pipeline.h

### External
- pybind11: C++/Python binding library (header-only)
- Python development headers (Python.h)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
