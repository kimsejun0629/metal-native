<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# interop
## Purpose
Framework interoperability layer. Provides DLPack zero-copy tensor exchange, PyTorch C++ integration stubs, and HuggingFace Accelerate backend hooks.
## Key Files
| File | Description |
|------|-------------|
| dlpack.h | DLPackExporter/Importer: convert MNTensor to/from DLManagedTensor |
| dlpack_header.h | DLPack C API header (third-party protocol definition) |
| torch_interop.h | TorchInterop: PyTorch C++ integration stubs for custom op registration |
| hf_accelerate.h | AccelerateBackend: HuggingFace Accelerate integration stubs |
## For AI Agents
### Working In This Directory
- **DLPack** enables zero-copy tensor exchange with PyTorch, JAX, TensorFlow, etc.
- **TorchInterop** prepares metadata for Python-side torch.Tensor conversion (via pybind11)
- **AccelerateBackend** provides hooks for HuggingFace Accelerate device placement
- All interop layers are stubs/scaffolding until Python bindings are implemented
### Common Patterns
```cpp
// Export to DLPack
MNTensor tensor = MNTensor::zeros({2, 3}, MNDType::Float32, device);
DLManagedTensor* dl_tensor = DLPackExporter::to_dlpack(tensor);
// ... pass to PyTorch/JAX ...
dl_tensor->deleter(dl_tensor);  // release

// Import from DLPack
MNTensor imported = DLPackImporter::from_dlpack(dl_tensor);

// PyTorch metadata (for Python binding layer)
TorchTensorMeta meta = TorchInterop::tensor_to_torch_metadata(tensor);
// Python: torch.as_tensor(meta.data_ptr, shape=meta.shape, dtype=meta.dtype_str)

// Accelerate backend
if (AccelerateBackend::is_available()) {
    AccelerateBackend::synchronize();
}
```
## Dependencies
### Internal
- core/ (MNTensor, MNDevice, MNDType)
### External
- DLPack header (third-party C API)
- PyTorch C++ API (optional, loaded dynamically)
<!-- MANUAL: -->
