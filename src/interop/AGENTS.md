<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# interop
## Purpose
Implementation files for framework interoperability. Implements DLPack conversion, PyTorch integration stubs, and HuggingFace Accelerate backend stubs.
## Key Files
| File | Description |
|------|-------------|
| CMakeLists.txt | Build configuration for interop module |
| dlpack.cpp | DLPack exporter/importer implementation (zero-copy tensor conversion) |
| torch_interop.cpp | PyTorch C++ integration stubs (metadata extraction, custom op registration) |
| hf_accelerate.cpp | HuggingFace Accelerate backend stubs (device name, availability checks) |
## For AI Agents
### Working In This Directory
- **dlpack.cpp**: Constructs DLManagedTensor with deleter callback that releases MNBuffer shared_ptr
- **torch_interop.cpp**: Extracts shape/strides/dtype/pointer for Python-side torch.Tensor construction
- **hf_accelerate.cpp**: Provides minimal interface for Accelerate device registration (mostly stubs)
### Common Patterns
```cpp
// DLPack export
DLManagedTensor* to_dlpack(const MNTensor& tensor) {
    auto* dl = new DLManagedTensor;
    dl->dl_tensor.data = tensor.raw_data();
    dl->dl_tensor.device.device_type = kDLMetal;
    dl->dl_tensor.ndim = tensor.ndim();
    // ... fill shape, strides, dtype ...
    dl->deleter = [](DLManagedTensor* self) {
        auto* manager_ctx = static_cast<BufferManager*>(self->manager_ctx);
        delete manager_ctx;  // releases shared_ptr<MNBuffer>
        delete self;
    };
    return dl;
}

// PyTorch metadata
TorchTensorMeta meta;
meta.shape = tensor.shape().dims();
meta.strides = tensor.strides();
meta.dtype_str = dtype_to_torch_string(tensor.dtype());
meta.data_ptr = tensor.raw_data();
```
## Dependencies
### Internal
- ../include/metal_native/interop/ (public headers)
- core/ (MNTensor, MNDevice, MNDType)
### External
- DLPack header (third-party C API)
- PyTorch C++ API (optional, for custom op registration - currently stubbed)
<!-- MANUAL: -->
