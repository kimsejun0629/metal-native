<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# integration

## Purpose
End-to-end integration tests for metal_native verifying memory management, tensor operations, and model inference patterns on real Metal hardware.

## Key Files
| File | Description |
|------|-------------|
| `conftest.py` | Pytest fixtures and `@requires_metal` decorator for hardware-dependent tests |
| `test_memory.py` | Memory allocation tracking, cache management, peak stats, synchronization |
| `test_tensor_ops.py` | Tensor creation, arithmetic operations, NumPy interop, dtype conversions |
| `test_model_inference.py` | Linear layers, two-layer networks, batch matmul, attention mechanisms, numerical stability |
| `__init__.py` | Package marker |

## For AI Agents

### Working In This Directory
- **Hardware Required**: Tests skip automatically if Metal not available (CI safety)
- **Pytest Markers**: Use `@pytest.mark.integration` and `@requires_metal` decorators
- **Real Operations**: Tests exercise full Python → C++ → Metal → GPU stack
- **Numerical Tolerance**: Use `np.allclose(atol=1e-5)` for float32, `atol=1e-2` for float16

### Testing Requirements
- Run with: `pytest tests/integration/ -v -m integration`
- Requires: macOS 14+, Apple Silicon, metal_native installed with C extension
- Memory tests use `mn.reset_peak_stats()` and `mn.synchronize()` for accurate measurements
- Model tests verify no NaN/Inf in outputs and check reasonable value ranges

### Common Patterns
- **Setup**: Import `metal_native as mn` inside test (not module level) to avoid import errors
- **Memory Cleanup**: Call `mn.empty_cache()` after large allocations to free GPU memory
- **Synchronization**: Use `mn.synchronize()` before checking memory stats or reading results
- **Numerical Checks**: Verify mean/std for random tensors, exact values for deterministic ops
- **Stability Tests**: Scale inputs (`* 0.01`) to prevent overflow in large matrix multiplies

## Dependencies

### Internal
- `metal_native`: Full package with C extension
- `conftest.py`: Provides `requires_metal` decorator

### External
- **Required**: pytest, numpy, metal_native (installed)
- **Optional**: PyTorch (for interop verification)
- **Runtime**: Metal-capable hardware

<!-- MANUAL: -->
