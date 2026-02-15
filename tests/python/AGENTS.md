<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# python

## Purpose
Python unit tests for metal_native package API surface, configuration, device queries, and data types. Uses mocking to test without requiring C extension or Metal hardware.

## Key Files
| File | Description |
|------|-------------|
| `conftest.py` | Pytest configuration and shared fixtures |
| `test_imports.py` | Module structure, __all__ exports, lazy loading, version consistency |
| `test_config.py` | Configuration dataclasses, JSON/YAML loading, environment variable parsing |
| `test_device.py` | Device query functions (mocked), Device context manager |
| `test_dtypes.py` | DType objects, NumPy conversions, equality/hashing, get_dtype() parsing |
| `test_tensor.py` | Tensor class properties, factory function signatures, __all__ exports |
| `test_utils.py` | Utils functions (mocked), SynchronizeContext, memory info formatting |
| `test_nn.py` | Neural network module structure (placeholder verification) |
| `test_optim.py` | Optimizer parameter validation, state management |
| `test_profiling.py` | Profiling API functions (mocked) |
| `test_interop.py` | PyTorch/NumPy conversion logic (mocked) |
| `test_version.py` | Version string format and consistency |
| `__init__.py` | Package marker |

## For AI Agents

### Working In This Directory
- **Mocking Strategy**: Use `unittest.mock.patch` to mock `metal_native._C` extension
- **No Hardware**: Tests run on any system (CI-friendly) without Metal or C extension
- **Module Reloading**: Some tests delete/reload modules to test import mechanics
- **Signature Testing**: Use `inspect.signature()` to verify function parameter names/annotations

### Testing Requirements
- Run with: `pytest tests/python/ -v`
- No dependencies on C extension or Metal hardware
- Mock `_C` methods to return appropriate types (int for memory, str for device name, etc.)
- Test both success paths (mocked C extension) and error paths (C extension unavailable)

### Common Patterns
- **Mock Setup**: `with patch.object(metal_native, '_C', mock_c):`
- **Module Reload**: `if 'metal_native.device' in sys.modules: del sys.modules['metal_native.device']`
- **Error Testing**: Verify RuntimeError with appropriate message when C extension missing
- **Config Testing**: Use `tempfile.NamedTemporaryFile` for JSON/YAML config file tests
- **All Exports**: Verify `set(module.__all__) == set(expected)` for public API

## Dependencies

### Internal
- All `metal_native` submodules (imported but mocked)

### External
- **Required**: pytest, numpy, unittest.mock
- **Optional**: PyTorch (for some import tests, but mocked)

<!-- MANUAL: -->
