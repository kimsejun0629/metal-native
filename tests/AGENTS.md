<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# tests
## Purpose
Comprehensive test suite covering C++ core functionality (Google Test), Python API (pytest), and integration tests (PyTorch interop, memory management, model inference). Three-tier test structure: cpp/ (unit), python/ (API), integration/ (end-to-end).

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| cpp/ | C++ unit tests using Google Test: allocator, buffer, dtype, error, shape, kernel_cache, kernel_registry. Low-level core functionality validation. |
| python/ | Python API tests using pytest: imports, device, dtypes, tensor, interop (DLPack/NumPy/PyTorch), nn layers, optimizers, profiling, config, utils, version. Tests Python bindings and high-level APIs. |
| integration/ | End-to-end integration tests: tensor_ops (operation chains), memory (allocation/deallocation), model_inference (full model forward passes). Cross-language and multi-component validation. |

## Key Files (Root)
| File | Description |
|------|-------------|
| CMakeLists.txt | C++ test build configuration: links Google Test, creates test executables, registers with CTest. Builds all cpp/*.cpp into test binaries. |
| test_dtype.cpp | DType enum tests: size calculations, name lookups, floating point checks, signed/unsigned validation. |
| test_error.cpp | Error handling tests: MNException construction, error messages, METAL_NATIVE_CHECK macro, exception propagation. |
| test_shape.cpp | Shape class tests: construction, indexing, broadcasting, stride calculations, contiguity checks. |
| test_numerical_accuracy.py | Reference implementation validators: pure NumPy versions of softmax, layer_norm, rms_norm, attention. Numerical accuracy testing framework with configurable tolerances. Can run standalone without metal_native (reference-only mode). |

## For AI Agents
### Working In This Directory
- Run all tests: `cd ../build && ctest` (C++) and `pytest .` (Python, from tests/ root)
- Run specific suite: `ctest -R test_dtype` (C++), `pytest python/test_device.py` (Python)
- Run with output: `ctest --output-on-failure`, `pytest -v -s`
- C++ tests are executables in ../build/tests/: `./build/tests/test_dtype`
- Python tests use pytest fixtures from conftest.py (each subdirectory)

### Testing Requirements
- C++ tests: Requires Google Test (auto-downloaded by CMake), built C++ libraries
- Python tests: Requires pytest, built _C extension, optional: torch (for interop tests)
- Integration tests: Requires torch, numpy, built metal_native package
- Run from project root: `./scripts/test.sh` (builds if needed, runs all tests)
- CI/CD: `cmake --build build && cd build && ctest && cd ../tests && pytest`

### Common Patterns
- C++ test structure: `TEST(Category, TestName) { ASSERT_EQ(actual, expected); }`
- Python test structure: `def test_feature(): assert actual == expected`
- Fixtures: `@pytest.fixture` in conftest.py for shared setup/teardown
- Parametrized tests: `@pytest.mark.parametrize("param", [val1, val2])` for data-driven tests
- Skipping tests: `@pytest.mark.skipif(not has_torch, reason="...")` for optional deps
- Numerical accuracy: `np.allclose(actual, expected, rtol=1e-5, atol=1e-8)` for floating point

## Dependencies
### Internal
- `../src/`: C++ core implementation (tested by cpp/ tests)
- `../python/metal_native/`: Python package (tested by python/ tests)
- `../build/`: Built binaries and extensions

### External
- Google Test: C++ test framework (cpp/ tests)
- pytest: Python test framework (python/ tests)
- numpy: Reference implementations and numerical checks
- torch: Optional for PyTorch interop tests (python/test_interop.py, integration/)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
