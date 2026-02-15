<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->

# metal_native

## Purpose
Core framework directory for MetalNative. Contains the full C++/ObjC++ backend, Metal shader kernels, Python bindings, build system, benchmarks, examples, and tests. The project is structured as a CMake project with pybind11 Python bindings.

## Key Files

| File | Description |
|------|-------------|
| `CMakeLists.txt` | Root CMake build configuration — defines build options, finds frameworks, adds subdirectories in dependency order |
| `pyproject.toml` | Python package metadata, tool configs (ruff, mypy, pytest, coverage) |
| `README.md` | Project overview with performance benchmarks (bilingual EN/KR) |
| `.clang-format` | C++ code formatting rules |
| `.clang-tidy` | C++ static analysis configuration |
| `.pre-commit-config.yaml` | Pre-commit hooks for code quality |
| `OPTIMIZATION_PLAN.md` | Optimization roadmap |
| `PHASE_0_2_FIX_SUMMARY.md` | Phase 0.2 fix summary |
| `PHASE_0_2_REGRESSION_ANALYSIS.md` | Phase 0.2 regression analysis |
| `benchmark_modern_models.py` | Model benchmark runner |
| `benchmark_*.json` | Benchmark result snapshots |
| `test_vec4_softmax.cpp` | Standalone vec4 softmax test |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `include/` | Public C++ headers organized by module (see `include/metal_native/AGENTS.md`) |
| `src/` | C++/ObjC++ implementation files organized by module |
| `shaders/` | Metal Shading Language (.metal) kernel files (see `shaders/AGENTS.md`) |
| `bindings/` | pybind11 C++→Python bindings (see `bindings/AGENTS.md`) |
| `python/` | Python package source (see `python/AGENTS.md`) |
| `tests/` | C++ and Python test suites (see `tests/AGENTS.md`) |
| `benchmarks/` | Performance benchmark scripts (see `benchmarks/AGENTS.md`) |
| `examples/` | Usage examples: HuggingFace, LLM, MNIST, profiling (see `examples/AGENTS.md`) |
| `cmake/` | Custom CMake modules for Apple frameworks (see `cmake/AGENTS.md`) |
| `scripts/` | Build, test, lint, benchmark shell scripts (see `scripts/AGENTS.md`) |
| `docs/` | Documentation and images (see `docs/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Build with: `mkdir build && cd build && cmake .. && make -j$(sysctl -n hw.ncpu)`
- Python install: `pip install -e .` (from this directory)
- The CMake build adds subdirectories in dependency order: core → memory → dispatch → kernels → graph → ops → interop → profiling → future → shaders → bindings
- Headers are in `include/metal_native/`, implementations in `src/`
- ObjC++ files (`.mm`) are used for Metal/MPS API calls; pure C++ (`.cpp`) for platform-independent logic

### Testing Requirements
- C++ tests: `cd build && ctest` (uses GoogleTest)
- Python tests: `pytest tests/python/` (from this directory)
- Integration tests: `pytest tests/integration/` (requires built C extension)
- Benchmarks: `python benchmarks/run_all.py`

### Common Patterns
- C++17 with ObjC++ for Apple framework interop
- RAII wrappers around Metal/MPS objects
- Pybind11 for Python bindings
- Fused kernel operations combine multiple GPU dispatches into single kernels
- Lazy graph execution batches GPU commands

## Dependencies

### External
- Metal 3.0+ framework (Apple GPU API)
- MetalPerformanceShaders (MPS) framework
- MetalPerformanceShadersGraph (MPSGraph) framework
- Accelerate framework (Apple BLAS/LAPACK)
- pybind11 >= 2.10 (C++ Python bindings)
- numpy >= 1.24.0 (Python array interop)
- CMake >= 3.25

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
