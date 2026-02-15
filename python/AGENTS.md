<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# python
## Purpose
Python package build configuration for MetalNative. Contains setup.py with CMake integration, pyproject.toml for modern Python packaging, and the metal_native/ package directory with Python-side APIs wrapping the C++ extension.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| metal_native/ | Python package source: __init__.py (public API), _C extension import, tensor wrappers, device module, nn module (layers), optim module (optimizers), utils, version |

## Key Files
| File | Description |
|------|-------------|
| setup.py | Setuptools build configuration with custom CMakeBuild class. Invokes CMake to build C++ extension (_C.so), links pybind11 bindings. Handles Debug/Release builds, macOS-specific flags (-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0), parallel compilation. Reads version from metal_native/_version.py. |
| pyproject.toml | PEP 517/518 build system specification. Declares build-time dependencies (setuptools, wheel, pybind11), project metadata, entry points. Enables modern `pip install -e .` editable installs. |

## For AI Agents
### Working In This Directory
- Install package: `cd python && pip install -e .` (editable mode, rebuilds on changes)
- Build only: `python setup.py build_ext --inplace` (faster for dev iteration)
- Clean build: `rm -rf build/ *.egg-info/ && pip install -e .`
- Version management: Edit `metal_native/_version.py`, propagates to setup.py and package
- CMake integration: setup.py calls cmake, respects CMAKE_ARGS env var for customization

### Testing Requirements
- Prerequisites: CMake 3.20+, Xcode CLI tools, pybind11 (auto-installed by pip)
- Build test: `pip install -e . && python -c "import metal_native; print(metal_native.__version__)"`
- Import test: `python -c "import metal_native._C as C; print(C.device.is_available())"`
- Full test: `pytest ../tests/python/test_imports.py ../tests/python/test_device.py`
- Clean test: `pip uninstall metal-native -y && pip install -e . && pytest ../tests/python/`

### Common Patterns
- CMakeExtension class: Custom Extension subclass with sourcedir parameter
- CMakeBuild.build_extension(): Override to call cmake configure + build instead of standard C compiler
- subprocess.check_call(): Run cmake commands, captures output/errors
- Path(__file__).parent: Locate setup.py directory for relative paths
- sys.executable: Pass current Python interpreter to CMake for consistency

## Dependencies
### Internal
- `../CMakeLists.txt`: Root CMake configuration (invoked by setup.py)
- `../bindings/`: Pybind11 C++ source (compiled into _C extension)
- `metal_native/`: Python package code (imports _C extension)

### External
- setuptools: Python packaging (setup(), Extension, build_ext)
- wheel: Binary distribution format
- pybind11: C++/Python binding (found by CMake)
- cmake: Build system (invoked by setup.py)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
