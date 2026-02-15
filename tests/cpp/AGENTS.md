<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# cpp

## Purpose
C++ unit tests for core MetalNative infrastructure using Google Test. Tests memory allocation, buffer management, type system, error handling, and kernel caching without requiring Python bindings.

## Key Files
| File | Description |
|------|-------------|
| `test_allocator.cpp` | MetalSmartAllocator tests: allocation/deallocation, cache reuse, size classes, LRU eviction, stats tracking |
| `test_buffer.cpp` | MNBuffer tests: shared/private buffers, GPU addresses, move semantics, data pointer access |
| `test_dtype.cpp` | MNDType tests: size/name utilities, floating-point/integer/signed checks, Metal format conversion |
| `test_error.cpp` | Error handling tests: MNException construction, MN_CHECK/MN_THROW macros, error code strings |
| `test_kernel_cache.cpp` | KernelCache tests: LRU eviction, hit rate tracking, pipeline state caching by key |
| `test_kernel_registry.cpp` | KernelRegistry tests: singleton access, kernel registration, precompilation coordination |
| `test_shape.cpp` | MNShape tests: construction, numel calculation, contiguous strides, broadcasting rules, negative indexing |

## For AI Agents

### Working In This Directory
- **Google Test**: Use `TEST(SuiteName, TestName)` or `TEST_F(Fixture, TestName)` macros
- **Metal Availability**: Tests use `GTEST_SKIP()` if Metal device not found (CI/non-Mac environments)
- **Fixture Pattern**: `SetUp()` initializes device/allocator, `TearDown()` cleans up resources
- **Assertions**: `EXPECT_*` for non-fatal checks, `ASSERT_*` when test cannot continue on failure

### Testing Requirements
- Tests must compile with C++17 standard
- Link against gtest, gtest_main, and metal_native core libraries
- Run via CMake/CTest or directly: `./test_allocator`
- Mock Metal device for CI environments without GPU

### Common Patterns
- **Device Check**: `GTEST_SKIP()` early if `device_->metal_device() == nullptr`
- **Resource Cleanup**: Use RAII or explicit cleanup in `TearDown()` to prevent leaks
- **Memory Tests**: Track allocator stats before/after operations to verify behavior
- **Move Semantics**: Test both move constructor and move assignment for GPU resources
- **Error Testing**: Use `EXPECT_THROW(expr, ExceptionType)` for exception validation

## Dependencies

### Internal
- `metal_native/core/`: device, buffer, dtype, error, shape
- `metal_native/memory/`: allocator
- `metal_native/kernels/`: kernel_cache, kernel_registry

### External
- **Required**: Google Test (gtest), Metal framework
- **Build**: CMake 3.15+, C++17 compiler (Clang on macOS)

<!-- MANUAL: -->
