cmake_minimum_required(VERSION 3.25)
project(metal_native VERSION 0.1.0 LANGUAGES CXX OBJCXX)

# scikit-build-core detection
if(DEFINED SKBUILD)
    message(STATUS "Building via scikit-build-core (wheel mode)")
    set(SKBUILD_BUILD ON)
else()
    set(SKBUILD_BUILD OFF)
endif()

# C++17 standard required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ObjC++ support
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Module path for custom Find*.cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Build options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_SHADERS "Compile Metal shaders (requires Metal toolchain)" ON)

# Include CMake modules
include(AppleSiliconFlags)
include(MetalShaderCompile)

# Find required frameworks
find_package(Metal REQUIRED)
find_package(MPS REQUIRED)
find_package(MPSGraph REQUIRED)
find_package(Accelerate REQUIRED)

# Optional Python bindings
if(BUILD_PYTHON)
    include(Pybind11Setup)
endif()

# Include directories for all targets
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Metal_INCLUDE_DIRS}
)

# Add subdirectories in dependency order
add_subdirectory(src/core)
add_subdirectory(src/memory)
add_subdirectory(src/dispatch)
add_subdirectory(src/kernels)
add_subdirectory(src/graph)
add_subdirectory(src/ops)
add_subdirectory(src/interop)
add_subdirectory(src/profiling)
add_subdirectory(src/future)
add_subdirectory(src/inference)
if(BUILD_SHADERS)
    add_subdirectory(shaders)
endif()

# Python bindings (depends on all other modules)
if(BUILD_PYTHON)
    add_subdirectory(bindings)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install configuration
install(DIRECTORY include/metal_native
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MetalNativeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(EXPORT MetalNativeTargets
        FILE MetalNativeTargets.cmake
        NAMESPACE MetalNative::
        DESTINATION lib/cmake/MetalNative)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MetalNativeConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/MetalNativeConfig.cmake"
    INSTALL_DESTINATION lib/cmake/MetalNative
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MetalNativeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MetalNativeConfigVersion.cmake"
    DESTINATION lib/cmake/MetalNative)
