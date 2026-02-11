# benchmarks/CMakeLists.txt
# Performance benchmarks for MetalNative

# Find or fetch Google Benchmark
find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

# Benchmark executable macro
macro(add_metal_native_benchmark benchmark_name)
    add_executable(${benchmark_name} ${ARGN})

    target_link_libraries(${benchmark_name}
        PRIVATE
            metal_native_core
            metal_native_dispatch
            metal_native_kernels
            metal_native_graph
            metal_native_ops
            metal_native_interop
            metal_native_profiling
            benchmark::benchmark
            benchmark::benchmark_main
    )

    apply_apple_silicon_flags(${benchmark_name})

    # Disable sanitizers for benchmarks (they add overhead)
    target_compile_options(${benchmark_name} PRIVATE
        $<$<CONFIG:Debug>:-fno-sanitize=address>
        $<$<CONFIG:Debug>:-fno-sanitize=undefined>
    )
    target_link_options(${benchmark_name} PRIVATE
        $<$<CONFIG:Debug>:-fno-sanitize=address>
        $<$<CONFIG:Debug>:-fno-sanitize=undefined>
    )

    # Install benchmarks
    install(TARGETS ${benchmark_name}
            RUNTIME DESTINATION bin/benchmarks)
endmacro()

# Add benchmark subdirectories if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ops")
    add_subdirectory(ops)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
    add_subdirectory(kernels)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/graph")
    add_subdirectory(graph)
endif()

# Example benchmark if no specific benchmarks exist yet
file(GLOB BENCHMARK_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
if(BENCHMARK_SOURCES)
    add_metal_native_benchmark(metal_native_benchmarks ${BENCHMARK_SOURCES})
endif()
