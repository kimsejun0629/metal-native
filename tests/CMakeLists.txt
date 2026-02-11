# tests/CMakeLists.txt
# Unit tests for MetalNative

# Find or fetch Google Test
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "GTest not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Test executable macro
macro(add_metal_native_test test_name)
    add_executable(${test_name} ${ARGN})

    target_link_libraries(${test_name}
        PRIVATE
            metal_native_core
            metal_native_dispatch
            metal_native_kernels
            metal_native_graph
            metal_native_ops
            metal_native_interop
            metal_native_profiling
            GTest::gtest_main
    )

    apply_apple_silicon_flags(${test_name})

    # Add to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})

    # Set test properties
    set_tests_properties(${test_name} PROPERTIES
        TIMEOUT 300
        LABELS "unit"
    )
endmacro()

# Add test subdirectories if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core")
    add_subdirectory(core)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dispatch")
    add_subdirectory(dispatch)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
    add_subdirectory(kernels)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/graph")
    add_subdirectory(graph)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ops")
    add_subdirectory(ops)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/interop")
    add_subdirectory(interop)
endif()

# Core-only tests (shape, dtype, error) - no ObjC framework dependencies
file(GLOB CORE_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_shape.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/test_dtype.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/test_error.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/cpp/test_shape.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/cpp/test_dtype.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/cpp/test_error.cpp")
if(CORE_TEST_SOURCES)
    add_executable(metal_native_core_tests ${CORE_TEST_SOURCES})
    target_link_libraries(metal_native_core_tests
        PRIVATE
            metal_native_core
            GTest::gtest_main
    )
    apply_apple_silicon_flags(metal_native_core_tests)
    add_test(NAME metal_native_core_tests COMMAND metal_native_core_tests)
    set_tests_properties(metal_native_core_tests PROPERTIES
        TIMEOUT 300
        LABELS "unit"
    )
endif()

# Extended C++ tests (buffer, allocator, kernel_cache) - require Metal frameworks
file(GLOB EXTENDED_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp/test_buffer.cpp"
                                "${CMAKE_CURRENT_SOURCE_DIR}/cpp/test_allocator.cpp"
                                "${CMAKE_CURRENT_SOURCE_DIR}/cpp/test_kernel_cache.cpp")
if(EXTENDED_TEST_SOURCES)
    add_executable(metal_native_extended_tests ${EXTENDED_TEST_SOURCES})
    target_link_libraries(metal_native_extended_tests
        PRIVATE
            metal_native_core
            metal_native_kernels
            GTest::gtest_main
    )
    apply_apple_silicon_flags(metal_native_extended_tests)
    add_test(NAME metal_native_extended_tests COMMAND metal_native_extended_tests)
    set_tests_properties(metal_native_extended_tests PROPERTIES
        TIMEOUT 300
        LABELS "unit;metal"
    )
endif()
