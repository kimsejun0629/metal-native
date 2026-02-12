# Pybind11Setup.cmake
# Configure pybind11 for Python bindings

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via CONFIG, attempting to use FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Helper function to create a pybind11 module
# Usage: create_pybind11_module(MODULE_NAME module_name SOURCES src1.cpp src2.cpp LINK_LIBRARIES lib1 lib2)
function(create_pybind11_module)
    cmake_parse_arguments(
        PY_MODULE
        ""
        "MODULE_NAME"
        "SOURCES;LINK_LIBRARIES"
        ${ARGN}
    )

    if(NOT PY_MODULE_MODULE_NAME)
        message(FATAL_ERROR "create_pybind11_module: MODULE_NAME is required")
    endif()

    if(NOT PY_MODULE_SOURCES)
        message(FATAL_ERROR "create_pybind11_module: SOURCES is required")
    endif()

    # Create pybind11 module
    pybind11_add_module(${PY_MODULE_MODULE_NAME} ${PY_MODULE_SOURCES})

    # Link libraries if specified
    if(PY_MODULE_LINK_LIBRARIES)
        target_link_libraries(${PY_MODULE_MODULE_NAME} PRIVATE ${PY_MODULE_LINK_LIBRARIES})
    endif()

    # Apply Apple Silicon flags
    apply_apple_silicon_flags(${PY_MODULE_MODULE_NAME})

    # Set output directory and install destination
    if(SKBUILD_BUILD)
        # scikit-build-core manages install — output to build dir only
        set_target_properties(${PY_MODULE_MODULE_NAME} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        )
        # install is handled in bindings/CMakeLists.txt
    else()
        # Development build: output to python/metal_native/ directly
        set_target_properties(${PY_MODULE_MODULE_NAME} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/metal_native"
        )
        install(TARGETS ${PY_MODULE_MODULE_NAME}
                LIBRARY DESTINATION "${Python3_SITEARCH}/metal_native")
    endif()

endfunction()
