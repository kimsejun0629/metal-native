# bindings/CMakeLists.txt
# Python bindings via pybind11

if(NOT BUILD_PYTHON)
    return()
endif()

# Binding source files
set(BINDING_SOURCES
    bind_core.cpp
    bind_ops.cpp
    bind_profiling.cpp
    bind_interop.cpp
)

# Create Python module (_C extension)
create_pybind11_module(
    MODULE_NAME _C
    SOURCES ${BINDING_SOURCES}
    LINK_LIBRARIES
        metal_native_core
        metal_native_dispatch
        metal_native_kernels
        metal_native_graph
        metal_native_ops
        metal_native_interop
        metal_native_profiling
        metal_native_future
        metal_native_memory
)

# Link shader library for metallib path definition
if(TARGET metal_native_shader_lib)
    target_link_libraries(_C PRIVATE metal_native_shader_lib)
endif()

# Install for wheel builds
if(SKBUILD_BUILD)
    install(TARGETS _C
            LIBRARY DESTINATION metal_native
            COMPONENT python)
endif()
