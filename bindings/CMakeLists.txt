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
)

# Python package files are in ../python/metal_native/
# No need to copy __init__.py from bindings directory
