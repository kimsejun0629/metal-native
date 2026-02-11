# src/interop/CMakeLists.txt
# PyTorch and framework interoperability

set(INTEROP_SOURCES
    dlpack.cpp
    torch_interop.cpp
    hf_accelerate.cpp
)

add_library(metal_native_interop STATIC ${INTEROP_SOURCES})

target_include_directories(metal_native_interop
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(metal_native_interop
    PUBLIC
        metal_native_core
        metal_native_ops
        Metal::Metal
)

apply_apple_silicon_flags(metal_native_interop)

set_target_properties(metal_native_interop PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "metal_native_interop"
)

install(TARGETS metal_native_interop
        EXPORT MetalNativeTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
