# src/kernels/CMakeLists.txt
# Custom Metal kernel implementations

file(GLOB_RECURSE KERNELS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.mm"
)

add_library(metal_native_kernels STATIC ${KERNELS_SOURCES})

target_include_directories(metal_native_kernels
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(metal_native_kernels
    PUBLIC
        metal_native_core
        metal_native_dispatch
        Metal::Metal
)

apply_apple_silicon_flags(metal_native_kernels)

set_target_properties(metal_native_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "metal_native_kernels"
)

install(TARGETS metal_native_kernels
        EXPORT MetalNativeTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
