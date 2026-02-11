# src/dispatch/CMakeLists.txt
# Kernel dispatch and execution system

file(GLOB_RECURSE DISPATCH_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.mm"
)

add_library(metal_native_dispatch STATIC ${DISPATCH_SOURCES})

target_include_directories(metal_native_dispatch
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(metal_native_dispatch
    PUBLIC
        metal_native_core
        Metal::Metal
        "-framework Foundation"
)

apply_apple_silicon_flags(metal_native_dispatch)

set_target_properties(metal_native_dispatch PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "metal_native_dispatch"
)

install(TARGETS metal_native_dispatch
        EXPORT MetalNativeTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
