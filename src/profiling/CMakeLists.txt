# src/profiling/CMakeLists.txt
# Performance profiling and tracing utilities

file(GLOB_RECURSE PROFILING_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.mm"
)

add_library(metal_native_profiling STATIC ${PROFILING_SOURCES})

target_include_directories(metal_native_profiling
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(metal_native_profiling
    PUBLIC
        metal_native_core
        Metal::Metal
        "-framework Foundation"
)

apply_apple_silicon_flags(metal_native_profiling)

set_target_properties(metal_native_profiling PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "metal_native_profiling"
)

install(TARGETS metal_native_profiling
        EXPORT MetalNativeTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
