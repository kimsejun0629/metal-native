# src/core/CMakeLists.txt
# Core runtime and device management

file(GLOB_RECURSE CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.mm"
)

add_library(metal_native_core STATIC ${CORE_SOURCES})

target_include_directories(metal_native_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

find_library(FOUNDATION_LIBRARY Foundation REQUIRED)

target_link_libraries(metal_native_core
    PUBLIC
        Metal::Metal
        ${FOUNDATION_LIBRARY}
    PRIVATE
        metal_native_dispatch
        metal_native_graph
)

apply_apple_silicon_flags(metal_native_core)

set_target_properties(metal_native_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "metal_native_core"
)

install(TARGETS metal_native_core
        EXPORT MetalNativeTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
