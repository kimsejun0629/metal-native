# src/future/CMakeLists.txt
# Future hardware feature stubs (Metal 4, BFloat16, Neural Engine, Fast Ops)

set(FUTURE_SOURCES
    future_stubs.mm
)

add_library(metal_native_future STATIC ${FUTURE_SOURCES})

target_include_directories(metal_native_future
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

find_library(FOUNDATION_LIBRARY Foundation REQUIRED)

target_link_libraries(metal_native_future
    PUBLIC
        metal_native_core
        Metal::Metal
        ${FOUNDATION_LIBRARY}
)

apply_apple_silicon_flags(metal_native_future)

set_target_properties(metal_native_future PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "metal_native_future"
)

install(TARGETS metal_native_future
        EXPORT MetalNativeTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
