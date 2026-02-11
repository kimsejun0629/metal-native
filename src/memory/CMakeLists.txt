# src/memory/CMakeLists.txt
# GPU memory management: MTLHeap manager and smart allocator

file(GLOB_RECURSE MEMORY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.mm"
)

add_library(metal_native_memory STATIC ${MEMORY_SOURCES})

target_include_directories(metal_native_memory
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(metal_native_memory
    PUBLIC
        metal_native_core
        Metal::Metal
        "-framework Foundation"
)

apply_apple_silicon_flags(metal_native_memory)

set_target_properties(metal_native_memory PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "metal_native_memory"
)

install(TARGETS metal_native_memory
        EXPORT MetalNativeTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
