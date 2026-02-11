# src/graph/CMakeLists.txt
# MPSGraph integration and graph compilation

set(GRAPH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_builder.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/fusion_patterns.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/shape_bucketing.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_cache.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_serializer.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/lazy_graph.mm
)

add_library(metal_native_graph STATIC ${GRAPH_SOURCES})

target_include_directories(metal_native_graph
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(metal_native_graph
    PUBLIC
        metal_native_core
        metal_native_dispatch
        Metal::Metal
        MPSGraph::MPSGraph
        "-framework Foundation"
)

apply_apple_silicon_flags(metal_native_graph)

set_target_properties(metal_native_graph PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "metal_native_graph"
)

install(TARGETS metal_native_graph
        EXPORT MetalNativeTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
