# src/ops/CMakeLists.txt
# High-level tensor operations

set(OPS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/op_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/fused_matmul.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/conv.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/pooling.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/loss.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/attention.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/softmax.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/normalization.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/embedding.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/reduction.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/mixed_precision.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batch_advisor.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/speculative_buffers.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/speculative.mm
)

add_library(metal_native_ops STATIC ${OPS_SOURCES})

target_include_directories(metal_native_ops
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(metal_native_ops
    PUBLIC
        metal_native_core
        metal_native_graph
        metal_native_dispatch
        MPSGraph::MPSGraph
        Metal::Metal
        "-framework Foundation"
        "-framework Accelerate"
    PRIVATE
        metal_native_memory
)

apply_apple_silicon_flags(metal_native_ops)

set_target_properties(metal_native_ops PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "metal_native_ops"
)

install(TARGETS metal_native_ops
        EXPORT MetalNativeTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
