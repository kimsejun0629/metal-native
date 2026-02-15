# src/inference/CMakeLists.txt
# Inference pipeline components

add_library(metal_native_inference
    phase_pipeline.mm
)

target_link_libraries(metal_native_inference
    PUBLIC
        metal_native_core
        metal_native_dispatch
    PRIVATE
        ${Metal_LIBRARIES}
)

target_include_directories(metal_native_inference
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

apply_apple_silicon_flags(metal_native_inference)

# Install rules
install(TARGETS metal_native_inference
    EXPORT MetalNativeTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
