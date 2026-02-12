# shaders/CMakeLists.txt
# Metal shader compilation

# Find all .metal shader files
file(GLOB METAL_SHADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.metal"
)

if(METAL_SHADERS)
    # Compile shaders into metallib, with common/ as an include path.
    # This creates a custom target: metal_native_shaders (${TARGET}_shaders).
    compile_metal_shaders(
        TARGET metal_native
        SOURCES ${METAL_SHADERS}
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    # Create an INTERFACE library consumers can link to get the metallib path.
    # Use a separate name to avoid conflicting with the custom target above.
    add_library(metal_native_shader_lib INTERFACE)
    add_dependencies(metal_native_shader_lib metal_native_shaders)

    set(METALLIB_PATH "${CMAKE_CURRENT_BINARY_DIR}/metal_native.metallib")
    target_compile_definitions(metal_native_shader_lib INTERFACE
        METAL_NATIVE_METALLIB_PATH="${METALLIB_PATH}"
    )

    # Install metallib to appropriate location
    if(SKBUILD_BUILD)
        # Wheel: install to site-packages/metal_native/shaders/
        install(FILES ${METALLIB_PATH}
                DESTINATION metal_native/shaders
                COMPONENT python)
    else()
        # Development mode: install to lib/metal_native/shaders/
        install(FILES ${METALLIB_PATH}
                DESTINATION lib/metal_native/shaders)
    endif()

else()
    message(WARNING "No Metal shader files found in ${CMAKE_CURRENT_SOURCE_DIR}")
    add_library(metal_native_shader_lib INTERFACE)
endif()

install(TARGETS metal_native_shader_lib
        EXPORT MetalNativeTargets)
