# MetalShaderCompile.cmake
# Compile Metal shaders (.metal) into metallib bundles

# Function to compile Metal shaders
# Usage: compile_metal_shaders(TARGET target_name SOURCES shader1.metal shader2.metal ...)
function(compile_metal_shaders)
    cmake_parse_arguments(
        SHADER
        ""
        "TARGET"
        "SOURCES;INCLUDE_DIRS"
        ${ARGN}
    )

    if(NOT SHADER_TARGET)
        message(FATAL_ERROR "compile_metal_shaders: TARGET is required")
    endif()

    if(NOT SHADER_SOURCES)
        message(FATAL_ERROR "compile_metal_shaders: SOURCES is required")
    endif()

    # Find xcrun for Metal compilation
    find_program(XCRUN xcrun)
    if(NOT XCRUN)
        message(FATAL_ERROR "xcrun not found - Metal shader compilation requires Xcode command line tools")
    endif()

    set(AIR_FILES "")
    set(METALLIB_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_TARGET}.metallib")

    # Build include-directory flags for the Metal compiler
    set(METAL_INCLUDE_FLAGS "")
    if(SHADER_INCLUDE_DIRS)
        foreach(INC_DIR ${SHADER_INCLUDE_DIRS})
            list(APPEND METAL_INCLUDE_FLAGS "-I" "${INC_DIR}")
        endforeach()
    endif()

    # Compile each .metal file to .air
    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
        get_filename_component(SHADER_ABS ${SHADER_SOURCE} ABSOLUTE)

        set(AIR_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.air")
        list(APPEND AIR_FILES ${AIR_FILE})

        add_custom_command(
            OUTPUT ${AIR_FILE}
            COMMAND ${XCRUN} -sdk macosx metal
                -c ${SHADER_ABS}
                -o ${AIR_FILE}
                -std=metal3.0
                -ffast-math
                -O3
                ${METAL_INCLUDE_FLAGS}
            DEPENDS ${SHADER_ABS}
            COMMENT "Compiling Metal shader ${SHADER_NAME}.metal to AIR"
            VERBATIM
        )
    endforeach()

    # Link all .air files into a single .metallib
    add_custom_command(
        OUTPUT ${METALLIB_FILE}
        COMMAND ${XCRUN} -sdk macosx metallib
            ${AIR_FILES}
            -o ${METALLIB_FILE}
        DEPENDS ${AIR_FILES}
        COMMENT "Linking Metal library ${SHADER_TARGET}.metallib"
        VERBATIM
    )

    # Create a custom target for the metallib
    add_custom_target(${SHADER_TARGET}_shaders ALL
        DEPENDS ${METALLIB_FILE}
    )

    # Set target property with metallib path for use by other targets
    set_target_properties(${SHADER_TARGET}_shaders PROPERTIES
        METALLIB_PATH ${METALLIB_FILE}
    )

    # Install the metallib
    install(FILES ${METALLIB_FILE}
            DESTINATION lib/metal_native/shaders)

endfunction()
