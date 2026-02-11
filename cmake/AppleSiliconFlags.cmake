# AppleSiliconFlags.cmake
# Apply Apple Silicon specific compiler flags and optimizations

# Function to apply Apple Silicon optimizations to a target
function(apply_apple_silicon_flags target)
    # Target Apple Silicon baseline architecture (A14 covers M1/M2/M3/M4)
    target_compile_options(${target} PRIVATE
        -mcpu=apple-a14
    )

    # Enable ARC for Objective-C++ files
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
    )

    # Link-time optimization for Release builds
    target_compile_options(${target} PRIVATE
        $<$<CONFIG:Release>:-flto=thin>
    )
    target_link_options(${target} PRIVATE
        $<$<CONFIG:Release>:-flto=thin>
    )

    # Sanitizers for Debug builds
    target_compile_options(${target} PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address>
        $<$<CONFIG:Debug>:-fsanitize=undefined>
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
    )
    target_link_options(${target} PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address>
        $<$<CONFIG:Debug>:-fsanitize=undefined>
    )

    # Warning flags
    target_compile_options(${target} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
    )

    # Additional optimization flags for Release
    target_compile_options(${target} PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-DNDEBUG>
    )

    # Debug flags
    target_compile_options(${target} PRIVATE
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Debug>:-DDEBUG>
    )
endfunction()
