# FindMetal.cmake
# Locate Metal.framework on macOS/iOS
#
# This module defines:
#  Metal_FOUND - True if Metal framework is found
#  Metal_LIBRARIES - Libraries to link
#  Metal_INCLUDE_DIRS - Include directories

find_library(Metal_LIBRARIES
    NAMES Metal
    PATHS
        /System/Library/Frameworks
        /Library/Frameworks
)

if(Metal_LIBRARIES)
    get_filename_component(Metal_FRAMEWORK_DIR ${Metal_LIBRARIES} DIRECTORY)
    set(Metal_INCLUDE_DIRS "${Metal_FRAMEWORK_DIR}/Headers")

    # Verify the include directory exists, fallback if it doesn't
    if(NOT EXISTS "${Metal_INCLUDE_DIRS}")
        # Try alternative SDK paths
        if(EXISTS "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Metal.framework/Headers")
            set(Metal_INCLUDE_DIRS "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Metal.framework/Headers")
        elseif(EXISTS "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Metal.framework")
            set(Metal_INCLUDE_DIRS "")
        endif()
    endif()

    set(Metal_FOUND TRUE)

    if(NOT TARGET Metal::Metal)
        add_library(Metal::Metal INTERFACE IMPORTED)
        if(Metal_INCLUDE_DIRS)
            set_target_properties(Metal::Metal PROPERTIES
                INTERFACE_LINK_LIBRARIES "${Metal_LIBRARIES}"
                INTERFACE_INCLUDE_DIRECTORIES "${Metal_INCLUDE_DIRS}"
            )
        else()
            set_target_properties(Metal::Metal PROPERTIES
                INTERFACE_LINK_LIBRARIES "${Metal_LIBRARIES}"
            )
        endif()
    endif()
else()
    set(Metal_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Metal
    REQUIRED_VARS Metal_LIBRARIES Metal_INCLUDE_DIRS
    FOUND_VAR Metal_FOUND
)

mark_as_advanced(Metal_LIBRARIES Metal_INCLUDE_DIRS)
