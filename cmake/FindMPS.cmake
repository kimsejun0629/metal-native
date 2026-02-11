# FindMPS.cmake
# Locate MetalPerformanceShaders.framework on macOS/iOS
#
# This module defines:
#  MPS_FOUND - True if MPS framework is found
#  MPS_LIBRARIES - Libraries to link
#  MPS_INCLUDE_DIRS - Include directories

find_library(MPS_LIBRARIES
    NAMES MetalPerformanceShaders
    PATHS
        /System/Library/Frameworks
        /Library/Frameworks
)

if(MPS_LIBRARIES)
    get_filename_component(MPS_FRAMEWORK_DIR ${MPS_LIBRARIES} DIRECTORY)
    set(MPS_INCLUDE_DIRS "${MPS_FRAMEWORK_DIR}/Headers")

    # Verify the include directory exists, fallback if it doesn't
    if(NOT EXISTS "${MPS_INCLUDE_DIRS}")
        # Try alternative SDK paths
        if(EXISTS "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShaders.framework/Headers")
            set(MPS_INCLUDE_DIRS "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShaders.framework/Headers")
        elseif(EXISTS "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShaders.framework")
            set(MPS_INCLUDE_DIRS "")
        endif()
    endif()

    set(MPS_FOUND TRUE)

    if(NOT TARGET MPS::MPS)
        add_library(MPS::MPS INTERFACE IMPORTED)
        if(MPS_INCLUDE_DIRS)
            set_target_properties(MPS::MPS PROPERTIES
                INTERFACE_LINK_LIBRARIES "${MPS_LIBRARIES}"
                INTERFACE_INCLUDE_DIRECTORIES "${MPS_INCLUDE_DIRS}"
            )
        else()
            set_target_properties(MPS::MPS PROPERTIES
                INTERFACE_LINK_LIBRARIES "${MPS_LIBRARIES}"
            )
        endif()
    endif()
else()
    set(MPS_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPS
    REQUIRED_VARS MPS_LIBRARIES MPS_INCLUDE_DIRS
    FOUND_VAR MPS_FOUND
)

mark_as_advanced(MPS_LIBRARIES MPS_INCLUDE_DIRS)
