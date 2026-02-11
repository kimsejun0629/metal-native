# FindMPSGraph.cmake
# Locate MetalPerformanceShadersGraph.framework on macOS/iOS
#
# This module defines:
#  MPSGraph_FOUND - True if MPSGraph framework is found
#  MPSGraph_LIBRARIES - Libraries to link
#  MPSGraph_INCLUDE_DIRS - Include directories

find_library(MPSGraph_LIBRARIES
    NAMES MetalPerformanceShadersGraph
    PATHS
        /System/Library/Frameworks
        /Library/Frameworks
)

if(MPSGraph_LIBRARIES)
    get_filename_component(MPSGraph_FRAMEWORK_DIR ${MPSGraph_LIBRARIES} DIRECTORY)
    set(MPSGraph_INCLUDE_DIRS "${MPSGraph_FRAMEWORK_DIR}/Headers")

    # Verify the include directory exists, fallback if it doesn't
    if(NOT EXISTS "${MPSGraph_INCLUDE_DIRS}")
        # Try alternative SDK paths
        if(EXISTS "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Headers")
            set(MPSGraph_INCLUDE_DIRS "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Headers")
        elseif(EXISTS "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShadersGraph.framework")
            set(MPSGraph_INCLUDE_DIRS "")
        endif()
    endif()

    set(MPSGraph_FOUND TRUE)

    if(NOT TARGET MPSGraph::MPSGraph)
        add_library(MPSGraph::MPSGraph INTERFACE IMPORTED)
        if(MPSGraph_INCLUDE_DIRS)
            set_target_properties(MPSGraph::MPSGraph PROPERTIES
                INTERFACE_LINK_LIBRARIES "${MPSGraph_LIBRARIES}"
                INTERFACE_INCLUDE_DIRECTORIES "${MPSGraph_INCLUDE_DIRS}"
            )
        else()
            set_target_properties(MPSGraph::MPSGraph PROPERTIES
                INTERFACE_LINK_LIBRARIES "${MPSGraph_LIBRARIES}"
            )
        endif()
    endif()
else()
    set(MPSGraph_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPSGraph
    REQUIRED_VARS MPSGraph_LIBRARIES MPSGraph_INCLUDE_DIRS
    FOUND_VAR MPSGraph_FOUND
)

mark_as_advanced(MPSGraph_LIBRARIES MPSGraph_INCLUDE_DIRS)
