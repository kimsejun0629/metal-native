# FindAccelerate.cmake
# Locate Accelerate.framework on macOS/iOS
#
# This module defines:
#  Accelerate_FOUND - True if Accelerate framework is found
#  Accelerate_LIBRARIES - Libraries to link
#  Accelerate_INCLUDE_DIRS - Include directories

find_library(Accelerate_LIBRARIES
    NAMES Accelerate
    PATHS
        /System/Library/Frameworks
        /Library/Frameworks
)

if(Accelerate_LIBRARIES)
    get_filename_component(Accelerate_FRAMEWORK_DIR ${Accelerate_LIBRARIES} DIRECTORY)
    set(Accelerate_INCLUDE_DIRS "${Accelerate_FRAMEWORK_DIR}/Headers")
    set(Accelerate_FOUND TRUE)

    if(NOT TARGET Accelerate::Accelerate)
        add_library(Accelerate::Accelerate INTERFACE IMPORTED)
        set_target_properties(Accelerate::Accelerate PROPERTIES
            INTERFACE_LINK_LIBRARIES "${Accelerate_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${Accelerate_INCLUDE_DIRS}"
        )
    endif()
else()
    set(Accelerate_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Accelerate
    REQUIRED_VARS Accelerate_LIBRARIES Accelerate_INCLUDE_DIRS
    FOUND_VAR Accelerate_FOUND
)

mark_as_advanced(Accelerate_LIBRARIES Accelerate_INCLUDE_DIRS)
