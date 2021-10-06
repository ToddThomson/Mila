# CMakeList.txt : Mila top-level CMake project file
#
cmake_minimum_required( VERSION 3.20 )

project ( "Mila" VERSION 0.0.1 )

option( BUILD_TESTS "Build unit tests" ON )
option( BUILD_SAMPLES "Build samples" ON )

# Include sub-projects.
add_subdirectory( "Source")
add_subdirectory( "Samples" )

if ( BUILD_TESTS )
    enable_testing()
    add_subdirectory( "Tests" )
endif()
