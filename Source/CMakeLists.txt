# CMakeList.txt : CMake project for Mila source
#
cmake_minimum_required (VERSION 3.20)

# Add source to this project's executable.
add_library( Mila STATIC 
    "Mila.cpp" "Mila.h")

configure_file( MilaConfig.h.in MilaConfig.h)

target_include_directories(
    Mila INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR} )
