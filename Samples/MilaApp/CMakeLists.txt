# CMakeList.txt : MilaApp source project
#
cmake_minimum_required (VERSION 3.20)

project( MilaApp VERSION 0.0.1 )

option( USE_MILA "Use Mila implementation" ON)

configure_file( MilaConfig.h.in MilaConfig.h )

if ( USE_MILA )
  list( APPEND EXTRA_LIBS Mila )
endif()

# Add source to this project's executable.
add_executable( MilaApp "MilaApp.cpp" "MilaApp.h" )

target_link_libraries( MilaApp Mila )

# add the binary tree to the search path for include files
# so that we will find Learner.h
target_include_directories(
    Mila PUBLIC
    "${PROJECT_BINARY_DIR}" )
