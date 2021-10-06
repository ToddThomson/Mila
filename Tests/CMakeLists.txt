# CMakeList.txt : Mila Testing subproject
#
cmake_minimum_required (VERSION 3.20)

enable_testing()

include( FetchContent )
FetchContent_Declare(
    googletest 
    URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable( googletest )
include( GoogleTest )

add_executable( "MilaTest" test.cpp )
target_link_libraries( MilaTest gtest_main )
    
gtest_discover_tests( MilaTest )