# CMake project for testing the Mila Library

enable_testing()

include(FetchContent)
   FetchContent_Declare(
     googletest
     URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set( gtest_force_shared_crt ON CACHE BOOL "" FORCE )

FetchContent_MakeAvailable( googletest )
include( GoogleTest )

add_executable( Gpt2Tests
    "main.cpp" 
    "Tokenizer.cpp")

target_link_libraries( Gpt2Tests PRIVATE Mila gtest )

add_custom_command(
    TARGET Gpt2Tests POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different 
        "${CMAKE_SOURCE_DIR}/data"
        "${CMAKE_CURRENT_BINARY_DIR}/data"
    COMMENT "Copying data directory to build output directory"
)

gtest_discover_tests(Gpt2Tests)