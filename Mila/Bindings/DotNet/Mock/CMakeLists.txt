# The mock native library (DotNetBinding.md section 5): all of mila.h in C17 with no Mila behind it,
# built as MilaNative so the C# library loads it unchanged. Standalone: configure this directory alone.
#
#   cmake -S Mila/Bindings/DotNet/Mock -B Mila/Bindings/DotNet/Mock/build
#   cmake --build Mila/Bindings/DotNet/Mock/build

cmake_minimum_required( VERSION 3.20 )

project( MilaNativeMock LANGUAGES C )

set( CMAKE_C_STANDARD 17 )
set( CMAKE_C_STANDARD_REQUIRED ON )
set( CMAKE_C_EXTENSIONS OFF )

set( MILA_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." )
file( READ "${MILA_REPOSITORY_ROOT}/Version.txt" MILA_VERSION )
string( STRIP "${MILA_VERSION}" MILA_VERSION )

add_library( MilaNative SHARED
    Mock.ChatProtocol.c
    Mock.Common.c
    Mock.Json.c
    Mock.Model.c
    Mock.Runtime.c
    Mock.Store.c
    Mock.Tokenizer.c
)

target_include_directories( MilaNative PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../Native" )

target_compile_definitions( MilaNative PRIVATE
    MILA_NATIVE_BUILDING
    MILA_MOCK_VERSION="${MILA_VERSION}"
)

# Only MILA_API symbols are exported, as from the real library. The output lands in the build directory
# itself for every configuration, which is where the tests' resolver looks.
set_target_properties( MilaNative PROPERTIES
    C_VISIBILITY_PRESET hidden
    LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}>"
    RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}>"
)

if( MSVC )
    target_compile_options( MilaNative PRIVATE /W4 /WX /experimental:c11atomics )
else()
    target_compile_options( MilaNative PRIVATE -Wall -Wextra -Wpedantic -Werror )
endif()
