# Mila QuickStart -- minimal example of consuming an INSTALLED Mila via find_package.
#
# This is a STANDALONE project. It is intentionally NOT add_subdirectory()'d by the
# Mila build (see ../CMakeLists.txt): it must be configured on its own against an
# install prefix, so that find_package(Mila) resolves the installed package rather
# than the in-tree target. Build it after `cmake --install`:
#
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<mila-install-prefix>
#   cmake --build build
#
# The Tests/Packaging gate automates exactly this and is how CI validates packaging.

cmake_minimum_required(VERSION 4.0)

project(MilaQuickStart LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Required for the consumer to scan main.cpp for `import Mila;` and to recompile
# Mila's installed C++23 module units (module BMIs are not portable across
# toolchains, so the consumer rebuilds them from the installed sources).
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

find_package(Mila CONFIG REQUIRED)

add_executable(mila_quickstart main.cpp)
target_link_libraries(mila_quickstart PRIVATE Mila::Mila)

# Clang needs explicit module flags to consume the installed module units; MSVC
# configures this automatically.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(mila_quickstart PRIVATE -fno-implicit-modules -fno-implicit-module-maps)
endif()
