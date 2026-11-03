# CPM-by-tag (release-access) consumer fixture for the packaging gate.
#
# Unlike fetchcontent_consumer (which embeds the LOCAL working tree via SOURCE_DIR,
# network-free), this fixture consumes Mila from the GitHub remote at a published
# tag via CPMAddPackage(GIT_TAG ...). It is the only gate that exercises the real
# release-access path: a downstream project pulling a tagged Mila over the network
# and consuming it as a subproject (CPM add_subdirectory's the fetched tree).
#
# Required (passed by drive_cpm.cmake):
#   MILA_CPM_GITHUB_REPOSITORY  owner/repo to fetch (e.g. ToddThomson/Mila)
#   MILA_CPM_GIT_TAG            published tag to fetch (e.g. v0.13.45-alpha.5)
# Optional:
#   MILA_CPM_SCRIPT             path to a local CPM.cmake (else downloaded)

cmake_minimum_required(VERSION 4.0)

project(MilaCpmConsumer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Required for the consumer to scan main.cpp for `import Mila;` and to compile the
# fetched Mila's C++23 module units (built in-tree by the CPM subproject add).
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

if(NOT DEFINED MILA_CPM_GITHUB_REPOSITORY)
    message(FATAL_ERROR "MILA_CPM_GITHUB_REPOSITORY must be set (e.g. ToddThomson/Mila).")
endif()
if(NOT DEFINED MILA_CPM_GIT_TAG)
    message(FATAL_ERROR "MILA_CPM_GIT_TAG must be set (e.g. v0.13.45-alpha.5).")
endif()

# Bring in CPM: prefer a local script provided by the driver (so CPM itself is not
# re-downloaded on every run); otherwise fetch the latest release.
if(DEFINED MILA_CPM_SCRIPT AND EXISTS "${MILA_CPM_SCRIPT}")
    include("${MILA_CPM_SCRIPT}")
else()
    set(_cpm_script "${CMAKE_BINARY_DIR}/CPM.cmake")
    if(NOT EXISTS "${_cpm_script}")
        file(DOWNLOAD
            https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake
            "${_cpm_script}"
            TLS_VERIFY ON
        )
    endif()
    include("${_cpm_script}")
endif()

# The release-access test: CPM git-clones Mila from the remote at the published tag
# and add_subdirectory's it (subproject mode), exposing the Mila::Mila target. Mila's
# own CMake reuses the CPM we already included here (it guards on COMMAND CPMAddPackage).
CPMAddPackage(
    NAME Mila
    GITHUB_REPOSITORY ${MILA_CPM_GITHUB_REPOSITORY}
    GIT_TAG ${MILA_CPM_GIT_TAG}
)

add_executable(mila_cpm_consumer main.cpp)
target_link_libraries(mila_cpm_consumer PRIVATE Mila::Mila)

# Clang needs explicit module flags to consume the module units; MSVC configures
# this automatically.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(mila_cpm_consumer PRIVATE -fno-implicit-modules -fno-implicit-module-maps)
endif()
