# Driver for the FetchContent (subproject) packaging gate (invoked via `cmake -P`).
#
# Configures and builds the fetchcontent_consumer fixture, which embeds the Mila
# source tree via FetchContent_MakeAvailable (add_subdirectory). No install step:
# unlike the find_package gate, the subproject path builds Mila in-tree. A failure
# means Mila is not subproject-friendly -- e.g. a CMAKE_SOURCE_DIR that resolves to
# the consumer's root, or an install rule leaking into the consumer.
#
# Required -D arguments:
#   MILA_SOURCE_DIR   path to the Mila source tree (the repo root)
#   CONSUMER_SOURCE   path to the fetchcontent_consumer fixture
#   WORK_DIR          scratch dir for the consumer build
#   GENERATOR         CMake generator to reuse (e.g. Ninja)
# Optional:
#   CXX_COMPILER, CUDA_COMPILER  pin the consumer to the same toolchain

cmake_minimum_required(VERSION 4.0)

set(consumer_build "${WORK_DIR}/fetchcontent-build")

# Start clean so a stale build dir cannot mask a regression.
file(REMOVE_RECURSE "${consumer_build}")

set(compiler_args "")
if(CXX_COMPILER)
    list(APPEND compiler_args "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}")
endif()
if(CUDA_COMPILER)
    list(APPEND compiler_args "-DCMAKE_CUDA_COMPILER=${CUDA_COMPILER}")
endif()

# Build the subproject (Mila + the consumer exe) in the same config as the parent build
# so the consumer is actually validated under that configuration, not an empty default.
if(BUILD_TYPE)
    list(APPEND compiler_args "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")
endif()

# 1. Configure the consumer. FetchContent_MakeAvailable add_subdirectory's the
#    Mila tree -- if anything in Mila's CMake assumes top-level, this aborts.
execute_process(
    COMMAND ${CMAKE_COMMAND}
        -S "${CONSUMER_SOURCE}"
        -B "${consumer_build}"
        -G "${GENERATOR}"
        -D "MILA_SOURCE_DIR=${MILA_SOURCE_DIR}"
        ${compiler_args}
    RESULT_VARIABLE configure_result
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Packaging gate: FetchContent consumer configure failed (${configure_result}). Mila is not subproject-friendly.")
endif()

# 2. Build the consumer. This compiles Mila in-tree as a subproject and links it.
execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${consumer_build}"
    RESULT_VARIABLE build_result
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Packaging gate: FetchContent consumer build failed (${build_result}).")
endif()

message(STATUS "Packaging gate: FetchContent (subproject) consumer built successfully.")
