# Driver for the find_package(Mila) packaging gate (invoked via `cmake -P`).
#
# Installs the freshly built Mila into a throwaway prefix, then configures and
# builds Samples/QuickStart against it through find_package(Mila). A failure at any
# step means the install tree is not consumable -- the exact regression Mila's own
# build cannot catch, because it links the in-tree target rather than the install.
#
# Required -D arguments:
#   MILA_BINARY_DIR        top-level Mila build dir (contains cmake_install.cmake)
#   QUICKSTART_SOURCE_DIR  path to Samples/QuickStart
#   WORK_DIR               scratch dir for the install prefix and consumer build
#   GENERATOR              CMake generator to reuse (e.g. Ninja)
# Optional:
#   CXX_COMPILER, CUDA_COMPILER  pin the consumer to the same toolchain

cmake_minimum_required(VERSION 4.0)

set(prefix "${WORK_DIR}/prefix")
set(consumer_build "${WORK_DIR}/quickstart-build")

# Start from a clean slate so a stale prefix or build dir cannot mask a regression.
file(REMOVE_RECURSE "${prefix}" "${consumer_build}")
file(MAKE_DIRECTORY "${prefix}")

# 1. Install the built Mila into the throwaway prefix.
execute_process(
    COMMAND ${CMAKE_COMMAND} --install "${MILA_BINARY_DIR}" --prefix "${prefix}"
    RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "Packaging gate: 'cmake --install' failed (${install_result}).")
endif()

# Pin the consumer to the same toolchain when the caller provided it.
set(compiler_args "")
if(CXX_COMPILER)
    list(APPEND compiler_args "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}")
endif()
if(CUDA_COMPILER)
    list(APPEND compiler_args "-DCMAKE_CUDA_COMPILER=${CUDA_COMPILER}")
endif()

# 2. Configure the consumer against the installed package.
execute_process(
    COMMAND ${CMAKE_COMMAND}
        -S "${QUICKSTART_SOURCE_DIR}"
        -B "${consumer_build}"
        -G "${GENERATOR}"
        -D "CMAKE_PREFIX_PATH=${prefix}"
        ${compiler_args}
    RESULT_VARIABLE configure_result
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Packaging gate: consumer configure (find_package(Mila)) failed (${configure_result}).")
endif()

# 3. Build the consumer. This recompiles Mila's installed module units in the
#    consumer toolchain, resolving their file-relative kernel-header includes
#    against the install tree -- the layout the packaging fix has to get right.
execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${consumer_build}"
    RESULT_VARIABLE build_result
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Packaging gate: consumer build failed (${build_result}). The install tree is not consumable.")
endif()

message(STATUS "Packaging gate: find_package(Mila) consumer built successfully.")
