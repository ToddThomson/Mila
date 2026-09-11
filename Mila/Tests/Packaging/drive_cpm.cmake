# Driver for the CPM-by-tag (release-access) packaging gate (invoked via `cmake -P`).
#
# Configures and builds the cpm_consumer fixture, which pulls Mila from the GitHub
# remote at a published tag via CPMAddPackage and links it as a subproject. Unlike
# the FetchContent gate (local SOURCE_DIR, network-free), this gate hits the network
# and validates the ACTUAL release: a downstream consumer fetching a tagged Mila. A
# configure failure means the tag is missing/unreachable; a build failure means the
# tagged tree is not subproject-consumable.
#
# Required -D arguments:
#   CONSUMER_SOURCE     path to the cpm_consumer fixture
#   WORK_DIR            scratch dir for the consumer build
#   GENERATOR           CMake generator to reuse (e.g. Ninja)
#   GITHUB_REPOSITORY   owner/repo to fetch (e.g. ToddThomson/Mila)
#   GIT_TAG             published tag to fetch (e.g. v0.13.45-alpha.5)
# Optional:
#   CXX_COMPILER, CUDA_COMPILER  pin the consumer to the same toolchain
#   CPM_SCRIPT                   local CPM.cmake to reuse (else downloaded)
#   CPM_SOURCE_CACHE             dir to cache CPM clones across runs

cmake_minimum_required(VERSION 4.0)

set(consumer_build "${WORK_DIR}/cpm-build")

# Start clean so a stale build dir cannot mask a regression.
file(REMOVE_RECURSE "${consumer_build}")

# Force a FRESH fetch of Mila itself every run. This gate's whole purpose is to prove a
# clean downstream consumer can clone + build the published tag, so a persisted Mila clone
# must never be reused -- in particular a dirty/partial clone left by a prior fail-closed
# run (tag not yet pushed) would otherwise be picked up and break with a missing Mila::Mila
# target. The rest of the CPM cache (cutlass/nlohmann/miniz: stable, expensive to re-clone)
# is intentionally kept.
if(CPM_SOURCE_CACHE)
    file(REMOVE_RECURSE "${CPM_SOURCE_CACHE}/mila")
endif()

set(consumer_args "")
if(CXX_COMPILER)
    list(APPEND consumer_args "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}")
endif()
if(CUDA_COMPILER)
    list(APPEND consumer_args "-DCMAKE_CUDA_COMPILER=${CUDA_COMPILER}")
endif()
if(CPM_SCRIPT)
    list(APPEND consumer_args "-DMILA_CPM_SCRIPT=${CPM_SCRIPT}")
endif()
if(CPM_SOURCE_CACHE)
    list(APPEND consumer_args "-DCPM_SOURCE_CACHE=${CPM_SOURCE_CACHE}")
endif()

# Build the fetched Mila + consumer in the same config as the parent build so the
# consumer is validated under that configuration, not an empty default.
if(BUILD_TYPE)
    list(APPEND consumer_args "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")
endif()

# Announced BEFORE the work, not only after it. A pass that names its tag only on success is
# indistinguishable from a stale pass when the run is short and cached, which is exactly how this
# gate once validated the previous release and reported green.
message(STATUS "Packaging gate: CPM release-access, fetching ${GITHUB_REPOSITORY}@${GIT_TAG}")

# 1. Configure the consumer. CPMAddPackage git-clones Mila at GIT_TAG from the
#    remote -- this is the step that exercises real release access.
execute_process(
    COMMAND ${CMAKE_COMMAND}
        -S "${CONSUMER_SOURCE}"
        -B "${consumer_build}"
        -G "${GENERATOR}"
        -D "MILA_CPM_GITHUB_REPOSITORY=${GITHUB_REPOSITORY}"
        -D "MILA_CPM_GIT_TAG=${GIT_TAG}"
        ${consumer_args}
    RESULT_VARIABLE configure_result
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Packaging gate: CPM consumer configure failed (${configure_result}). Tag '${GIT_TAG}' on '${GITHUB_REPOSITORY}' is unreachable, missing, or not consumable.")
endif()

# 2. Build the consumer. Compiles the fetched Mila in-tree and links it.
execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${consumer_build}"
    RESULT_VARIABLE build_result
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Packaging gate: CPM consumer build failed (${build_result}).")
endif()

message(STATUS "Packaging gate: CPM (release-access) consumer built successfully from ${GITHUB_REPOSITORY}@${GIT_TAG}.")
