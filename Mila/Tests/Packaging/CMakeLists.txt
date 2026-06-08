# find_package(Mila) packaging gate.
#
# This is an orchestration test, not a link-and-run test: it does not link Mila. It
# installs the built Mila into a throwaway prefix and builds Samples/QuickStart
# against it via find_package(Mila). That is the only way to exercise the install
# tree -- the in-tree build and the unit tests link the Mila target directly and so
# cannot catch packaging regressions (a missing installed header still resolves
# file-relative in the source tree, leaving their builds green).

add_test(
    NAME packaging_find_package_consumer
    COMMAND ${CMAKE_COMMAND}
        -D "MILA_BINARY_DIR=${CMAKE_BINARY_DIR}"
        -D "QUICKSTART_SOURCE_DIR=${CMAKE_SOURCE_DIR}/Mila/Samples/QuickStart"
        -D "WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/work"
        -D "GENERATOR=${CMAKE_GENERATOR}"
        -D "CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        -D "CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
        -D "BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/drive_consumer.cmake"
)

# Generous timeout: the consumer recompiles Mila's module graph from source.
set_tests_properties(packaging_find_package_consumer PROPERTIES
    TIMEOUT 900
    LABELS "packaging"
)

# FetchContent / subproject gate: embeds the Mila source tree via
# add_subdirectory and links the in-tree target. Proves nothing in Mila's CMake
# assumes top-level (the find_package gate above cannot catch that -- it consumes
# an install, never add_subdirectory).
add_test(
    NAME packaging_fetchcontent_consumer
    COMMAND ${CMAKE_COMMAND}
        -D "MILA_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
        -D "CONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/fetchcontent_consumer"
        -D "WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/work"
        -D "GENERATOR=${CMAKE_GENERATOR}"
        -D "CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        -D "CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
        -D "BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/drive_fetchcontent.cmake"
)

# Heavier than the find_package gate: a subproject build compiles ALL of Mila
# in-tree (every .cu kernel), not just the module interfaces.
set_tests_properties(packaging_fetchcontent_consumer PROPERTIES
    TIMEOUT 1800
    LABELS "packaging"
)

# CPM release-access gate (OPT-IN). The two gates above are network-free -- they
# consume the LOCAL working tree -- so they validate packaging correctness but NOT
# that a published release is actually fetchable. This gate closes that gap: CPM
# git-clones Mila from the GitHub remote at a published tag and builds against it,
# exercising the real downstream release-consumption path. It requires network
# access AND a pushed tag, so it is OFF by default and excluded from the standard
# (network-free) CI green. Enable once a tag is published:
#   -DMILA_ENABLE_CPM_GATE=ON [-DMILA_CPM_GIT_TAG=<tag>] [-DMILA_CPM_GITHUB_REPOSITORY=<owner/repo>]
# The tag defaults to the current Version.txt (e.g. v0.13.45-alpha.5).
option(MILA_ENABLE_CPM_GATE "Enable the CPM-by-tag release-access packaging gate (network; requires a published tag)." OFF)

if(MILA_ENABLE_CPM_GATE)
    if(MILA_PRERELEASE_TAG)
        set(_mila_default_tag "v${MILA_VERSION_NUMERIC}-${MILA_PRERELEASE_TAG}")
    else()
        set(_mila_default_tag "v${MILA_VERSION_NUMERIC}")
    endif()

    set(MILA_CPM_GITHUB_REPOSITORY "ToddThomson/Mila" CACHE STRING "owner/repo for the CPM release-access gate.")
    set(MILA_CPM_GIT_TAG "${_mila_default_tag}" CACHE STRING "Published tag for the CPM release-access gate.")

    add_test(
        NAME packaging_cpm_consumer
        COMMAND ${CMAKE_COMMAND}
            -D "CONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/cpm_consumer"
            -D "WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/work"
            -D "GENERATOR=${CMAKE_GENERATOR}"
            -D "CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            -D "CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
            -D "BUILD_TYPE=${CMAKE_BUILD_TYPE}"
            -D "GITHUB_REPOSITORY=${MILA_CPM_GITHUB_REPOSITORY}"
            -D "GIT_TAG=${MILA_CPM_GIT_TAG}"
            -D "CPM_SCRIPT=${CMAKE_SOURCE_DIR}/cmake/CPM.cmake"
            -D "CPM_SOURCE_CACHE=${CMAKE_CURRENT_BINARY_DIR}/cpm-cache"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/drive_cpm.cmake"
    )

    # Network clone + full in-tree subproject build of the fetched tree.
    set_tests_properties(packaging_cpm_consumer PROPERTIES
        TIMEOUT 1800
        LABELS "packaging"
    )
endif()
