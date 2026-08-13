# Packaging gates. FetchContent (subproject consumption) is the ONLY supported way to depend
# on Mila; find_package and the install/export rules it needed were removed in 0.20.0-beta.3.
#
# The reasoning, kept here because it is the thing to re-read before anyone proposes bringing
# find_package back: a C++23 module library is a source distribution. Module BMIs are not
# portable, so a consumer recompiles Mila's .ixx graph in its own toolchain either way, which
# voids find_package's prebuilt-binary benefit. What it does NOT void is the cost -- the whole
# install-layout apparatus (FILE_SET destinations, kernel-header mirroring, the MilaTargets
# export, a generated config) plus toolchain/ABI coupling between the prebuilt archive and the
# consumer's recompiled modules. FetchContent compiles Mila once, in one coherent toolchain.
#
# The honest trade, which the gate timeouts below record: a subproject consumer builds ALL of
# Mila, every .cu kernel, not just the module interfaces. That is the price of the supported
# path and it is paid knowingly.

# FetchContent / subproject gate -- the PRIMARY (supported) packaging gate. Embeds the
# Mila source tree via add_subdirectory and links the in-tree Mila::Mila target, proving
# nothing in Mila's CMake assumes top-level (CMAKE_SOURCE_DIR vs PROJECT_SOURCE_DIR,
# PROJECT_IS_TOP_LEVEL option gating, install-rule leakage). This is exactly how a
# downstream consumer embeds Mila via FetchContent_Declare(Mila GIT_REPOSITORY/URL ...).
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

# A subproject build compiles ALL of Mila in-tree (every .cu kernel), not just the
# module interfaces -- hence the wide timeout.
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
