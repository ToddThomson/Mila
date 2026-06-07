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
        -P "${CMAKE_CURRENT_SOURCE_DIR}/drive_fetchcontent.cmake"
)

# Heavier than the find_package gate: a subproject build compiles ALL of Mila
# in-tree (every .cu kernel), not just the module interfaces.
set_tests_properties(packaging_fetchcontent_consumer PROPERTIES
    TIMEOUT 1800
    LABELS "packaging"
)
