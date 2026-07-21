# The docs are generated from the canonical standalone Doxyfile (Mila/Docs/Doxyfile),
# NOT from doxygen_add_docs. This is deliberate: the docs CI job runs Doxygen directly
# against that Doxyfile, so it needs only Doxygen + Graphviz and NOT a (CUDA-dependent)
# CMake configure. This 'docs' target invokes the same Doxyfile for local dev, keeping a
# single source of truth (all Doxygen settings live in Mila/Docs/Doxyfile).
#
# Paths in the Doxyfile are relative to the repository root, so the command runs with
# WORKING_DIRECTORY at ${CMAKE_SOURCE_DIR}; HTML lands in <repo>/build/docs/html.
# The Doxyfile enables WARN_AS_ERROR (the ratchet), so this target fails on any doc
# warning. It needs only Doxygen (graphs are disabled in the Doxyfile; no Graphviz).

# Not REQUIRED: MILA_ENABLE_DOCS defaults ON, but a normal compile (CI's build-and-gate
# job, or a contributor without Doxygen installed) must still configure -- the docs are
# built by the separate docs.yml workflow, not this build. Missing Doxygen degrades to a
# warning and no 'docs' target (handled by the DOXYGEN_FOUND branch below), not a hard
# configure failure.
find_package(Doxygen)

if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/build/docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen (Mila/Docs/Doxyfile)"
        VERBATIM)

    set_property(TARGET docs PROPERTY FOLDER "Docs")
    message(STATUS "Doxygen documentation target 'docs' created (uses Mila/Docs/Doxyfile)")
    message(STATUS "Documentation will be generated in: ${CMAKE_SOURCE_DIR}/build/docs")
else()
    message(WARNING "Doxygen not found - Documentation will not be built")
endif()
