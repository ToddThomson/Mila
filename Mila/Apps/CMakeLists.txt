# The apps Mila ships to the people who run it (see Apps/README.md and MilaProductFamily.md):
# Chat -> a human at a prompt; Assistant -> Mila, the assistant app, whose session process is built here;
# Server -> a foreign harness over HTTP (Python, importing the
# mila binding built from Mila/Bindings); Tools/Store -> the model store from the shell.
# Gated on MILA_ENABLE_APPS so a find_package/CPM consumer embedding Mila builds none of them.

# Chat and the Mila app's session process run models, and CUDA is the only compute backend; mila-app packages the
# session with the window and page, which are Node.
if(MILA_ENABLE_APPS AND MILA_ENABLE_CUDA)
    add_subdirectory( Chat )
    add_subdirectory( Assistant )
elseif(MILA_ENABLE_APPS)
    message(STATUS "MILA_ENABLE_CUDA is OFF - skipping mila-chat (instruct/CUDA-only)")
endif()

# The store needs only the store and the hub, so it builds on any backend.
if(MILA_ENABLE_APPS)
    add_subdirectory( Tools/Store )
endif()

# The server is not a CMake target -- it consumes the Mila Python binding, which is built from
# Mila/Bindings (added at the Mila/ level, not here).
#
# CMake does write its version, for the same reason it writes the binding's: the server ships as
# mila-llm-server, and a hand-maintained number in pyproject.toml is the shape that had already
# drifted 18 builds for the binding. One SemVer -> PEP 440 mapping serves both
# (cmake/MilaVersion.cmake); generated, so it is gitignored and a `python -m build` on an
# unconfigured checkout fails loudly rather than publishing a stale number.
file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/Server/VERSION" "${MILA_PEP440_VERSION}\n")
