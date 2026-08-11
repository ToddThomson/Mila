# CMake configuration for the MilaPy pybind11 extension module (module name: mila).
# Requires CUDA; binds the Gemma 4 and Llama 3.x inference paths.

cmake_minimum_required(VERSION 4.0)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

if(NOT MILA_HAS_CUDA)
    message(FATAL_ERROR "MilaPy requires CUDA. Enable MILA_ENABLE_CUDA or disable MILA_ENABLE_PYTHON_BINDINGS.")
endif()

set(CMAKE_CXX_SCAN_FOR_MODULES ON)

pybind11_add_module(MilaPy
    Mila_py.cpp
    Mila_py.Wrappers.cpp
)

# All Mila access is confined to this module interface unit; Mila_py.cpp binds
# against its std-only opaque handles and never imports Mila. The latest VS2026
# MSVC raises C2079 (basic_istream::sentry undefined) whenever Mila is imported
# into an ordinary .cpp that also includes std headers such as <string>.
target_sources(MilaPy
PUBLIC
    FILE_SET module_files
    TYPE CXX_MODULES FILES
        "Mila_py.Wrappers.ixx"
)

set_target_properties(MilaPy PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    OUTPUT_NAME "_mila"
)

target_compile_features(MilaPy PUBLIC cxx_std_23)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(MilaPy PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/std:c++latest /permissive- /Zc:preprocessor /W4>
    )
endif()

target_link_libraries(MilaPy PRIVATE Mila pybind11::module)

# ---------------------------------------------------------------------------
# Python package staging
#
# `_mila` is only half of what a consumer imports: the other half is the `mila`
# package whose __init__ registers the CUDA DLL directories BEFORE the extension
# loads (Windows does not search PATH for an extension's dependencies) and then
# re-exports from it. So every destination below receives a package DIRECTORY --
# __init__.py beside the extension -- never a bare .pyd.
# ---------------------------------------------------------------------------

set(MILA_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Package")
set(MILA_PACKAGE_INIT "${MILA_PACKAGE_SOURCE_DIR}/src/mila/__init__.py")

# The wheel version, derived from Version.txt rather than hand-maintained in
# pyproject.toml -- where it had already drifted 18 builds. Written into the package
# tree (not the build tree) because setup.py reads it, and because the wheel build
# copies that tree; generated, so it is gitignored and a `pip wheel` without a CMake
# build fails loudly rather than publishing a stale number.
file(WRITE "${MILA_PACKAGE_SOURCE_DIR}/VERSION" "${MILA_PEP440_VERSION}\n")

# 1. Into the package source tree, so `pip wheel Mila/Bindings/Package` packages a
#    freshly built extension. The wheel build is a packaging step over the CMake
#    build, not a second build system; setuptools never compiles anything here.
#
#    Top-level builds ONLY. This writes into the SOURCE tree, which a consumer that
#    pulled Mila in via FetchContent has no business doing -- it would stage a wheel
#    tree nobody in that build will package, into whatever directory Mila was fetched
#    from. It also broke: PROJECT_SOURCE_DIR below was CMAKE_SOURCE_DIR, which is the
#    TOP-LEVEL source dir, so under FetchContent it resolved to the consumer's root and
#    looked for a License.md that is not there. That failed
#    packaging_fetchcontent_consumer from 0.20.0-beta.2+17 until the guard was added.
if(PROJECT_IS_TOP_LEVEL)
    add_custom_command(TARGET MilaPy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:MilaPy>
            "${MILA_PACKAGE_SOURCE_DIR}/src/mila/$<TARGET_FILE_NAME:MilaPy>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PROJECT_SOURCE_DIR}/License.md"
            "${MILA_PACKAGE_SOURCE_DIR}/LICENSE"
        COMMENT "Staging mila extension into the wheel source tree"
        VERBATIM
    )
endif()

# 2. The neutral build-tree location, importable without installing anything: put
#    ${CMAKE_BINARY_DIR}/python on sys.path and `import mila` works. Per build
#    directory, so Debug and Release do not overwrite each other. Samples/Python
#    discovers it by glob; anything else can read MilaPy_PYTHON_DIR.
set(MILA_PYTHON_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python")

set(MilaPy_PYTHON_DIR "${MILA_PYTHON_PACKAGE_DIR}" CACHE INTERNAL
    "Directory to put on sys.path to import the built mila package")

add_custom_command(TARGET MilaPy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MILA_PYTHON_PACKAGE_DIR}/mila"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:MilaPy>
        "${MILA_PYTHON_PACKAGE_DIR}/mila/$<TARGET_FILE_NAME:MilaPy>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MILA_PACKAGE_INIT}"
        "${MILA_PYTHON_PACKAGE_DIR}/mila/__init__.py"
    COMMENT "Publishing the mila package to ${MILA_PYTHON_PACKAGE_DIR}"
    VERBATIM
)

# There is deliberately no third destination. MIS used to receive its own copy of the
# package beside main.py, which shadowed any installed mila-llm -- the server directory
# is first on sys.path, so a stale copy always won over a correct install. MIS now
# declares mila-llm as an ordinary dependency and a development checkout satisfies it
# with an editable install of Package above, which picks up destination 1 on every
# build. See Mila/Adaptors/Inference/Server/README.md.