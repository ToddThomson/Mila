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

# 1. Into the package source tree, so `pip wheel Mila/Bindings/Package` packages a
#    freshly built extension. The wheel build is a packaging step over the CMake
#    build, not a second build system; setuptools never compiles anything here.
add_custom_command(TARGET MilaPy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:MilaPy>
        "${MILA_PACKAGE_SOURCE_DIR}/src/mila/$<TARGET_FILE_NAME:MilaPy>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/License.md"
        "${MILA_PACKAGE_SOURCE_DIR}/LICENSE"
    COMMENT "Staging mila extension into the wheel source tree"
    VERBATIM
)

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

# 3. Convenience: the MIS server directory, so its `python main.py` imports the
#    binding without PYTHONPATH setup. Retained because MIS's documented run
#    instructions depend on it. Once mila-llm is installed from a wheel, MIS should
#    depend on it like any other package and this copy can go.
set(MILA_SERVER_DIR "${CMAKE_SOURCE_DIR}/Mila/Adaptors/Inference/Server")

add_custom_command(TARGET MilaPy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MILA_SERVER_DIR}/mila"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:MilaPy>
        "${MILA_SERVER_DIR}/mila/$<TARGET_FILE_NAME:MilaPy>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MILA_PACKAGE_INIT}"
        "${MILA_SERVER_DIR}/mila/__init__.py"
    COMMENT "Copying the mila package to the Server directory"
    VERBATIM
)