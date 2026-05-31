# CMake configuration for the MilaPy pybind11 extension module.
# Requires CUDA; targets Llama 3.2 3B Instruct on CUDA BF16.

cmake_minimum_required(VERSION 3.31)

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
    OUTPUT_NAME "mila"
)

target_compile_features(MilaPy PUBLIC cxx_std_23)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(MilaPy PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/std:c++latest /permissive- /Zc:preprocessor /W4>
    )
endif()

target_link_libraries(MilaPy PRIVATE Mila pybind11::module)

# Copy the built extension module into the server directory so Python can
# import it without any PYTHONPATH configuration.
set(MILA_SERVER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Server")

add_custom_command(TARGET MilaPy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:MilaPy>
        "${MILA_SERVER_DIR}/$<TARGET_FILE_NAME:MilaPy>"
    COMMENT "Copying mila extension module to Server directory"
    VERBATIM
)