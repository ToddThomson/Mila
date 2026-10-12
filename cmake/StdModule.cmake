# The standard library's `std` and `std.compat` modules, built as an ordinary module library from the sources
# the toolchain ships, so neither Mila nor a FetchContent consumer needs CMake's experimental `import std` gate.
# Replace with CXX_MODULE_STD once CMake's support is no longer experimental.

# Finds the source of `std` and `std.compat` for the compiler in use, from the manifest a standard library
# publishes beside itself (libstdc++ 15, libc++ 17), or from the MSVC toolset's own modules directory.
function(mila_find_std_module_sources std_variable compat_variable)
    if(MSVC)
        # cl.exe lives in <toolset>/bin/Host<arch>/<arch>; the STL's module sources are in <toolset>/modules.
        cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH compiler_directory)
        cmake_path(APPEND compiler_directory "../../../modules" OUTPUT_VARIABLE module_directory)
        cmake_path(NORMAL_PATH module_directory)
        set(${std_variable} "${module_directory}/std.ixx" PARENT_SCOPE)
        set(${compat_variable} "${module_directory}/std.compat.ixx" PARENT_SCOPE)

        return()
    endif()

    foreach(library libstdc++ libc++)
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=${library}.modules.json
            OUTPUT_VARIABLE manifest OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(IS_ABSOLUTE "${manifest}" AND EXISTS "${manifest}")
            break()
        endif()

        set(manifest "")
    endforeach()

    if(NOT manifest)
        message(FATAL_ERROR "${CMAKE_CXX_COMPILER}'s standard library ships no std module; Mila needs libstdc++ 15 or libc++ 17")
    endif()

    file(READ "${manifest}" json)
    cmake_path(GET manifest PARENT_PATH manifest_directory)
    string(JSON count LENGTH "${json}" modules)
    math(EXPR last "${count} - 1")

    foreach(index RANGE ${last})
        string(JSON name GET "${json}" modules ${index} logical-name)
        string(JSON path GET "${json}" modules ${index} source-path)
        cmake_path(ABSOLUTE_PATH path BASE_DIRECTORY "${manifest_directory}" NORMALIZE)

        if(name STREQUAL "std")
            set(${std_variable} "${path}" PARENT_SCOPE)
        elseif(name STREQUAL "std.compat")
            set(${compat_variable} "${path}" PARENT_SCOPE)
        endif()
    endforeach()
endfunction()

# Adds `target`, a static library whose module file set is `std` and `std.compat`. A target that imports either
# links it; the compile options here must match the importers', since a module is usable only by a translation
# unit compiled with the same language options.
function(mila_add_std_module_library target)
    mila_find_std_module_sources(std_source compat_source)

    foreach(source std_source compat_source)
        if(NOT EXISTS "${${source}}")
            message(FATAL_ERROR "No ${source} at '${${source}}'")
        endif()
    endforeach()

    cmake_path(GET std_source PARENT_PATH base_directory)
    message(STATUS "Mila: std module from ${std_source}")

    add_library(${target} STATIC)
    target_sources(${target} PUBLIC FILE_SET CXX_MODULES BASE_DIRS "${base_directory}" FILES "${std_source}" "${compat_source}")
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CXX_SCAN_FOR_MODULES ON
        POSITION_INDEPENDENT_CODE ON)

    if(cxx_std_23 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        target_compile_features(${target} INTERFACE cxx_std_23)
    else()
        target_compile_features(${target} INTERFACE cxx_std_20)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target} PRIVATE /std:c++latest /permissive- /Zc:preprocessor)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # The sources declare `export module std;`, a name reserved to the implementation.
        target_compile_options(${target} PRIVATE -std=c++23 -fno-implicit-modules -fno-implicit-module-maps
            -Wno-reserved-module-identifier)
    endif()
endfunction()
