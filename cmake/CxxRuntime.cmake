# On macOS, every program and module Mila links carries LLVM's C++ runtime inside itself. Mila needs LLVM's own
# Clang and libc++ there (Apple's toolchain ships no `std` module source), and macOS has no copy of that libc++, so a
# program linked to it dynamically starts only on a Mac where the toolchain is installed (MilaDesktop.md 9.4).

# Finds `archive` for the compiler in use: where the compiler itself looks, else in `subdirectory` of the library
# folder beside it, which is where Homebrew's LLVM keeps libc++ (lib/c++) and libunwind (lib/unwind).
function(mila_find_cxx_runtime_archive variable archive subdirectory)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=${archive}
        OUTPUT_VARIABLE path OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(NOT (IS_ABSOLUTE "${path}" AND EXISTS "${path}"))
        cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH compiler_directory)
        cmake_path(APPEND compiler_directory "../lib/${subdirectory}/${archive}" OUTPUT_VARIABLE path)
        cmake_path(NORMAL_PATH path)
    endif()

    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "No ${archive} for ${CMAKE_CXX_COMPILER}: a macOS build links LLVM's C++ runtime statically")
    endif()

    set(${variable} "${path}" PARENT_SCOPE)
endfunction()

# Links libc++, libc++abi and LLVM's libunwind into every executable and module defined from the calling directory
# down. libc++abi needs LLVM's unwinder rather than the one in libSystem. The archives are loaded hidden, so a module
# that carries them (a Python extension) exports none of their symbols to the process that loads it.
function(mila_link_cxx_runtime_statically)
    if(NOT (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
        return()
    endif()

    mila_find_cxx_runtime_archive(libcxx libc++.a c++)
    mila_find_cxx_runtime_archive(libcxxabi libc++abi.a c++)
    mila_find_cxx_runtime_archive(libunwind libunwind.a unwind)

    message(STATUS "Mila: C++ runtime linked statically from ${libcxx}")
    add_link_options(-nostdlib++
        "LINKER:-load_hidden,${libcxx}"
        "LINKER:-load_hidden,${libcxxabi}"
        "LINKER:-load_hidden,${libunwind}")
endfunction()
