# Chat's tests: the rich-text formatter.
#
# Its own executable, because Chat.RichText imports nothing -- no Mila, no CUDA -- and linking it against the library
# would give the cheapest test surface in the tree the slowest build in the tree. It runs anywhere, including a CPU-only
# configure.
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

include( GoogleTest )

add_executable( ChatRichTextTests
    "RichText.Cpu.cpp"
)

# BASE_DIRS is explicit because a module file set otherwise roots itself at this directory, and the module being tested
# is the app's own.
target_sources( ChatRichTextTests
PUBLIC
    FILE_SET module_files
    TYPE CXX_MODULES
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../Src"
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Console/Chat.RichText.ixx"
)

set_target_properties( ChatRichTextTests PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# MilaStd for `import std;`: the module under test links nothing else of Mila's.
target_link_libraries( ChatRichTextTests PRIVATE MilaStd gtest gtest_main )

gtest_discover_tests( ChatRichTextTests
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
