# Mila Desktop's tests: its settings file and activity prompts (MilaDesktop.md 3.1, 3.7), which need no model.
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

include( GoogleTest )

add_executable( DesktopSessionTests
    "SettingsFile.cpp"
    "PromptFile.cpp"
)

# BASE_DIRS is explicit because a module file set otherwise roots itself at this directory, and the modules being tested
# are the session's own.
target_sources( DesktopSessionTests
PRIVATE
    FILE_SET module_files
    TYPE CXX_MODULES
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../Src"
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Desktop.Settings.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Desktop.SettingsFile.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Desktop.PromptTool.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Desktop.PromptFile.ixx"
)

set_target_properties( DesktopSessionTests PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if( MSVC )
    target_compile_options( DesktopSessionTests PRIVATE /bigobj )
endif()

target_link_libraries( DesktopSessionTests PRIVATE Mila MilaAgents gtest gtest_main )

gtest_discover_tests( DesktopSessionTests
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
