# Mila Desktop's tests: its settings file, licence and prompt (AssistantInterface.md 5.5, MilaDesktop.md 3.7), which need no model.
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

include( GoogleTest )

add_executable( DesktopSessionTests
    "SettingsFile.cpp"
    "PromptFile.cpp"
    "MilaLicense.cpp"
    "ConversationSummary.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Settings/Implementation/MilaLicense.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Implementation/ConversationSummary.cpp"
)

# BASE_DIRS is explicit because a module file set otherwise roots itself at this directory, and the modules being tested
# are the session's own.
target_sources( DesktopSessionTests
PRIVATE
    FILE_SET module_files
    TYPE CXX_MODULES
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../Src"
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Settings/Desktop.Settings.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Settings/Desktop.SettingsFile.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Settings/Desktop.MilaLicense.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Desktop.PromptTool.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Desktop.PromptPart.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Desktop.PromptFile.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Desktop.ConversationSummary.ixx"
)

set_target_properties( DesktopSessionTests PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if( MSVC )
    target_compile_options( DesktopSessionTests PRIVATE /bigobj )
endif()

target_link_libraries( DesktopSessionTests PRIVATE Mila gtest gtest_main )

gtest_discover_tests( DesktopSessionTests
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
