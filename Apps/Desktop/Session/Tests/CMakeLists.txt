# Mila Desktop's tests: its settings file, licence, prompt and workspace store (AssistantInterface.md 5.5 and 8, MilaDesktop.md
# 3.7), which need no model.
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

include( GoogleTest )
include( "${CMAKE_CURRENT_SOURCE_DIR}/../KeyStore.cmake" )

add_executable( DesktopSessionTests
    "SettingsFile.cpp"
    "PromptFile.cpp"
    "MilaLicense.cpp"
    "ConversationSummary.cpp"
    "Creations.cpp"
    "ExposureRecord.cpp"
    "SearchKeyStore.cpp"
    "ConversationStore.cpp"
    "ProjectStore.cpp"
    "EventLog.cpp"
    "Places.cpp"
    "RecordSearch.cpp"
    "Offers.cpp"
    "OfferAnswer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Settings/Implementation/MilaLicense.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Implementation/ConversationSummary.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Implementation/Creations.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Implementation/ExposureRecord.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Store/Implementation/ConversationStore.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Store/Implementation/ProjectStore.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Store/Implementation/EventLog.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Store/Implementation/StoreFile.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Implementation/Places.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Implementation/RecordSearch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Implementation/Offers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Implementation/OfferAnswer.cpp"
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
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Desktop.Places.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Desktop.RecordSearch.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Desktop.Offers.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Desktop.OfferAnswer.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Desktop.ConversationSummary.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Desktop.Creations.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Desktop.ExposureRecord.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Store/Desktop.ConversationStore.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Store/Desktop.ProjectStore.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Store/Desktop.EventLog.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Store/Desktop.StoreFile.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Store/Desktop.WorkspaceFolder.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../${MILA_SESSION_KEY_STORE}"
)

set_target_properties( DesktopSessionTests PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if( MSVC )
    target_compile_options( DesktopSessionTests PRIVATE /bigobj )
endif()

target_link_libraries( DesktopSessionTests PRIVATE Mila gtest gtest_main ${MILA_SESSION_KEY_STORE_LIBRARIES} )

gtest_discover_tests( DesktopSessionTests
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
