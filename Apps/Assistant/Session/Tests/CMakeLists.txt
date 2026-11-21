# Mila Assistant's tests: its settings file, licence, prompt and workspace store (AssistantInterface.md 5.5 and 8, MilaAssistant.md
# 3.7), which need no model.
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

include( GoogleTest )
include( "${CMAKE_CURRENT_SOURCE_DIR}/../KeyStore.cmake" )

add_executable( AssistantSessionTests
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
    "Understanding.cpp"
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
    "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Implementation/Understanding.cpp"
)

# BASE_DIRS is explicit because a module file set otherwise roots itself at this directory, and the modules being tested
# are the session's own.
target_sources( AssistantSessionTests
PRIVATE
    FILE_SET module_files
    TYPE CXX_MODULES
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../Src"
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Settings/Assistant.Settings.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Settings/Assistant.SettingsFile.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Settings/Assistant.MilaLicense.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Assistant.PromptTool.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Assistant.PromptPart.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Assistant.PromptFile.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Assistant.Places.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Assistant.RecordSearch.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Assistant.Offers.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Assistant.OfferAnswer.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Tools/Assistant.Understanding.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Assistant.ConversationSummary.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Assistant.Creations.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Assistant.ExposureRecord.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Store/Assistant.ConversationStore.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Store/Assistant.ProjectStore.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Store/Assistant.EventLog.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Store/Assistant.StoreFile.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Src/Store/Assistant.WorkspaceFolder.ixx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../${MILA_SESSION_KEY_STORE}"
)

set_target_properties( AssistantSessionTests PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if( MSVC )
    target_compile_options( AssistantSessionTests PRIVATE /bigobj )
endif()

target_link_libraries( AssistantSessionTests PRIVATE Mila gtest gtest_main ${MILA_SESSION_KEY_STORE_LIBRARIES} )

gtest_discover_tests( AssistantSessionTests
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
