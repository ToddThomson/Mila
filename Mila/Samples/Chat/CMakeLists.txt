add_executable( ChatApp 
    "Src/main.cpp"
)

set_target_properties( ChatApp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_sources( ChatApp
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
        "Src/Chat.ixx"
        "Src/Chat.Renderer.ixx"
        "Src/Chat.RichText.ixx"
        "Src/Chat.StreamingDisplay.ixx"
	    "Src/Chat.Config.ixx"
        "Src/Chat.ModelCatalog.ixx"
        "Src/Chat.Message.ixx"
        "Src/Chat.MessageFormatter.ixx"
        "Src/Chat.SystemPrompt.ixx"
        "Src/Chat.ToolCallParser.ixx"
        "Src/Chat.GemmaToolCallParser.ixx"
        "Src/Chat.ChannelParser.ixx"
        "Src/Json.ixx"
)

# Define the Data/Models directory path
set(MODELS_DIR "${CMAKE_SOURCE_DIR}/Data/Models")
target_compile_definitions( ChatApp PRIVATE MODELS_DIR="${MODELS_DIR}" )

# ChatApp uses no CUDA APIs directly. When CUDA is enabled, Mila propagates
# CUDA::cudart/nvtx3 as PUBLIC usage requirements, so linking Mila alone is
# sufficient; when CUDA is OFF those targets do not exist. Relisting them here
# would be redundant in the first case and a hard error in the second.
target_link_libraries( ChatApp PRIVATE Mila )

# Copy chat session data to the output directory alongside the executable
add_custom_command( TARGET ChatApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/Data"
        "$<TARGET_FILE_DIR:ChatApp>/Data"
)
