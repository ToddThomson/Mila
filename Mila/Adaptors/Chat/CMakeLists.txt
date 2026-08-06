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
        "Src/Chat.Ansi.ixx"
        "Src/Chat.Renderer.ixx"
        "Src/Chat.RichText.ixx"
        "Src/Chat.StreamingDisplay.ixx"
	    "Src/Chat.Config.ixx"
        "Src/Chat.Footprint.ixx"
        "Src/Chat.ModelCatalog.ixx"
        "Src/Chat.Message.ixx"
        "Src/Chat.MessageFormatter.ixx"
        "Src/Chat.SystemPrompt.ixx"
        "Src/Chat.ToolCallParser.ixx"
        "Src/Chat.ChannelParser.ixx"
        "Src/Json.ixx"
)

# No models directory is compiled in: the model store is the only source of weight paths,
# and it resolves them at runtime from records the build cannot know about.

# Chat.ixx crossed MSVC's 65535-section object limit (C1128). It is one translation unit
# holding the whole chat loop plus model hot-switching, so it instantiates the model
# templates across every family and quantization it can switch between -- section count
# tracks that, and it only grows. /bigobj widens the count field in the COFF object; it
# costs nothing at runtime and changes no code generation. MSVC-only: the flag does not
# exist for clang or gcc, which have no such limit.
if(MSVC)
    target_compile_options( ChatApp PRIVATE /bigobj )
endif()

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
