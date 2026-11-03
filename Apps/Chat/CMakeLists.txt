add_executable( mila-chat
    "Src/main.cpp"
)

set_target_properties( mila-chat PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_sources( mila-chat
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
        "Src/Chat.Session.ixx"
        "Src/Chat.SessionEvents.ixx"
        "Src/Chat.TextChannel.ixx"
        "Src/Chat.ToolDecision.ixx"
        "Src/Chat.RoundStats.ixx"
        "Src/Models/Chat.ContextFloor.ixx"
        "Src/Chat.ParsedResponse.ixx"
        "Src/Console/Chat.Ansi.ixx"
        "Src/Settings/Chat.Config.ixx"
        "Src/Settings/Chat.QuantizationKeyword.ixx"
        "Src/Settings/Chat.KvCacheKeyword.ixx"
        "Src/Settings/Chat.DetailLevel.ixx"
        "Src/Settings/Chat.Settings.ixx"
        "Src/Models/Chat.Footprint.ixx"
        "Src/Models/Chat.FootprintVerdict.ixx"
        "Src/Console/Chat.ScopedLogSuppression.ixx"
        "Src/Models/Chat.ModelCatalog.ixx"
        "Src/Models/Chat.ResolvedModel.ixx"
        "Src/Models/Chat.RefusalAudience.ixx"
        "Src/Models/Chat.ModelListing.ixx"
        "Src/Models/Chat.PublishedModel.ixx"
        "Src/Models/Chat.FootprintBudget.ixx"
        "Src/Settings/Chat.SettingsResolution.ixx"
        "Src/Settings/Chat.ConfigError.ixx"
        "Src/Models/Chat.ModelNotFound.ixx"
        "Src/Chat.Message.ixx"
        "Src/Chat.SystemPrompt.ixx"
        "Src/Console/Chat.ConsoleFrontEnd.ixx"
        "Src/Console/Chat.Renderer.ixx"
        "Src/Console/Chat.RichText.ixx"
        "Src/Console/Chat.IncrementalRichFormatter.ixx"
        "Src/Console/Chat.StreamingDisplay.ixx"
)

# Chat.Session.ixx crossed MSVC's 65535-section object limit (C1128) when it was Chat.ixx. It is one translation unit
# holding the turn loop plus model hot-switching, so it instantiates the model templates across every family and
# quantization it can switch between -- section count tracks that, and it only grows. /bigobj widens the count field
# in the COFF object; it costs nothing at runtime and changes no code generation. MSVC-only: the flag does not exist
# for clang or gcc, which have no such limit.
if(MSVC)
    target_compile_options( mila-chat PRIVATE /bigobj )
endif()

# No models directory is compiled in: the model store is the only source of weight paths,
# and it resolves them at runtime from records the build cannot know about.

# Chat uses no CUDA APIs directly. When CUDA is enabled, Mila propagates CUDA::cudart/nvtx3 as PUBLIC usage
# requirements, so linking Mila alone is sufficient; when CUDA is OFF those targets do not exist. Relisting them here
# would be redundant in the first case and a hard error in the second.
target_link_libraries( mila-chat PRIVATE Mila )

# The system prompts are content, and they are the ONLY thing the build places beside the
# executable: no configuration ships, because the compiled defaults are a working configuration and
# a shipped file would outrank the user's own. This directory is where a named prompt is searched
# for after the user's config root -- see Specifications/ChatConfiguration.md section 8.
add_custom_command( TARGET mila-chat POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/Prompts"
        "$<TARGET_FILE_DIR:mila-chat>/Prompts"
)
