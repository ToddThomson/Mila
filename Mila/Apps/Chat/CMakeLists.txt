# The session: the model, the conversation, the settings and the turn loop, with no console in it
# (Specifications/ChatDesktop.md section 7). It is a library so that the console and the session process are two
# front ends over one session, and so the build itself keeps the console out: nothing here can import the renderer.
add_library( MilaChatSession STATIC )

target_sources( MilaChatSession
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
        "Src/Chat.Session.ixx"
        "Src/Chat.SessionEvents.ixx"
        "Src/Chat.TextChannel.ixx"
        "Src/Chat.ToolDecision.ixx"
        "Src/Chat.RoundStats.ixx"
        "Src/Chat.ThinkingEffort.ixx"
        "Src/Chat.ContextFloor.ixx"
        "Src/Chat.StreamingChannelRouter.ixx"
        "Src/Chat.StreamingDetokenizer.ixx"
        "Src/Chat.GemmaStreamTokens.ixx"
        "Src/Chat.Ansi.ixx"
	    "Src/Chat.Config.ixx"
        "Src/Chat.ModelType.ixx"
        "Src/Chat.ModelSize.ixx"
        "Src/Chat.QuantizationMode.ixx"
        "Src/Chat.KvCache.ixx"
        "Src/Chat.DetailLevel.ixx"
        "Src/Chat.FamilyTraits.ixx"
        "Src/Chat.Settings.ixx"
        "Src/Chat.Footprint.ixx"
        "Src/Chat.FootprintVerdict.ixx"
        "Src/Chat.ScopedLogSuppression.ixx"
        "Src/Chat.ModelCatalog.ixx"
        "Src/Chat.ResolvedModel.ixx"
        "Src/Chat.RefusalAudience.ixx"
        "Src/Chat.ModelListing.ixx"
        "Src/Chat.FootprintBudget.ixx"
        "Src/Chat.Message.ixx"
        "Src/Chat.MessageFormatter.ixx"
        "Src/Chat.SystemPrompt.ixx"
        "Src/Chat.ToolCallParser.ixx"
        "Src/Chat.ChannelParser.ixx"
        "Src/Json.ixx"
)

# Chat.Session.ixx crossed MSVC's 65535-section object limit (C1128) when it was Chat.ixx. It is one translation unit
# holding the turn loop plus model hot-switching, so it instantiates the model templates across every family and
# quantization it can switch between -- section count tracks that, and it only grows. /bigobj widens the count field
# in the COFF object; it costs nothing at runtime and changes no code generation. MSVC-only: the flag does not exist
# for clang or gcc, which have no such limit.
if(MSVC)
    target_compile_options( MilaChatSession PRIVATE /bigobj )
endif()

# The session uses no CUDA APIs directly. When CUDA is enabled, Mila propagates CUDA::cudart/nvtx3 as PUBLIC usage
# requirements, so linking Mila alone is sufficient; when CUDA is OFF those targets do not exist. Relisting them here
# would be redundant in the first case and a hard error in the second.
target_link_libraries( MilaChatSession PUBLIC Mila )

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
        "Src/Chat.ConsoleFrontEnd.ixx"
        "Src/Chat.Renderer.ixx"
        "Src/Chat.RichText.ixx"
        "Src/Chat.IncrementalRichFormatter.ixx"
        "Src/Chat.StreamingDisplay.ixx"
)

# No models directory is compiled in: the model store is the only source of weight paths,
# and it resolves them at runtime from records the build cannot know about.

target_link_libraries( mila-chat PRIVATE MilaChatSession )

# The system prompts are content, and they are the ONLY thing the build places beside the
# executable: no configuration ships, because the compiled defaults are a working configuration and
# a shipped file would outrank the user's own. This directory is where a named prompt is searched
# for after the user's config root -- see Specifications/ChatConfiguration.md section 8.
add_custom_command( TARGET mila-chat POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/Prompts"
        "$<TARGET_FILE_DIR:mila-chat>/Prompts"
)
