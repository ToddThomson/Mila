# The Mila desktop app's session process (Specifications/ChatDesktop.md section 6.1): Chat's session, spoken to as JSON
# lines on stdio by the window. It links Chat's session library, so the two front ends are one session.
add_executable( mila-desktop-session
    "Src/main.cpp"
)

set_target_properties( mila-desktop-session PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_sources( mila-desktop-session
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
        "Src/Desktop.SessionProcess.ixx"
        "Src/Desktop.ConversationStore.ixx"
)

target_link_libraries( mila-desktop-session PRIVATE MilaChatSession )

# The named system prompts, beside the executable where a prompt name is searched for (ChatConfiguration.md section 8),
# as mila-chat places them.
add_custom_command( TARGET mila-desktop-session POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../../Chat/Prompts"
        "$<TARGET_FILE_DIR:mila-desktop-session>/Prompts"
)
