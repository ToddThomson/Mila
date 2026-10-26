# The Mila desktop app's session process (Specifications/ChatDesktop.md section 6.1): Chat's session, spoken to as JSON
# lines on stdio by the window. It links Chat's session library, so the two front ends are one session.
add_executable( mila-session
    "Src/main.cpp"
)

set_target_properties( mila-session PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_sources( mila-session
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
        "Src/Desktop.SessionProcess.ixx"
        "Src/Desktop.ConversationStore.ixx"
)

target_link_libraries( mila-session PRIVATE MilaChatSession )

# The named system prompts, beside the executable where a prompt name is searched for (ChatConfiguration.md section 8),
# as mila-chat places them.
add_custom_command( TARGET mila-session POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../../Chat/Prompts"
        "$<TARGET_FILE_DIR:mila-session>/Prompts"
)

# What the window's package carries (ChatDesktop.md 9.4): the session, its prompts, and the CUDA libraries it loads,
# found by reading the program's imports, so a toolkit that renames them needs no edit here. Installed only on request:
#   cmake --install <build directory> --component mila-session --prefix Mila/Apps/Desktop/Window/session
# The CUDA 13 toolkit keeps its Windows libraries in bin/x64.
install( TARGETS mila-session
    RUNTIME_DEPENDENCY_SET mila-session-libraries
    RUNTIME DESTINATION . COMPONENT mila-session EXCLUDE_FROM_ALL
)

install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../Chat/Prompts"
    DESTINATION . COMPONENT mila-session EXCLUDE_FROM_ALL
)

# The filters come before the RUNTIME group: CMake 4.0 refuses them after it, whatever its documentation's order.
install( RUNTIME_DEPENDENCY_SET mila-session-libraries
    PRE_EXCLUDE_REGEXES "^api-ms-" "^ext-ms-"
    POST_EXCLUDE_REGEXES "[/\\\\][Ss]ystem32[/\\\\]" "^/lib" "^/usr/lib"
    DIRECTORIES "${CUDAToolkit_BIN_DIR}/x64" "${CUDAToolkit_BIN_DIR}" "${CUDAToolkit_LIBRARY_DIR}"
    RUNTIME DESTINATION . COMPONENT mila-session EXCLUDE_FROM_ALL
)

# The Visual C++ runtime, beside the session from the toolset's redistributable folder, so a machine whose installed
# runtime is older than this toolset's still starts it and the per-user installer needs no administrator. The set
# above cannot carry it: Windows resolution finds the System32 copies first.
if( MSVC )
    # A CMake that does not know the VS 2026 toolset (4.0 does not; Visual Studio's 4.3 does) reports it as VS 2022's
    # and would carry VS 2022's older runtime.
    if( MSVC_VERSION GREATER_EQUAL 1950 AND MSVC_TOOLSET_VERSION LESS 145 )
        message( WARNING "CMake ${CMAKE_VERSION} does not know the MSVC ${MSVC_VERSION} toolset: the mila-session component carries no Visual C++ runtime. Configure with Visual Studio's CMake." )
    else()
        set( CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE )
        include( InstallRequiredSystemLibraries )

        install( PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
            DESTINATION . COMPONENT mila-session EXCLUDE_FROM_ALL
        )
    endif()
endif()
