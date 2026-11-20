# Mila Desktop's session process (Specifications/MilaDesktop.md section 6.1): an agent and its session,
# spoken to as JSON lines on stdio by the window. It links the Mila library alone.
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
        "Src/Desktop.ResidentAgent.ixx"
        "Src/Settings/Desktop.Settings.ixx"
        "Src/Settings/Desktop.SettingsFile.ixx"
        "Src/Settings/Desktop.MilaLicense.ixx"
        "Src/Tools/Desktop.PromptTool.ixx"
        "Src/Tools/Desktop.PromptPart.ixx"
        "Src/Tools/Desktop.PromptFile.ixx"
        "Src/Desktop.Models.ixx"
        "Src/Desktop.RoundTimer.ixx"
        "Src/Desktop.MemoryHolder.ixx"
        "Src/Store/Desktop.ConversationStore.ixx"
        "Src/Store/Desktop.ProjectStore.ixx"
        "Src/Store/Desktop.EventLog.ixx"
        "Src/Store/Desktop.StoreFile.ixx"
        "Src/Store/Desktop.WorkspaceFolder.ixx"
        "Src/Desktop.ConversationSummary.ixx"
        "Src/Desktop.Creations.ixx"
        "Src/Desktop.ExposureRecord.ixx"
        "Src/Tools/Desktop.CommandTool.ixx"
        "Src/Tools/Desktop.TaskPlan.ixx"
        "Src/Settings/Desktop.SearchKeyFile.ixx"
        "Src/Tools/Desktop.WebSearch.ixx"
)

target_sources( mila-session PRIVATE
    "Src/Settings/Implementation/MilaLicense.cpp"
    "Src/Implementation/SessionProcess.cpp"
    "Src/Store/Implementation/ConversationStore.cpp"
    "Src/Store/Implementation/ProjectStore.cpp"
    "Src/Store/Implementation/EventLog.cpp"
    "Src/Store/Implementation/StoreFile.cpp"
    "Src/Implementation/ConversationSummary.cpp"
    "Src/Implementation/Creations.cpp"
    "Src/Implementation/ExposureRecord.cpp"
)

# Desktop.ProgramSetup has two candidate files and exactly one is compiled, as Agents.MxcProgramContainer does:
# commands run in a Windows container, so elsewhere there are no build tools to find.
if( WIN32 )
    target_sources( mila-session PUBLIC FILE_SET module_files TYPE CXX_MODULES FILES
        "Src/Tools/Desktop.ProgramSetup.Windows.ixx"
    )
else()
    target_sources( mila-session PUBLIC FILE_SET module_files TYPE CXX_MODULES FILES
        "Src/Tools/Desktop.ProgramSetup.Posix.ixx"
    )
endif()

include( "${CMAKE_CURRENT_SOURCE_DIR}/KeyStore.cmake" )

target_sources( mila-session PUBLIC FILE_SET module_files TYPE CXX_MODULES FILES "${MILA_SESSION_KEY_STORE}" )
target_link_libraries( mila-session PRIVATE ${MILA_SESSION_KEY_STORE_LIBRARIES} )

# The session builds the model templates of every family it can open, past MSVC's section limit (C1128) without it.
if( MSVC )
    target_compile_options( mila-session PRIVATE /bigobj )
endif()

# Mila carries CUDA's usage requirements when CUDA is on; relisting them would be an error when it is off.
target_link_libraries( mila-session PRIVATE Mila )

# The activities' prompts, beside the executable where the session looks for them.
add_custom_command( TARGET mila-session POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../Prompts"
        "$<TARGET_FILE_DIR:mila-session>/Prompts"
)

# Mila's licence beside the executable, where the session reads it: nothing is used until it is accepted
# (AssistantOnboarding.md 4.1), so a package without it does not start.
set( MILA_SESSION_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../../../License.md" )

add_custom_command( TARGET mila-session POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MILA_SESSION_LICENSE}" "$<TARGET_FILE_DIR:mila-session>/License.md"
)

# The licence text of something the session carries, compiled in or beside it, installed under its own name in
# Licences/<name>/. The name is NOTICE.md's for it -- a fetched dependency's GitHub owner/repo, or the name in its
# table of what Mila Assistant carries -- and check_pins.py --verify-notice holds the two to each other; the window's
# About shows the folder's path as the name (Apps/Desktop/Window/src/licences.js). A text that is not there fails the
# install, so no package is made without it.
function( mila_session_licence name file )
    install( FILES "${file}" DESTINATION "Licences/${name}" COMPONENT mila-session EXCLUDE_FROM_ALL )
endfunction()

mila_session_licence( nlohmann/json "${nlohmann_json_SOURCE_DIR}/LICENSE.MIT" )
mila_session_licence( richgel999/miniz "${miniz_SOURCE_DIR}/LICENSE" )

# As the top level fetches curl: a Mac's transports are its own.
if( MILA_ENABLE_NETWORKING AND NOT APPLE )
    mila_session_licence( curl/curl "${curl_SOURCE_DIR}/COPYING" )
endif()

if( MILA_HAS_METAL )
    mila_session_licence( apple/metal-cpp "${metal_cpp_SOURCE_DIR}/LICENSE.txt" )
endif()

# NVIDIA's terms ask no notice, but that the terms the application is given under be consistent with them (EULA
# 1.1.2, item 5), so the libraries carry the EULA itself.
if( MILA_HAS_CUDA )
    cmake_path( SET mila_cuda_eula NORMALIZE "${CUDAToolkit_BIN_DIR}/../EULA.txt" )
    mila_session_licence( CUDA "${mila_cuda_eula}" )
endif()

# libc++, libc++abi and libunwind are linked in on a Mac (cmake/CxxRuntime.cmake). LLVM's exception may waive the
# notice for them; its text is carried whatever that reading.
if( APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
    cmake_path( GET CMAKE_CXX_COMPILER PARENT_PATH mila_llvm_bin )
    cmake_path( SET mila_llvm_license NORMALIZE "${mila_llvm_bin}/../LICENSE.TXT" )
    mila_session_licence( LLVM "${mila_llvm_license}" )
endif()

# MXC's library beside the session, which loads it from there when commands are on (ContainedPrograms.md 4.10).
if( MILA_ENABLE_MXC AND MILA_MXC_LIBRARY )
    get_filename_component( MILA_MXC_FOLDER "${MILA_MXC_LIBRARY}" DIRECTORY )

    if( NOT EXISTS "${MILA_MXC_FOLDER}/LICENSE.md" )
        message( WARNING "No LICENSE.md beside ${MILA_MXC_LIBRARY}: the session's install, and so the installer, will fail without MXC's licence." )
    endif()

    add_custom_command( TARGET mila-session POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MILA_MXC_LIBRARY}" "$<TARGET_FILE_DIR:mila-session>/mxc_ffi.dll"
    )

    install( FILES "${MILA_MXC_LIBRARY}" DESTINATION . RENAME mxc_ffi.dll COMPONENT mila-session EXCLUDE_FROM_ALL )
    mila_session_licence( MXC "${MILA_MXC_FOLDER}/LICENSE.md" )
elseif( MILA_ENABLE_MXC )
    message( STATUS "MILA_MXC_LIBRARY names no mxc_ffi.dll: Mila Desktop will say it cannot run commands." )
endif()

# What the window's package carries (MilaDesktop.md 9.4): the session, its prompts, its licence and the third-party
# texts above, and the CUDA libraries it loads,
# found by reading the program's imports, so a toolkit that renames them needs no edit here. Installed only on request:
#   cmake --install <build directory> --component mila-session --prefix Apps/Desktop/Window/session
# The CUDA 13 toolkit keeps its Windows libraries in bin/x64.
install( TARGETS mila-session
    RUNTIME_DEPENDENCY_SET mila-session-libraries
    RUNTIME DESTINATION . COMPONENT mila-session EXCLUDE_FROM_ALL
)

install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../Prompts"
    DESTINATION . COMPONENT mila-session EXCLUDE_FROM_ALL
)

install( FILES "${MILA_SESSION_LICENSE}" DESTINATION . COMPONENT mila-session EXCLUDE_FROM_ALL )

# The filters come before the RUNTIME group: CMake 4.0 refuses them after it, whatever its documentation's order.
# macOS's own libraries are under /System and /usr/lib, which the package never carries.
set( mila_session_library_folders "" )

if( MILA_HAS_CUDA )
    set( mila_session_library_folders "${CUDAToolkit_BIN_DIR}/x64" "${CUDAToolkit_BIN_DIR}" "${CUDAToolkit_LIBRARY_DIR}" )
endif()

install( RUNTIME_DEPENDENCY_SET mila-session-libraries
    PRE_EXCLUDE_REGEXES "^api-ms-" "^ext-ms-"
    POST_EXCLUDE_REGEXES "[/\\\\][Ss]ystem32[/\\\\]" "^/lib" "^/usr/lib" "^/System/"
    DIRECTORIES ${mila_session_library_folders}
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
