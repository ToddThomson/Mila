# Mila::Agents, the library's agents API (Specifications/Direction.md 3.2): a model at work in a program, with its session,
# its tools and the loop that runs their calls. It builds on the rest of Mila, which builds without it.

set(CMAKE_CXX_SCAN_FOR_MODULES ON)

add_library( MilaAgents STATIC )
add_library( Mila::Agents ALIAS MilaAgents )

set_target_properties( MilaAgents PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options( MilaAgents PRIVATE /std:c++latest /permissive- /Zc:preprocessor /W4 )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options( MilaAgents PRIVATE -std=c++23 -fno-implicit-modules -fno-implicit-module-maps )
endif()

target_link_libraries( MilaAgents PUBLIC Mila )

# A program that makes an agent instantiates the model templates of every family, past MSVC's section limit (C1128).
if(MSVC)
    target_compile_options( MilaAgents PUBLIC /bigobj )
endif()

target_sources( MilaAgents
    PUBLIC
        FILE_SET module_files
        TYPE CXX_MODULES FILES
            "Tools/ToolNeeds.ixx"
            "Grants/FolderAccess.ixx"
            "Grants/GrantFolder.ixx"
            "Grants/FolderEntry.ixx"
            "Grants/GrantPath.ixx"
            "Grants/PublicAddress.ixx"
            "Grants/GrantFile.ixx"
            "Programs/ProgramRequest.ixx"
            "Programs/ProgramResult.ixx"
            "Programs/ProgramFolders.ixx"
            "Programs/ContainedRun.ixx"
            "Programs/ProgramContainer.ixx"
            "Grants/ToolGrant.ixx"
            "Tools/Tool.ixx"
            "ProvidedTools/FileTools.ixx"
            "ProvidedTools/HtmlText.ixx"
            "ProvidedTools/WebPageReader.ixx"
            "Tools/ToolQuestion.ixx"
            "Tools/ToolAnswer.ixx"
            "Tools/ToolPermission.ixx"
            "Tools/ArgumentCondition.ixx"
            "Tools/ToolRule.ixx"
            "Tools/ToolRules.ixx"
            "Tools/ToolDecision.ixx"
            "Tools/ToolOutcome.ixx"
            "Tools/ArgumentSchema.ixx"
            "Tools/ToolGate.ixx"
            "AgentEvent.ixx"
            "AgentLimits.ixx"
            "Response.ixx"
            "SamplingOverride.ixx"
            "Session.ixx"
            "Agent.ixx"
            "Agents.ixx"
)

# ----------------------------------------------------------------------------
# The file calls a tool's grant makes
#
# Agents.GrantFileSystem has two candidate source files and exactly one is compiled, as
# Distribution.HttpTransportBackend does. Windows holds a tool to its folders; elsewhere every call
# refuses, so no folder can be granted where none can be enforced (ToolCalling.md 3.4).
# ----------------------------------------------------------------------------
if(WIN32)
    target_sources( MilaAgents
        PUBLIC
            FILE_SET module_files
            TYPE CXX_MODULES FILES
                "Grants/GrantFileSystem.Windows.ixx"
    )

    # NtCreateFile: creating a name inside a folder already opened, rather than by walking a path again.
    target_link_libraries( MilaAgents PRIVATE ntdll )
else()
    target_sources( MilaAgents
        PUBLIC
            FILE_SET module_files
            TYPE CXX_MODULES FILES
                "Grants/GrantFileSystem.Posix.ixx"
    )
endif()

# ----------------------------------------------------------------------------
# The container a grant runs programs through (ContainedPrograms.md 4.11)
#
# A tool's programs run in Microsoft's MXC container, loaded at run time from a file the application
# names, never linked; experimental in v0.21.0 and Windows only. Off, a grant runs no programs:
# Agents.MxcProgramContainer has two candidate source files and exactly one is compiled.
# Agents.SecondNameWatch and Agents.ToolDesktop are plain Windows and build on Windows regardless.
# ----------------------------------------------------------------------------
option( MILA_ENABLE_MXC "Run a tool's programs in Microsoft's MXC container (Windows, experimental)." OFF )

# MXC's library, v1.0.1, as Microsoft's release ships it: copied beside mila-session and carried by its
# installer, with the LICENSE.md beside it when that is there. Empty, Mila Desktop runs no commands.
set( MILA_MXC_LIBRARY "" CACHE FILEPATH "mxc_ffi.dll from MXC v1.0.1, carried beside mila-session (MILA_ENABLE_MXC)." )

if(WIN32)
    target_sources( MilaAgents
        PUBLIC
            FILE_SET module_files
            TYPE CXX_MODULES FILES
                "Programs/SecondNameWatch.ixx"
                "Programs/ToolDesktop.ixx"
    )
endif()

if(WIN32 AND MILA_ENABLE_MXC)
    target_sources( MilaAgents
        PUBLIC
            FILE_SET module_files
            TYPE CXX_MODULES FILES
                "Programs/MxcProgramContainer.Windows.ixx"
    )

    # The person's local application data folder, asked of the shell (SHGetKnownFolderPath).
    target_link_libraries( MilaAgents PRIVATE shell32 ole32 )

    message( STATUS "Added program container: MXC, loaded at run time" )
else()
    target_sources( MilaAgents
        PUBLIC
            FILE_SET module_files
            TYPE CXX_MODULES FILES
                "Programs/MxcProgramContainer.Absent.ixx"
    )

    message( STATUS "Added program container: none" )
endif()
