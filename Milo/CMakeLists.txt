# Milo, the agentic library on Mila (Specifications/Direction.md 3.2): a model at work in a program, with its session,
# its tools and the loop that runs their calls. It builds on Mila; Mila builds without it.

set(CMAKE_CXX_SCAN_FOR_MODULES ON)

add_library( Milo STATIC )
add_library( Milo::Milo ALIAS Milo )

set_target_properties( Milo PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options( Milo PRIVATE /std:c++latest /permissive- /Zc:preprocessor /W4 )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options( Milo PRIVATE -std=c++23 -fno-implicit-modules -fno-implicit-module-maps )
endif()

target_link_libraries( Milo PUBLIC Mila )

# A program that makes an agent instantiates the model templates of every family, past MSVC's section limit (C1128).
if(MSVC)
    target_compile_options( Milo PUBLIC /bigobj )
endif()

target_sources( Milo
    PUBLIC
        FILE_SET module_files
        TYPE CXX_MODULES FILES
            "Src/ToolNeeds.ixx"
            "Src/FolderAccess.ixx"
            "Src/GrantFolder.ixx"
            "Src/FolderEntry.ixx"
            "Src/GrantPath.ixx"
            "Src/PublicAddress.ixx"
            "Src/GrantFile.ixx"
            "Src/ProgramRequest.ixx"
            "Src/ProgramResult.ixx"
            "Src/ProgramFolders.ixx"
            "Src/ContainedRun.ixx"
            "Src/ProgramContainer.ixx"
            "Src/ToolGrant.ixx"
            "Src/Tool.ixx"
            "Src/FileTools.ixx"
            "Src/HtmlText.ixx"
            "Src/WebPageReader.ixx"
            "Src/ToolQuestion.ixx"
            "Src/ToolAnswer.ixx"
            "Src/ToolPermission.ixx"
            "Src/ArgumentCondition.ixx"
            "Src/ToolRule.ixx"
            "Src/ToolRules.ixx"
            "Src/ToolDecision.ixx"
            "Src/ToolOutcome.ixx"
            "Src/ArgumentSchema.ixx"
            "Src/ToolGate.ixx"
            "Src/AgentEvent.ixx"
            "Src/AgentLimits.ixx"
            "Src/Response.ixx"
            "Src/SamplingOverride.ixx"
            "Src/Session.ixx"
            "Src/Agent.ixx"
            "Src/Milo.ixx"
)

# ----------------------------------------------------------------------------
# The file calls a tool's grant makes
#
# Milo.GrantFileSystem has two candidate source files and exactly one is compiled, as
# Distribution.HttpTransportBackend does. Windows holds a tool to its folders; elsewhere every call
# refuses, so no folder can be granted where none can be enforced (ToolCalling.md 3.4).
# ----------------------------------------------------------------------------
if(WIN32)
    target_sources( Milo
        PUBLIC
            FILE_SET module_files
            TYPE CXX_MODULES FILES
                "Src/GrantFileSystem.Windows.ixx"
    )

    # NtCreateFile: creating a name inside a folder already opened, rather than by walking a path again.
    target_link_libraries( Milo PRIVATE ntdll )
else()
    target_sources( Milo
        PUBLIC
            FILE_SET module_files
            TYPE CXX_MODULES FILES
                "Src/GrantFileSystem.Posix.ixx"
    )
endif()

# ----------------------------------------------------------------------------
# The container a grant runs programs through (ContainedPrograms.md 4.11)
#
# A tool's programs run in Microsoft's MXC container, loaded at run time from a file the application
# names, never linked; experimental in v0.21.0 and Windows only. Off, a grant runs no programs:
# Milo.MxcProgramContainer has two candidate source files and exactly one is compiled.
# Milo.SecondNameWatch and Milo.ToolDesktop are plain Windows and build on Windows regardless.
# ----------------------------------------------------------------------------
option( MILO_ENABLE_MXC "Run a tool's programs in Microsoft's MXC container (Windows, experimental)." OFF )

# MXC's library, v1.0.1, as Microsoft's release ships it: copied beside mila-session and carried by its
# installer, with the LICENSE.md beside it when that is there. Empty, Mila Desktop runs no commands.
set( MILO_MXC_LIBRARY "" CACHE FILEPATH "mxc_ffi.dll from MXC v1.0.1, carried beside mila-session (MILO_ENABLE_MXC)." )

if(WIN32)
    target_sources( Milo
        PUBLIC
            FILE_SET module_files
            TYPE CXX_MODULES FILES
                "Src/SecondNameWatch.ixx"
                "Src/ToolDesktop.ixx"
    )
endif()

if(WIN32 AND MILO_ENABLE_MXC)
    target_sources( Milo
        PUBLIC
            FILE_SET module_files
            TYPE CXX_MODULES FILES
                "Src/MxcProgramContainer.Windows.ixx"
    )

    # The person's local application data folder, asked of the shell (SHGetKnownFolderPath).
    target_link_libraries( Milo PRIVATE shell32 ole32 )

    message( STATUS "Added program container: MXC, loaded at run time" )
else()
    target_sources( Milo
        PUBLIC
            FILE_SET module_files
            TYPE CXX_MODULES FILES
                "Src/MxcProgramContainer.Absent.ixx"
    )

    message( STATUS "Added program container: none" )
endif()
