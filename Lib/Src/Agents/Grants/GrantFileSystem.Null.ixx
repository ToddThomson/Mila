/**
 * @file GrantFileSystem.Null.ixx
 * @brief The grant's file calls where they are not written: every one refuses, so no folder can be granted.
 *
 * One of three candidate files for Agents.GrantFileSystem; CMake compiles this one off Windows and Apple platforms.
 * Mila never claims containment it does not enforce (ToolCalling.md 3.4), so a grant here holds no folder and a tool
 * reaches none.
 */

export module Agents.GrantFileSystem;

import std;
import Agents.FolderEntry;
import Agents.GrantPathForm;

namespace Mila::Agents::GrantFileSystem
{
    export using NativeHandle = std::uintptr_t;

    export inline constexpr NativeHandle kNoHandle = 0;

    /// Whether this build can hold a tool to a folder.
    export inline constexpr bool kEnforcesFolders = false;

    /// The form of path a grant accepts here.
    export inline constexpr GrantPathForm kPathForm = GrantPathForm::Posix;

    namespace
    {
        [[noreturn]] void refuse()
        {
            throw std::runtime_error( "a grant of folders is not enforced on this platform, so none is granted here" );
        }
    }

    export std::filesystem::path rootPath( std::string_view )
    {
        refuse();
    }

    export std::optional<NativeHandle> openExisting( const std::filesystem::path&, bool )
    {
        refuse();
    }

    export std::optional<NativeHandle> openChild( NativeHandle, const std::filesystem::path&, bool, bool )
    {
        refuse();
    }

    export NativeHandle createChild( NativeHandle, const std::filesystem::path&, bool )
    {
        refuse();
    }

    export std::filesystem::path finalPath( NativeHandle )
    {
        refuse();
    }

    export bool isFolder( NativeHandle )
    {
        refuse();
    }

    export std::uint32_t linkCount( NativeHandle )
    {
        refuse();
    }

    export std::uint64_t size( NativeHandle )
    {
        refuse();
    }

    export std::string read( NativeHandle, std::uint64_t )
    {
        refuse();
    }

    export void replace( NativeHandle, std::string_view )
    {
        refuse();
    }

    export std::vector<FolderEntry> list( NativeHandle )
    {
        refuse();
    }

    export void close( NativeHandle ) noexcept
    {
    }
}
