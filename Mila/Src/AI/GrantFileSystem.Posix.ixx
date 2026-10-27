/**
 * @file GrantFileSystem.Posix.ixx
 * @brief The grant's file calls where they are not written yet: every one refuses, so no folder can be granted.
 *
 * One of two candidate files for AI.GrantFileSystem; CMake compiles this one off Windows. Mila never claims
 * containment it does not enforce (ToolCalling.md 3.4), so a grant here holds no folder and a tool reaches none.
 */

export module AI.GrantFileSystem;

import std;
import AI.FolderEntry;

namespace Mila::AI::GrantFileSystem
{
    export using NativeHandle = std::uintptr_t;

    export inline constexpr NativeHandle kNoHandle = 0;

    /// Whether this build can hold a tool to a folder.
    export inline constexpr bool kEnforcesFolders = false;

    namespace
    {
        [[noreturn]] void refuse()
        {
            throw std::runtime_error( "a grant of folders is enforced only on Windows so far, so none is granted here" );
        }
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
