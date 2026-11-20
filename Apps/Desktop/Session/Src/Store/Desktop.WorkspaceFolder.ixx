/**
 * @file Desktop.WorkspaceFolder.ixx
 * @brief Where the person's workspace is kept on each platform, and the folders and files inside it
 *        (AssistantInterface.md 8, *Where it is*).
 */

export module Desktop.WorkspaceFolder;

import std;

import Mila;

namespace Mila::Desktop
{
    /**
     * @brief The workspace when the window names none: the app's own folder, Mila, where each platform keeps an
     *        application's data.
     *
     * `%LOCALAPPDATA%\Mila` on Windows, where the model store keeps its own folders beside the workspace's and never
     * inside them; `~/Library/Application Support/Mila` on macOS; `$XDG_DATA_HOME/mila`, else `~/.local/share/mila`,
     * on Linux.
     */
    export std::filesystem::path defaultWorkspaceFolder()
    {
        using Mila::Distribution::readEnvironmentVariable;

#if defined( _WIN32 )
        if ( const auto local_app_data = readEnvironmentVariable( "LOCALAPPDATA" ) )
        {
            return std::filesystem::path( *local_app_data ) / "Mila";
        }
#elif defined( __APPLE__ )
        if ( const auto home = readEnvironmentVariable( "HOME" ) )
        {
            return std::filesystem::path( *home ) / "Library" / "Application Support" / "Mila";
        }
#else
        if ( const auto xdg_data = readEnvironmentVariable( "XDG_DATA_HOME" ) )
        {
            return std::filesystem::path( *xdg_data ) / "mila";
        }

        if ( const auto home = readEnvironmentVariable( "HOME" ) )
        {
            return std::filesystem::path( *home ) / ".local" / "share" / "mila";
        }
#endif

        return std::filesystem::temp_directory_path() / "mila";
    }

    /// One JSON file per conversation (MilaDesktop.md 6.3); Linux names its folders in lower case, as its own do.
    export std::filesystem::path conversationsFolder( const std::filesystem::path& workspace )
    {
#if defined( _WIN32 ) || defined( __APPLE__ )
        return workspace / "Conversations";
#else
        return workspace / "conversations";
#endif
    }

    /// One folder per project, holding its record and its history.
    export std::filesystem::path projectsFolder( const std::filesystem::path& workspace )
    {
#if defined( _WIN32 ) || defined( __APPLE__ )
        return workspace / "Projects";
#else
        return workspace / "projects";
#endif
    }

    /// The workspace's events that belong to no project.
    export std::filesystem::path activityFile( const std::filesystem::path& workspace )
    {
        return workspace / "activity.jsonl";
    }
}
