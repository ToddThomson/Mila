/**
 * @file Assistant.WorkspaceFolder.ixx
 * @brief Where the person's workspace is kept on each platform, and the folders and files inside it
 *        (AssistantInterface.md 8, *Where it is*).
 */

export module Assistant.WorkspaceFolder;

import std;

import Mila;

namespace Mila::Assistant
{
    /**
     * @brief The workspace when the window names none: the app's area of the person's Mila folder,
     *        `%LOCALAPPDATA%\Mila\Assistant` on Windows, `~/Library/Application Support/Mila/Assistant` on macOS,
     *        `~/.local/share/mila/assistant` on Linux. The model store is the folder's `Store`, beside it.
     */
    export std::filesystem::path defaultWorkspaceFolder()
    {
        return Mila::Distribution::resolveMilaArea( "Assistant" );
    }

    /// One JSON file per conversation (MilaAssistant.md 6.3); Linux names its folders in lower case, as its own do.
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
