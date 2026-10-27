/**
 * @file GrantFolder.ixx
 * @brief One folder a grant reaches, and whether a tool may write in it.
 */

export module AI.GrantFolder;

import std;
import AI.FolderAccess;

namespace Mila::AI
{
    /**
     * @brief A folder in a grant, as the application names it (ToolCalling.md 3.4).
     *
     * Everything under the folder's own final path is in the grant: what a name leads to is decided when it is
     * opened, not by the name.
     */
    export struct GrantFolder
    {
        std::filesystem::path folder;

        FolderAccess access{ FolderAccess::Read };
    };
}
