/**
 * @file FolderEntry.ixx
 * @brief One file or folder found in a folder through a grant.
 */

export module AI.FolderEntry;

import std;

namespace Mila::AI
{
    /**
     * @brief A name in a folder, as the folder lists it.
     *
     * A listed name has not been opened, so what a link among them leads to is not known until it is.
     */
    export struct FolderEntry
    {
        /// The name, in UTF-8.
        std::string name;

        bool is_folder{ false };

        /// Whether the name is a link -- a symbolic link, junction or mount point -- leading elsewhere.
        bool is_link{ false };

        /// The size of a file; zero for a folder.
        std::uint64_t bytes{ 0 };
    };
}
