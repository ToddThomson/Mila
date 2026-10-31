/**
 * @file ContainedRun.ixx
 * @brief One run as a grant hands it to its container: the command, and everything the run may reach, every folder
 *        by its own final path.
 */

export module AI.ContainedRun;

import std;
import AI.GrantFolder;

namespace Mila::AI
{
    /**
     * @brief What a container is asked to run, decided by the grant (ContainedPrograms.md 4.2).
     *
     * Every path is a final path the operating system reported when the grant was made or the request checked, so
     * the container decides nothing by a name the model gave.
     */
    export struct ContainedRun
    {
        std::string command_line;

        /// The folder the program starts in, a granted folder or one within it.
        std::filesystem::path start_folder;

        /// The granted folders and their access.
        std::vector<GrantFolder> folders;

        /// Folders the program's own tools live in, given to the run to read and to no file tool (4.6).
        std::vector<std::filesystem::path> program_folders;

        bool network{ false };

        std::chrono::milliseconds time{};

        std::size_t output_bytes{ 0 };
    };
}
