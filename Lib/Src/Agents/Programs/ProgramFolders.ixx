/**
 * @file ProgramFolders.ixx
 * @brief What the application gives the programs a grant runs: the folders their tools live in, and variables of
 *        their environment.
 */

export module Agents.ProgramFolders;

import std;

namespace Mila::Agents
{
    /**
     * @brief The application's part of a run's setup (ContainedPrograms.md 4.6, 4.7), never the model's.
     *
     * The folders are given to the run to read and put on its `PATH`. The variables are the ones a tool needs that
     * the container does not write itself -- a compiler's `INCLUDE`, say -- each `NAME=value`.
     */
    export struct ProgramFolders
    {
        std::vector<std::filesystem::path> folders;
        std::vector<std::string> environment;
    };

    /**
     * @brief The variables the container writes for every run, which the application may not give: the system's
     *        own, `PATH`, and the folders that hold a run's home, temporary files and settings.
     *
     * Matched without regard to case, as Windows matches them.
     */
    export inline bool isContainerVariable( std::string_view name ) noexcept
    {
        static constexpr std::array<std::string_view, 17> kNames{
            "SystemRoot", "windir", "ComSpec", "PATHEXT", "PATH", "TEMP", "TMP", "LOCALAPPDATA", "USERPROFILE", "HOME",
            "APPDATA", "ProgramFiles", "ProgramFiles(x86)", "CommonProgramFiles", "ProgramData", "ALLUSERSPROFILE",
            "SystemDrive" };

        return std::ranges::any_of( kNames, [name]( std::string_view owned )
            {
                return owned.size() == name.size() && std::ranges::equal( owned, name, []( char left, char right )
                    {
                        return std::tolower( static_cast<unsigned char>( left ) ) == std::tolower( static_cast<unsigned char>( right ) );
                    } );
            } );
    }
}
