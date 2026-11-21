/**
 * @file Assistant.ProgramSetup.Posix.ixx
 * @brief Assistant.ProgramSetup where commands do not run: their container is Windows only (ContainedPrograms.md 4.11).
 *
 * One of two candidate files for Assistant.ProgramSetup; this one is compiled everywhere but Windows.
 */

export module Assistant.ProgramSetup;

import std;

import Mila;

namespace Mila::Assistant
{
    /**
     * @brief The build tools a command may use: none, since no command runs here.
     *
     * The session asks for these only once a container has opened, which it does not off Windows, so these answer
     * a question that is never asked; they keep the session one program on every platform.
     */
    export class ProgramSetup
    {
    public:

        [[nodiscard]] static Mila::Agents::ProgramFolders find()
        {
            return {};
        }

        [[nodiscard]] static Mila::Agents::ProgramFolders forProject( Mila::Agents::ProgramFolders programs, const std::filesystem::path& )
        {
            return programs;
        }

        static void writeGitIdentity( const std::filesystem::path& )
        {
        }
    };
}
