/**
 * @file ProgramRequest.ixx
 * @brief What a tool asks its grant to run: a command line, the folder it starts in, its time and how much output
 *        comes back.
 */

export module AI.ProgramRequest;

import std;

namespace Mila::AI
{
    /**
     * @brief One program a tool asks to run through its grant (ContainedPrograms.md 4.1).
     *
     * What the program reaches is never the request's: the folders, how, and whether the network are the grant's.
     */
    export struct ProgramRequest
    {
        /// The command line, run as written, the program first.
        std::string command_line;

        /// The folder it starts in, in a form the grant accepts; empty is the first folder granted.
        std::string folder;

        /// The most the run may take, the check of its folders included.
        std::chrono::milliseconds time{ std::chrono::minutes( 2 ) };

        /// The most of standard output, and the most of standard error, kept; the rest is read and dropped.
        std::size_t output_bytes{ 64 * 1024 };
    };
}
