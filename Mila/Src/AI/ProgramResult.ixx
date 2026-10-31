/**
 * @file ProgramResult.ixx
 * @brief What came of a program a grant ran: its exit code, its output as cut, and the files it was not given.
 */

export module AI.ProgramResult;

import std;

namespace Mila::AI
{
    /// The result of one program run through a grant (ContainedPrograms.md 4.1).
    export struct ProgramResult
    {
        /// The program's exit code; meaningless when it was stopped or ran out of time.
        std::int32_t exit_code{ 0 };

        /// Standard output, up to the request's cap, and how many bytes past it were dropped.
        std::string output;
        std::uint64_t output_dropped{ 0 };

        /// Standard error, likewise.
        std::string errors;
        std::uint64_t errors_dropped{ 0 };

        /// Whether the request's time ran out and the run was ended.
        bool timed_out{ false };

        /// Whether the call was stopped, and the run with it.
        bool stopped{ false };

        /// Files in the granted folders the run was not given, each by its name inside: each has another name
        /// outside them, or in a folder granted less (ContainedPrograms.md 4.3).
        std::vector<std::string> withheld;
    };
}
