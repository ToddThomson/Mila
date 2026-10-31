/**
 * @file ProgramContainer.ixx
 * @brief What a grant runs a program through: one run at a time held to what the grant decided.
 */

export module AI.ProgramContainer;

import std;
import AI.ContainedRun;
import AI.ProgramResult;

namespace Mila::AI
{
    /**
     * @brief The container a grant holds, as it holds its transport (ContainedPrograms.md 3).
     *
     * A run, and everything it starts, reaches the run's folders as given, its program folders to read, and the
     * network only when given it, public addresses alone. Nothing it starts outlives it.
     */
    export class ProgramContainer
    {
    public:

        virtual ~ProgramContainer() = default;

        /**
         * @brief Runs a program and waits for it, ending it when its time runs out or stop is requested.
         *
         * @throws std::runtime_error when it could not be started, or when its folders could not be checked within
         *         its time; the message says which.
         */
        virtual ProgramResult run( const ContainedRun& run, std::stop_token stop ) const = 0;
    };
}
