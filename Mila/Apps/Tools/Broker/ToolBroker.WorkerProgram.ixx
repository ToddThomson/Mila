/**
 * @file ToolBroker.WorkerProgram.ixx
 * @brief The program a worker runs and the arguments it is started with.
 */

export module ToolBroker.WorkerProgram;

import std;

namespace Mila::ToolBroker
{
    /**
     * @brief A worker's executable and its arguments.
     *
     * Each argument arrives in the worker's `argv` unchanged: the launcher quotes it, so a space, a quote or a
     * trailing backslash is never reinterpreted.
     */
    export struct WorkerProgram
    {
        std::filesystem::path executable;
        std::vector<std::wstring> arguments;
    };
}
