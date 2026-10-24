/**
 * @file ToolBroker.WorkerLimits.ixx
 * @brief What a worker is given to work in, handed to the launcher beside its program.
 */

export module ToolBroker.WorkerLimits;

import std;

namespace Mila::ToolBroker
{
    /**
     * @brief The one description of a worker's limits, which a sandbox will enforce once it arrives.
     *
     * Today the launcher enforces the environment, and a worker's own checks keep its tools to the project area
     * (ToolBroker.md section 6.2). A sandbox (section 6.3) takes the same description and enforces all of it, which is
     * why the limits travel together rather than as launcher arguments.
     */
    export struct WorkerLimits
    {
        /**
         * The folder the worker starts in and its file tools are kept to. Empty for a worker given no files, which
         * then starts in the broker's own working directory.
         */
        std::filesystem::path project_area;

        /** The whole environment the worker sees: nothing of the broker's own is added (section 6.2). */
        std::vector<std::pair<std::wstring, std::wstring>> environment;
    };
}
