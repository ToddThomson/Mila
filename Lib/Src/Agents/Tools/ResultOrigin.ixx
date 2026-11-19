/**
 * @file ResultOrigin.ixx
 * @brief Where a tool's result comes from: outside the program, or the program's own answer.
 */

export module Agents.ResultOrigin;

import std;

namespace Mila::Agents
{
    /**
     * @brief Whether a tool's result may expose the conversation (ToolCalling.md 3.3, *What a result may cause*).
     */
    export enum class ResultOrigin
    {
        /// Content that may not direct the agent: a page, search results, a file or its name, a command's output.
        /// A conversation that reads one is exposed from then on.
        Outside,

        /// The program's own answer, such as a folder the person chose; it exposes nothing.
        Application,
    };
}
