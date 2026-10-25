/**
 * @file AI.Tools.ToolResult.ixx
 * @brief What a tool call gives back to the model.
 */

export module AI.Tools.ToolResult;

import std;

namespace Mila::AI
{
    /**
     * @brief A call's result as text, and whether it failed.
     *
     * A call that did not run -- refused by the rules, declined by the person, or an unknown tool -- is a failed
     * result saying so, which the model can act on, not an exception.
     */
    export struct ToolResult
    {
        std::string text;

        bool is_error = false;
    };
}
