/**
 * @file ArgumentCondition.ixx
 * @brief One condition a rule puts on a tool call's arguments.
 */

export module AI.ArgumentCondition;

import std;

namespace Mila::AI
{
    /**
     * @brief A condition on one argument, which holds only when the call has that argument.
     *
     * A string argument is compared as its text and any other value as its JSON text, so `true` or `3`. The
     * comparison is on text: a pattern on a path decides whether a call is asked about, and never keeps a tool
     * inside a folder, which is what the tool's grant does (ToolCalling.md 3.4).
     */
    export struct ArgumentCondition
    {
        enum class Test
        {
            /// The argument is one of the values.
            IsOneOf,

            /// The argument matches one of the values as a pattern, where * is any run of characters.
            Matches,
        };

        /// The argument as a JSON Pointer into the arguments: "/path", or "/options/recursive".
        std::string argument;

        Test test{ Test::IsOneOf };

        std::vector<std::string> values;
    };
}
