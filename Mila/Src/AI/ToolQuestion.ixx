/**
 * @file ToolQuestion.ixx
 * @brief What the person is shown when a tool call needs their answer.
 */

export module AI.ToolQuestion;

import std;
import AI.ToolNeeds;

namespace Mila::AI
{
    /**
     * @brief A call the person must allow or decline, built from the tool as registered (ToolCalling.md 3.3).
     *
     * The name, source, summary and needs are the application's, never text the model wrote; the arguments are the
     * model's, in full. Valid for the duration of the question only.
     */
    export struct ToolQuestion
    {
        std::string_view name;
        std::string_view source;
        std::string_view summary;

        /// The call's arguments as the model wrote them, a JSON object.
        std::string_view arguments;

        /// What the call will touch.
        const ToolNeeds& needs;
    };
}
