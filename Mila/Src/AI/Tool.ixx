/**
 * @file Tool.ixx
 * @brief A tool an agent may call: what the model is told about it, and the function that runs it.
 */

export module AI.Tool;

import std;

namespace Mila::AI
{
    /**
     * @brief One tool, registered on an agent (ToolCalling.md 3.1).
     *
     * The name, description and parameters are prompt text the model reads. The function is the application's own
     * code, run in its process; what it throws reaches the model as an error result, never the application.
     */
    export struct Tool
    {
        std::string name;
        std::string description;

        /// The arguments' JSON Schema, an object, as the chat template declares it to the model.
        std::string parameters;

        /// Runs one call: the arguments as the model wrote them, a JSON object; returns the result the model reads.
        std::function<std::string( std::string_view arguments )> run;
    };
}
