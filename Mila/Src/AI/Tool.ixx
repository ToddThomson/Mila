/**
 * @file Tool.ixx
 * @brief A tool an agent may call: what the model is told about it, what the person is shown, and the function that
 *        runs it.
 */

export module AI.Tool;

import std;
import Dnn.TensorTypes;
import AI.ToolNeeds;

namespace Mila::AI
{
    /**
     * @brief One tool, registered on an agent (ToolCalling.md 3.1).
     *
     * The name, description and parameters are prompt text the model reads. The source, summary and needs are what
     * the person is shown when asked, and the model never reads them. The function is the application's own code,
     * run in its process; what it throws reaches the model as an error result, never the application.
     */
    export struct Tool
    {
        /// Letters, digits, _ and -, the names every template and harness accepts.
        std::string name;

        std::string description;

        /// The arguments' JSON Schema, an object, as the chat template declares it to the model. A call's
        /// arguments are checked against it before anything runs (ArgumentSchema).
        std::string parameters;

        /**
         * @brief Runs one call: the arguments as the model wrote them, a JSON object; returns the result the model reads.
         *
         * The stop token is requested when the reply is stopped, or the response's or the call's time runs out, while
         * the call runs. A tool that can end early should, since what it returns after a stop does not reach the model.
         */
        std::function<std::string( std::string_view arguments, std::stop_token stop )> run;

        /// Where the tool comes from, such as the application that registers it.
        std::string source;

        /// What the tool does, written for the person; the description is written for the model.
        std::string summary;

        ToolNeeds needs;

        /// The longest one call may run; the response's tool_time when not set.
        std::optional<std::chrono::milliseconds> time;

        /// The most tokens of a result the model reads, the rest cut; the response's result_tokens when not set.
        std::optional<Dnn::dim_t> result_tokens;
    };
}
