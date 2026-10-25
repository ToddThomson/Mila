/**
 * @file AI.Tools.BrokerTool.ixx
 * @brief One tool the broker offers, as a model is shown it.
 */

export module AI.Tools.BrokerTool;

import std;

namespace Mila::AI
{
    /**
     * @brief A tool's name, description and the JSON Schema of its arguments.
     *
     * The description and schema are the tool server's own text, which a model reads as part of its prompt.
     */
    export struct BrokerTool
    {
        /** `<server>.<tool>`, unique across the broker. */
        std::string name;

        std::string description;

        /** The JSON Schema of the arguments, as JSON text. */
        std::string input_schema;
    };
}
