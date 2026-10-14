/**
 * @file Conversation.PromptOptions.ixx
 * @brief What a prompt is rendered with besides its history: template variables and tools.
 */

export module Dnn.Models.Conversation.PromptOptions;

import std;

namespace Mila::Dnn::Conversation
{
    /**
     * @brief The settings a protocol renders a prompt with.
     */
    export struct PromptOptions
    {
        /// Template variables by the template's own names. A variable not set takes the template's
        /// default; a name or value the template does not accept is refused (Protocol::templateVariables).
        std::map<std::string, std::string> template_variables;

        /// The tools offered, as a JSON array of OpenAI-style function schemas. Each template
        /// declares them its own way; empty or `[]` offers none.
        std::string tools_json;
    };
}
