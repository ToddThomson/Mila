/**
 * @file Mila_py.ChatProtocol.ixx
 * @brief One chat template's protocol, with a std-only surface for Mila_py.cpp to bind.
 *
 * Kept apart from Mila.Bindings for the same reason that module exists: Mila_py.cpp must not import Mila, so the
 * Mila types stay in the implementation unit (Mila_py.ChatProtocol.cpp). Conversations and replies cross as JSON
 * text, which keeps this surface to one type.
 */

export module Mila.Bindings.ChatProtocol;

import std;

namespace Mila::Bindings
{
    /**
     * @brief A checkpoint's chat template: render a conversation, parse a reply (ModelHandle.md 3.4).
     *
     * One factory per template the library implements. Choosing which one a loaded model uses is the caller's
     * until the model handle names it.
     */
    export class ChatProtocolHandle
    {
    public:

        /**
         * @brief The template an installed model runs with, from the architecture its manifest declares.
         *
         * @throws std::invalid_argument for a model not in the store, a base model, which has no chat template,
         *         or an architecture the library has no template for.
         */
        static ChatProtocolHandle forInstalledModel( const std::string& name );

        static ChatProtocolHandle gemma();
        static ChatProtocolHandle qwen();
        static ChatProtocolHandle llama31();
        static ChatProtocolHandle llama32();

        /**
         * @brief Render a conversation as the template's prompt.
         *
         * @param history_json A JSON array of turns: {role, content, tool_calls: [{name, arguments}], tool_name,
         *        reasoning}, role one of system, user, assistant and tool.
         * @param template_variables Template variables by the template's own names.
         * @param tools_json A JSON array of tool schemas, or empty for none.
         * @return The prompt, and whether the reply begins inside the reasoning channel.
         * @throws std::invalid_argument for a turn, variable or history the template does not accept.
         */
        std::pair<std::string, bool> formatPrompt( const std::string& history_json,
            const std::map<std::string, std::string>& template_variables, const std::string& tools_json ) const;

        /// The reply as JSON: {reasoning, content, tool_calls: [{name, arguments}], malformed_tool_calls}.
        std::string parseReply( const std::string& text, bool reply_begins_in_reasoning ) const;

        std::string formatToolResult( const std::string& name, const std::string& result ) const;

        /// The markers as JSON: {reasoning_open, reasoning_close, tool_call_open, tool_call_close, tool_response_open, stop_after}.
        std::string markersJson() const;

        /// The template's variables as JSON: [{name, values, default}].
        std::string templateVariablesJson() const;

        /// Whether a system turn after the first renders where it stands; formatPrompt refuses one where it does not.
        bool rendersLaterSystemTurns() const;

    private:

        struct Implementation;
        std::shared_ptr<const Implementation> implementation_;

        explicit ChatProtocolHandle( std::shared_ptr<const Implementation> implementation );
    };
}
