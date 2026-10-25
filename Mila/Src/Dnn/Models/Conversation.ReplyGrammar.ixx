/**
 * @file Conversation.ReplyGrammar.ixx
 * @brief How one chat template's replies are written: where reasoning, calls and results begin and end.
 */

export module Dnn.Models.Conversation.ReplyGrammar;

import std;
export import Dnn.Models.Conversation.ToolCall;

namespace Mila::Dnn::Conversation
{
    /**
     * @brief The grammar a ReplyReader reads a reply by, as one template writes it.
     *
     * A description rather than code, so that a reply read as it streams and a reply read whole are read by
     * the same reader and cannot disagree. Only reading a call's arguments is the template's own code
     * (read_tool_call): the markers say where a call is, not how its arguments are written.
     *
     * An empty marker is one the template does not have.
     */
    export struct ReplyGrammar
    {
        std::string reasoning_open;
        std::string reasoning_close;

        /// A reasoning span whose first line is one of these opens with the channel's name: the line is not reasoning.
        std::vector<std::string> reasoning_labels;

        /// What joins two reasoning spans of one reply.
        std::string reasoning_separator{ "\n" };

        std::string tool_call_open;

        /// Empty when a call has no close: it runs to the end of the reply.
        std::string tool_call_close;

        /// Text that, opening a reply, begins a call written with no marker.
        std::string bare_tool_call_open;

        /// A result is the engine's to write; one the model writes is not part of its reply.
        std::string tool_response_open;
        std::string tool_response_close;

        /// Text after which nothing is part of the reply.
        std::vector<std::string> reply_end;

        /// Control text that is never part of the answer.
        std::vector<std::string> hidden;

        /// Whether any `<|name|>` and the bare `<|>` are control text too: Gemma 4 registers more of them
        /// than any list holds, and a model slips them into prose.
        bool hides_pipe_tokens{ false };

        /**
         * @brief Read one call span, its markers included, into a call; nothing when it is not one.
         *
         * Never throws: a span it cannot read is a malformed call, which the reply reports.
         */
        std::optional<ToolCall>( *read_tool_call )( std::string_view span ) = nullptr;
    };
}
