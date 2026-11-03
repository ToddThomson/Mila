/**
 * @file Conversation.ReplyEvent.ixx
 * @brief One thing a reply did as it streamed: text on a channel, or a span opening or closing.
 */

export module Dnn.Models.Conversation.ReplyEvent;

import std;

namespace Mila::Dnn::Conversation
{
    /**
     * @brief What a ReplyReader read from the text it was given.
     *
     * The text of every Reasoning event, in order, is the reply's reasoning, and of every Answer event its
     * content: what was streamed is what was parsed. A call's syntax is never text; it is read when it closes.
     */
    export struct ReplyEvent
    {
        enum class Kind
        {
            /// A reasoning span starts, or the reply starts inside one.
            ReasoningOpened,
            Reasoning,
            ReasoningClosed,
            Answer,

            /// The model began writing a call.
            ToolCallOpened,

            /// The call closed and was read: the reply's last call, or its last malformed one.
            ToolCallClosed,
        };

        Kind kind;

        /// Valid until the reader is next fed. Empty for every kind but Reasoning and Answer.
        std::string_view text;
    };
}
