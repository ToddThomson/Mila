/**
 * @file Conversation.Reply.ixx
 * @brief One model turn, parsed out of the template's grammar.
 */

export module Dnn.Models.Conversation.Reply;

import std;
export import Dnn.Models.Conversation.ToolCall;

namespace Mila::Dnn::Conversation
{
    /**
     * @brief A model's reply: its reasoning, what the user is shown, and every call it made.
     */
    export struct Reply
    {
        /// The reasoning channel's text, empty when there was none.
        std::string reasoning;

        /// What the user is shown: the reply with its reasoning and call spans removed.
        std::string content;

        /// Every call, in the order the model made them.
        std::vector<ToolCall> tool_calls;

        /**
         * @brief Each call span the grammar could not read, as the model wrote it.
         *
         * An unclosed span included. Never dropped and never thrown: what a malformed call does
         * to the turn is the consumer's decision, and that it learns of it is not.
         */
        std::vector<std::string> malformed_tool_calls;
    };
}
