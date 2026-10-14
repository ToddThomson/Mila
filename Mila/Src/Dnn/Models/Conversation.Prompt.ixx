/**
 * @file Conversation.Prompt.ixx
 * @brief A rendered prompt and what its last tokens leave open.
 */

export module Dnn.Models.Conversation.Prompt;

import std;

namespace Mila::Dnn::Conversation
{
    /**
     * @brief The text a model continues, and whether it continues inside its reasoning channel.
     *
     * The second is a fact of the rendering, not of the template: Qwen 3.8's primer leaves
     * `<think>` open while thinking is on, and Gemma 4's leaves its thought channel open only on a
     * resume after a tool result with thinking on. A reply is read against the prompt that
     * produced it, because a reply that begins inside the channel carries its close and no open.
     */
    export struct Prompt
    {
        std::string text;
        bool reply_begins_in_reasoning{ false };
    };
}
