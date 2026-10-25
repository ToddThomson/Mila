/**
 * @file Chat.Message.ixx
 * @brief Chat's names for the runtime's conversation history.
 *
 * The history a chat template renders is Mila::Dnn::Conversation::Turn, in the runtime, because
 * the templates that render it are: a protocol module cannot take an adaptor's message class
 * without the adaptor becoming a dependency of the library. These aliases keep Chat's own
 * vocabulary (MessageRole / ToolCall / ChatMessage) pointing at that one type, so nothing in
 * the harness converts a history on its way to a template.
 */

export module Chat.Message;

import std;
export import Dnn.Models.Conversation;

namespace Mila::ChatApp
{
    /// @brief Role of a participant in a conversation turn.
    export using MessageRole = Mila::Dnn::Conversation::Role;

    /// @brief A tool call issued by the model in an assistant turn.
    export using ToolCall = Mila::Dnn::Conversation::ToolCall;

    /// @brief A single turn in a structured conversation.
    export using ChatMessage = Mila::Dnn::Conversation::Turn;
}
