/**
 * @file Chat.Message.ixx
 * @brief Chat's names for the runtime's conversation history, plus Llama's role spelling.
 *
 * The history a chat template renders is Mila::Dnn::Conversation::Turn, in the runtime, because
 * the templates that render it are: a protocol module cannot take an adaptor's message class
 * without the adaptor becoming a dependency of the library. These aliases keep Chat's own
 * vocabulary (MessageRole / ToolCall / ChatMessage) pointing at that one type, so nothing in
 * the harness converts a history on its way to a template.
 *
 * What stays here is what only Llama needs: messageRoleToString spells Tool as "ipython",
 * which is Llama 3.x's role header and no other family's.
 */

module;
#include <string_view>

export module Chat.Message;

export import Dnn.Models.Conversation;

namespace Mila::ChatApp
{
    /// @brief Role of a participant in a conversation turn.
    export using MessageRole = Mila::Dnn::Conversation::Role;

    /// @brief A tool call issued by the model in an assistant turn.
    export using ToolCall = Mila::Dnn::Conversation::ToolCall;

    /// @brief A single turn in a structured conversation.
    export using ChatMessage = Mila::Dnn::Conversation::Turn;

    /**
     * @brief Returns the Llama 3.x role header token for a MessageRole value.
     *
     * Llama's spelling, not a universal one -- Tool is "ipython" here, where Gemma has no such
     * role at all and Qwen renders a tool result as a user turn. Used by MessageFormatter, which
     * builds Llama's template, and for diagnostics.
     */
    export constexpr std::string_view messageRoleToString( MessageRole role )
    {
        switch ( role )
        {
            case MessageRole::System:    return "system";
            case MessageRole::User:      return "user";
            case MessageRole::Assistant: return "assistant";
            case MessageRole::Tool:      return "ipython";
        }

        return "unknown";
    }
}
