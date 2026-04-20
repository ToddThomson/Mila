/**
 * @file Chat.Message.ixx
 * @brief Structured message types for the Mila chat application.
 *
 * Provides MessageRole, ToolCall, and ChatMessage used by MessageFormatter
 * and Chat::run() to represent a typed conversation history.
 */

module;
#include <string>
#include <vector>
#include <optional>

export module Chat.Message;

namespace Mila::ChatApp
{
    /**
     * @brief Role of a participant in a conversation turn.
     *
     * Matches the Llama 3.x instruct role header vocabulary:
     *   system    — injected context and tool definitions
     *   user      — human turn
     *   assistant — model response, may contain tool calls
     *   tool      — result returned to the model after a tool call
     */
    export enum class MessageRole
    {
        System,
        User,
        Assistant,
        Tool
    };

    /**
     * @brief A tool call issued by the model in an assistant turn.
     *
     * Populated by ToolCallParser when the model emits a <|python_tag|>-bounded
     * JSON block. name and arguments are extracted from the parsed JSON.
     * call_id is assigned by the application to correlate the call with its
     * subsequent Tool-role result message.
     */
    export struct ToolCall
    {
        std::string id;
        std::string name;
        std::string arguments;
    };

    /**
     * @brief A single turn in a structured conversation.
     *
     * Role determines how MessageFormatter renders the turn into the model's
     * instruct template. tool_calls is non-empty only on Assistant turns where
     * the model issued one or more tool calls. tool_call_id is non-empty only
     * on Tool turns and must match the ToolCall::id it is responding to.
     *
     * Ownership: content and tool fields are value types; ChatMessage is
     * cheap to move and should be held in a std::vector by the session.
     */
    export struct ChatMessage
    {
        MessageRole role;
        std::string content;
        std::vector<ToolCall> tool_calls;
        std::optional<std::string> tool_call_id;
    };

    /**
     * @brief Returns a human-readable string for a MessageRole value.
     *
     * Used by MessageFormatter to emit role header tokens and for diagnostics.
     */
    export constexpr std::string_view messageRoleToString( MessageRole role )
    {
        switch ( role )
        {
            case MessageRole::System:    return "system";
            case MessageRole::User:      return "user";
            case MessageRole::Assistant: return "assistant";
            case MessageRole::Tool:      return "tool";
        }

        return "unknown";
    }
}