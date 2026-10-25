/**
 * @file AI.Tools.ApprovalRequest.ixx
 * @brief What an application shows the person when the tool broker asks before a call runs.
 */

export module AI.Tools.ApprovalRequest;

import std;
import AI.Tools.ApprovalChoice;

namespace Mila::AI
{
    /**
     * @brief One call waiting on the person: the tool, what it says it does, and its arguments in full.
     *
     * The tool's name and arguments are the client's own copy of the call, not text the broker sent; `message` is
     * the broker's question.
     */
    export struct ApprovalRequest
    {
        /** As the model sees it, `<server>.<tool>`. */
        std::string tool;

        /** The tool's description from the broker's list, or empty if the tools were never listed. */
        std::string description;

        /** The arguments as JSON text, exactly as they will be sent. */
        std::string arguments;

        std::string message;

        /** Whether `ApprovalChoice::AllowForConversation` is offered; it is only for a call that names a conversation. */
        bool offers_conversation = false;
    };

    /**
     * @brief Asks the person and returns their choice. An empty one means nobody can be asked, and every call the
     *        rules leave at ask is refused.
     */
    export using Approver = std::function<ApprovalChoice( const ApprovalRequest& )>;
}
