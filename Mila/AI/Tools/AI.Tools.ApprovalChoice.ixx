/**
 * @file AI.Tools.ApprovalChoice.ixx
 * @brief The person's answer when the tool broker asks before a call runs.
 */

export module AI.Tools.ApprovalChoice;

import std;

namespace Mila::AI
{
    /**
     * @brief The three choices of ToolBroker.md section 5; making a rule permanent is done in the rules, not here.
     */
    export enum class ApprovalChoice
    {
        AllowOnce,

        /** Every later call of this tool in the same conversation runs without asking. */
        AllowForConversation,

        Deny
    };
}
