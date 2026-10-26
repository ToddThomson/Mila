/**
 * @file ToolDecision.ixx
 * @brief Who decided whether a tool call ran, and what they decided.
 */

export module AI.ToolDecision;

import std;

namespace Mila::AI
{
    /**
     * @brief How a call was answered (ToolCalling.md 3.3): the record's decision and who made it.
     */
    export enum class ToolDecision
    {
        /// A rule allowed it.
        AllowedByRule,

        /// The person allowed this call.
        AllowedByPerson,

        /// The person allowed this call, and every later call of the tool in the conversation.
        AllowedByPersonForConversation,

        /// The person allowed the tool for the conversation earlier, so they were not asked.
        AllowedForConversation,

        /// A rule denied it.
        DeniedByRule,

        /// The person declined it.
        DeclinedByPerson,

        /// The rules said to ask and the program gave no way to ask, so it did not run.
        NoOneToAsk,

        /// The model called a tool it was not offered.
        NotOffered,

        /// The arguments did not match the tool's parameters, so no one was asked and it did not run.
        InvalidArguments,

        /// The response reached its limit of rounds, so it did not run.
        NotRun,
    };

    /// Whether the decision ran the call.
    export constexpr bool isAllowed( ToolDecision decision ) noexcept
    {
        return decision == ToolDecision::AllowedByRule || decision == ToolDecision::AllowedByPerson
            || decision == ToolDecision::AllowedByPersonForConversation || decision == ToolDecision::AllowedForConversation;
    }
}
