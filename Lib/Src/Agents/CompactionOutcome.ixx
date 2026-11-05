/**
 * @file CompactionOutcome.ixx
 * @brief What a call to compact a conversation came to.
 */

export module Agents.CompactionOutcome;

namespace Mila::Agents
{
    export enum class CompactionOutcome
    {
        /// The history was replaced by a summary.
        Compacted,

        /// Clearing earlier reasoning brought the prompt below the trigger, so no summary was written.
        ReasoningCleared,

        /// There was nothing before the turn in progress but an earlier summary. The conversation is unchanged.
        NothingToReplace,

        /// The summary failed. The conversation is unchanged.
        Failed,

        /// Stop was asked for, or the response's time ran out. The conversation is unchanged.
        Stopped,
    };
}
