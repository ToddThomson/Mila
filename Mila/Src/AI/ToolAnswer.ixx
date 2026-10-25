/**
 * @file ToolAnswer.ixx
 * @brief What the person answered when asked whether a tool call may run.
 */

export module AI.ToolAnswer;

import std;

namespace Mila::AI
{
    /**
     * @brief The person's answer to a ToolQuestion (ToolCalling.md 3.3).
     *
     * Making a rule permanent is a change to the rules, never an answer, so there is no "always".
     */
    export enum class ToolAnswer
    {
        /// Run this call.
        Once,

        /// Run this call, and every later call of the same tool until the conversation ends.
        ForConversation,

        /// Do not run it: the model reads that the person declined.
        Deny,
    };
}
