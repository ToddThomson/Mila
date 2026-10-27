/**
 * @file Chat.ToolDecision.ixx
 * @brief The person's answer when the session asks before running a tool (MilaApp.md 4.5).
 */

export module Chat.ToolDecision;

import std;

namespace Mila::ChatApp
{
    /**
     * @brief Run this call, run this tool for the rest of the conversation, or do not run it.
     *
     * A denied call is not run; the model reads that the person declined, and the reply continues.
     */
    export enum class ToolDecision
    {
        AllowOnce,
        AllowConversation,
        Deny,
    };
}
