/**
 * @file ToolPermission.ixx
 * @brief What a rule says about a tool call: allow, ask or deny.
 */

export module AI.ToolPermission;

import std;

namespace Mila::AI
{
    /**
     * @brief A rule's permission (ToolCalling.md 3.3).
     */
    export enum class ToolPermission
    {
        /// The call runs without asking.
        Allow,

        /// The person is asked before the call runs; with no one to ask, it does not run.
        Ask,

        /// The call never runs, and the model reads that the rules do not allow it.
        Deny,
    };
}
