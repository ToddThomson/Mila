/**
 * @file ToolBroker.ToolRule.ixx
 * @brief What the person's rules say about one agent calling one tool.
 */

export module ToolBroker.ToolRule;

import std;

namespace Mila::ToolBroker
{
    /**
     * @brief Allow, ask or deny (ToolBroker.md section 4).
     */
    export enum class ToolRule
    {
        /** The call runs without asking. */
        Allow,

        /** The person is asked before the call runs; an agent with nobody to ask is refused. */
        Ask,

        /** The call never runs, and the model is told so. */
        Deny
    };
}
