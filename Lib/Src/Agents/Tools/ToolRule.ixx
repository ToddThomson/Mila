/**
 * @file ToolRule.ixx
 * @brief One of the person's rules: a tool, conditions on its arguments, and allow, ask or deny.
 */

export module Agents.ToolRule;

import std;
import Agents.ToolPermission;
import Agents.ArgumentCondition;

namespace Mila::Agents
{
    /**
     * @brief A rule that applies to a call of a tool it names when every one of its conditions holds.
     */
    export struct ToolRule
    {
        /// The tool's name, or a pattern where * is any run of characters: "*" is every tool.
        std::string tool;

        ToolPermission permission{ ToolPermission::Ask };

        /// Every condition must hold; with none the rule applies to every call of the tool.
        std::vector<ArgumentCondition> when;
    };
}
