/**
 * @file ToolBroker.Policy.ixx
 * @brief The person's rules: for each agent and tool, allow, ask or deny.
 */

export module ToolBroker.Policy;

import std;
import ToolBroker.ToolRule;

namespace Mila::ToolBroker
{
    /**
     * @brief The rules the broker decides each call by, held in memory.
     *
     * A tool is named as the model sees it, `<server>.<tool>`. Anything without a rule is ask, so a newly registered
     * agent has nothing allowed (ToolBroker.md section 4). Where the rules are kept and how the person edits them is
     * open (section 10.4); this is what the broker reads them into.
     */
    export class Policy
    {
    public:
        void setRule( std::string agent, std::string tool, ToolRule rule )
        {
            rules_[ { std::move( agent ), std::move( tool ) } ] = rule;
        }

        ToolRule ruleFor( std::string_view agent, std::string_view tool ) const
        {
            const auto found = rules_.find( { std::string( agent ), std::string( tool ) } );

            return found == rules_.end() ? ToolRule::Ask : found->second;
        }

    private:
        std::map<std::pair<std::string, std::string>, ToolRule> rules_;
    };
}
