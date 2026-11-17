/**
 * @file Desktop.PromptPart.ixx
 * @brief One passage of the system prompt, kept or left out by whether the tool it speaks of is offered.
 */

export module Desktop.PromptPart;

import std;

namespace Mila::Desktop
{
    /**
     * @brief A passage of the chat's system prompt (AssistantInterface.md 5.5).
     *
     * A prompt naming a tool not offered draws calls of it, so a passage about a tool is kept only while that tool is
     * offered, and a passage standing in for a tool only while it is not.
     */
    export struct PromptPart
    {
        std::string text;

        /// The tool that must be offered for the passage to be kept; empty for every chat.
        std::string when;

        /// The tool that must not be offered for the passage to be kept; empty for none.
        std::string unless;

        [[nodiscard]] bool keptFor( const std::set<std::string, std::less<>>& offered ) const
        {
            return ( when.empty() || offered.contains( when ) ) && ( unless.empty() || !offered.contains( unless ) );
        }
    };
}
