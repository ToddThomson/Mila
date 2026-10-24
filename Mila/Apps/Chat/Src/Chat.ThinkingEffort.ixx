/**
 * @file Chat.ThinkingEffort.ixx
 * @brief The 1-5 thinking-effort scale: each point's name and the reasoning tokens it budgets.
 */

export module Chat.ThinkingEffort;

import std;

namespace Mila::ChatApp
{
    /**
     * @brief A point on the thinking-effort scale: its name and the reasoning tokens a context should leave room for.
     *
     * The model reads the level only where its template has levels: Qwen 3.8's reasoning_effort. Gemma 4's
     * thinking is a switch, so its prompt carries no effort at all.
     */
    export struct ThinkingEffort
    {
        int              budget;
        std::string_view name;
    };

    inline constexpr ThinkingEffort kThinkingEfforts[ 5 ] = {
        { 128,  "minimal" },
        { 256,  "low" },
        { 512,  "balanced" },
        { 1024, "high" },
        { 2048, "exhaustive" },
    };

    /// Resolve a 1..5 effort level (clamped) to its scale entry.
    export constexpr const ThinkingEffort& thinkingEffort( int level )
    {
        return kThinkingEfforts[ std::clamp( level, 1, 5 ) - 1 ];
    }
}
