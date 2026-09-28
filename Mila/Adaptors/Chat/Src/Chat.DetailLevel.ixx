/**
 * @file Chat.DetailLevel.ixx
 * @brief How much of a turn's internal activity Chat shows, and its keyword.
 */

module;
#include <optional>
#include <string_view>

export module Chat.DetailLevel;

namespace Mila::ChatApp
{
    /**
     * @brief Display-verbosity ladder for a chat turn. Each level includes the lower ones.
     *
     * Controls how much of the model's internal activity is shown -- independent of
     * whether thinking mode is active (that is the model-side toggle). Off shows the
     * answer plus the always-visible agentic trace (tool calls); Thoughts adds the
     * reasoning channel; All adds raw model output plus INFO logging and load dumps.
     */
    export enum class DetailLevel
    {
        Off,       ///< Answer + agentic trace (tool calls).
        Thoughts,  ///< + the reasoning channel.
        All,       ///< + raw model output, INFO logging, and model/memory load dumps.
    };

    /**
     * @brief Parse a detail keyword ("off"/"thoughts"/"all") to a DetailLevel.
     */
    export constexpr std::optional<DetailLevel> parseDetailLevel( std::string_view s )
    {
        if ( s.empty() || s == "off" || s == "none" ) return DetailLevel::Off;
        if ( s == "thoughts" )                        return DetailLevel::Thoughts;
        if ( s == "all" )                             return DetailLevel::All;
        return std::nullopt;
    }

    /**
     * @brief Display name for a DetailLevel.
     */
    export constexpr std::string_view detailLevelName( DetailLevel level )
    {
        switch ( level )
        {
            case DetailLevel::Off:      return "off";
            case DetailLevel::Thoughts: return "thoughts";
            case DetailLevel::All:      return "all";
        }

        return "off";
    }
}
