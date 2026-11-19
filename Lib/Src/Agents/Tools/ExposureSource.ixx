/**
 * @file ExposureSource.ixx
 * @brief One call whose result, from outside the program, exposed a conversation.
 */

export module Agents.ExposureSource;

import std;

namespace Mila::Agents
{
    /**
     * @brief What a conversation read from outside, and when (ToolCalling.md 3.3, *What a result may cause*).
     */
    export struct ExposureSource
    {
        /// The tool whose result it was.
        std::string tool;

        /// The call's arguments as the model wrote them; empty when the conversation was reopened without them.
        std::string arguments;

        /// When the call ran; empty when the conversation was reopened without it.
        std::optional<std::chrono::system_clock::time_point> when;

        bool operator==( const ExposureSource& ) const = default;
    };
}
