/**
 * @file Mila_py.Wrappers.TurnInfo.ixx
 * @brief One turn of a conversation, in the shape a host already holds it.
 */

export module Mila.Bindings:TurnInfo;

import std;
import :ToolCallInfo;

namespace Mila::Bindings
{
    /**
     * @brief One turn of a conversation, in the shape a host already holds it.
     *
     * Role is a string -- "system", "user", "assistant", "tool" -- rather than an enum, because
     * every wire protocol a host speaks already carries it that way. An enum would make Python
     * translate twice for no gain.
     */
    export struct TurnInfo
    {
        std::string role;
        std::string content;
        std::vector<ToolCallInfo> tool_calls;
    };
}
