/**
 * @file Mila_py.Wrappers.ProtocolTokens.ixx
 * @brief The control tokens a family's grammar is built from.
 */

export module Mila.Bindings:ProtocolTokens;

import std;

namespace Mila::Bindings
{
    /**
     * @brief The control tokens a family's grammar is built from.
     *
     * Exposed because a host that streams has to recognise them as they arrive -- generation
     * stops at a closing tool-call marker, or the model fabricates the result itself. Reported
     * from the runtime rather than written down again, which is the whole point of this section.
     */
    export struct ProtocolTokens
    {
        std::string turn_open;
        std::string turn_close;
        std::string reasoning_open;
        std::string reasoning_close;
        std::string tool_call_open;
        std::string tool_call_close;
        std::string tool_response_open;
        std::string tool_response_close;
    };
}
