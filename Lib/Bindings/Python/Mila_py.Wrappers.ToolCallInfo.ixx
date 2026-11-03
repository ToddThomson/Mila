/**
 * @file Mila_py.Wrappers.ToolCallInfo.ixx
 * @brief One call the model asked for.
 */

export module Mila.Bindings:ToolCallInfo;

import std;

namespace Mila::Bindings
{
    /**
     * @brief One call the model asked for.
     *
     * Arguments are a JSON object as text, which is what the model wrote and what a host hands
     * on. Parsing it into a typed value here could only lose something.
     *
     * No id: no template in the library renders one, so correlating a call with its result is
     * the host's business and a host that needs an id mints it where it is needed.
     */
    export struct ToolCallInfo
    {
        std::string name;
        std::string arguments;
    };
}
