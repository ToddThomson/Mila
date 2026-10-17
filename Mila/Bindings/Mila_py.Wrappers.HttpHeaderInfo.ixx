/**
 * @file Mila_py.Wrappers.HttpHeaderInfo.ixx
 * @brief One HTTP header a transport sends.
 */

export module Mila.Bindings:HttpHeaderInfo;

import std;

namespace Mila::Bindings
{
    export struct HttpHeaderInfo
    {
        std::string name;
        std::string value;
    };
}
