/**
 * @file Mila_py.Wrappers.HttpResponseInfo.ixx
 * @brief What a caller-supplied HTTP transport saw.
 */

export module Mila.Bindings:HttpResponseInfo;

import std;

namespace Mila::Bindings
{
    export struct HttpResponseInfo
    {
        long http_code{ 0 };
        std::string location;
        std::uint64_t content_length{ 0 };
        bool transport_failed{ false };
        std::string message;
    };
}
