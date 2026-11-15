/**
 * @file HttpHeader.ixx
 * @brief One header of an HTTP request, as a transport sends it.
 */

export module Networking.HttpHeader;

import std;

namespace Mila::Networking
{
    export struct HttpHeader
    {
        std::string name;
        std::string value;
    };
}
