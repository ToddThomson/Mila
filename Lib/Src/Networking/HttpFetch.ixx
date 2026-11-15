/**
 * @file HttpFetch.ixx
 * @brief One HTTP request as a transport is handed it: the URL, exactly the headers to send, and any body.
 */

export module Networking.HttpFetch;

import std;

export import Networking.HttpHeader;

namespace Mila::Networking
{
    export struct HttpFetch
    {
        std::string url;

        /// Exactly what to send. A transport adds nothing and removes nothing.
        std::vector<HttpHeader> headers;

        /// When set, the request is a POST carrying exactly these bytes, and its Content-Type is one of the headers;
        /// a GET when not.
        std::optional<std::string> body;

        /// Seconds with no data before the transfer is abandoned. Zero disables.
        long low_speed_timeout_seconds{ 60 };
    };
}
