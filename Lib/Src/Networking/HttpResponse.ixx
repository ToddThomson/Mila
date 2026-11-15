/**
 * @file HttpResponse.ixx
 * @brief What a transport reports of one request: the status, the Location, the length, or how it failed.
 */

export module Networking.HttpResponse;

import std;

namespace Mila::Networking
{
    export struct HttpResponse
    {
        /// Zero when the request never completed.
        long http_code{ 0 };

        /// The Location header verbatim, empty when absent. Resolving it is the client's job.
        std::string location;

        /// Server-reported length of this response body. Zero if absent.
        std::uint64_t content_length{ 0 };

        /// Connection, TLS or I/O failure -- distinct from any HTTP status.
        bool transport_failed{ false };

        /// Detail on failure, or a bounded prefix of a non-2xx body. Never contains a token.
        std::string message;
    };
}
