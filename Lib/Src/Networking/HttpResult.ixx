/**
 * @file HttpResult.ixx
 * @brief What HttpClient reports of a request: the outcome, the bytes, and the URL that answered.
 */

export module Networking.HttpResult;

import std;

export import Networking.HttpStatus;

namespace Mila::Networking
{
    export struct HttpResult
    {
        HttpStatus status{ HttpStatus::TransportError };
        long http_code{ 0 };
        std::uint64_t bytes_received{ 0 };

        /// Server-reported total for this transfer, excluding any resumed prefix. Zero if absent.
        std::uint64_t content_length{ 0 };

        /// Populated on failure. Never contains the authorization token.
        std::string message;

        /// The URL actually being fetched when the result was produced, which after a
        /// redirect is not the one requested. Reporting the requested URL instead hides
        /// which hop failed.
        std::string final_url;

        /// Where a redirect not followed led, for RedirectLeftSite; empty otherwise.
        std::string redirect_target;

        /// The body reached the request's `maximum_body_bytes` and the transfer stopped there: what the sink has is
        /// the body's start, not all of it.
        bool cut{ false };

        bool ok() const noexcept { return status == HttpStatus::Ok; }
    };
}
