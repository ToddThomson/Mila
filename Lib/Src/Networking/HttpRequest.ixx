/**
 * @file HttpRequest.ixx
 * @brief A request as HttpClient is asked it: the URL, its headers and body, a token, a range, and its bounds.
 */

export module Networking.HttpRequest;

import std;

export import Networking.HttpHeader;

namespace Mila::Networking
{
    export struct HttpRequest
    {
        std::string url;

        /// Sent on every hop, to every host the request reaches, so never a secret: a token goes in `token`. A
        /// request naming Authorization here is refused.
        std::vector<HttpHeader> headers;

        /// Bearer token, or empty for an anonymous request. Never logged, and never sent to
        /// a host other than the one the request started on.
        std::string token;

        /// A POST of these bytes when set, a GET when not. A request with a body follows no redirect: a redirect
        /// answer is reported as RedirectRefused.
        std::optional<std::string> body;

        /// Byte offset to resume from. Non-zero sends a Range header; a 200 response then
        /// means the server ignored it and the caller must restart.
        std::uint64_t resume_from{ 0 };

        /// The last byte to read, inclusive, for a part of a file: a Range is then sent even from offset zero, and
        /// a 200 means the server ignored it. Unset reads to the end.
        std::optional<std::uint64_t> last_byte;

        /// The most body bytes read: the transfer stops there, the sink has the body's start, and the result says
        /// `cut`. Zero reads the whole body.
        std::uint64_t maximum_body_bytes{ 0 };

        /// Redirect hops permitted before giving up.
        int maximum_redirects{ 5 };

        /// Follow a redirect only on the host the request started on, and never from https to http. Another is
        /// reported as RedirectLeftSite, with where it led in the result's `redirect_target`.
        bool redirects_stay_on_site{ false };

        /// Seconds with no data before the transfer is abandoned. Zero disables.
        long low_speed_timeout_seconds{ 60 };
    };
}
