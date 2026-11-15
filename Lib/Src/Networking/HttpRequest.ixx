/**
 * @file HttpRequest.ixx
 * @brief A request as HttpClient is asked it: the URL, a token, a range, and the redirect bound.
 */

export module Networking.HttpRequest;

import std;

namespace Mila::Networking
{
    export struct HttpRequest
    {
        std::string url;

        /// Bearer token, or empty for an anonymous request. Never logged, and never sent to
        /// a host other than the one the request started on.
        std::string token;

        /// Byte offset to resume from. Non-zero sends a Range header; a 200 response then
        /// means the server ignored it and the caller must restart.
        std::uint64_t resume_from{ 0 };

        /// The last byte to read, inclusive, for a part of a file: a Range is then sent even from offset zero, and
        /// a 200 means the server ignored it. Unset reads to the end.
        std::optional<std::uint64_t> last_byte;

        /// Redirect hops permitted before giving up.
        int maximum_redirects{ 5 };

        /// Seconds with no data before the transfer is abandoned. Zero disables.
        long low_speed_timeout_seconds{ 60 };
    };
}
