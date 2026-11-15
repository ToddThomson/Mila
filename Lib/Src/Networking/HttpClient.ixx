/**
 * @file HttpClient.ixx
 * @brief How to ask, and how to read the answer: redirects, the token rule, resume, body bounds, status.
 *
 * Every decision that used to live inside the libcurl implementation, now written once above
 * any transport and depending on nothing. That matters most for one rule: the authorization
 * header is dropped when the host changes, so no transport can leak a bearer token to a CDN,
 * because no transport is ever told one. See Specifications/HttpClient.md.
 */

export module Networking.HttpClient;

import std;

export import Networking.HttpTransport;
export import Networking.HttpRequest;
export import Networking.HttpResult;

namespace Mila::Networking
{
    /**
     * @brief Called as bytes arrive. Return false to abort the transfer.
     *
     * total_bytes is zero when the server does not report a length.
     */
    export using ProgressCallback =
        std::function<bool( std::uint64_t bytes_so_far, std::uint64_t total_bytes )>;

    /**
     * @brief Extract the host from a URL, for deciding whether auth may cross a redirect.
     */
    export std::string hostOf( const std::string& url );

    /**
     * @brief Resolve a Location header against the URL it came from.
     *
     * A Location may be relative -- RFC 7231 permits it and HuggingFace uses it, answering
     * a manifest request with a 307 to `/api/resolve-cache/...`. Handing that to a transport
     * verbatim yields a malformed URL, since a bare path has no scheme.
     *
     * Handles the four forms: absolute, protocol-relative, root-relative, and path-relative.
     */
    export std::string resolveRedirect( const std::string& base, const std::string& location );

    /**
     * @brief Whether a URL is one the client fetches: its scheme is http or https, in any case.
     *
     * Every hop is held to it, the first and each redirect, so no request reaches a file or any other scheme.
     */
    export bool isWebAddress( std::string_view url );

    /**
     * @brief Fetch a URL, a GET or a POST of the request's body, streaming the reply's body to a sink, over
     *        whichever transport it is given.
     *
     * Redirects are followed here rather than by the transport, for one reason: the
     * authorization header must be dropped when the host changes. HuggingFace redirects LFS
     * files to a pre-signed CDN URL that carries its own authorization and needs no token, and
     * forwarding one there hands it to whoever operates that host. Following hops above the
     * transport is what lets that rule be written once instead of once per implementation.
     *
     * A resume request answered 200 rather than 206 is reported as RangeIgnored rather than
     * treated as success: the server is sending the whole file, so appending it to an existing
     * partial would silently concatenate.
     *
     * The status is the client's reading of the reply; what a status means to a person is the caller's to say,
     * from the status or `http_code`.
     */
    export class HttpClient
    {
    public:

        /// @throws std::runtime_error when the transport is null.
        explicit HttpClient( std::shared_ptr<const IHttpTransport> transport );

        /// The transport underneath, for messages. "none" when this build has no client.
        std::string transportName() const { return transport_->name(); }

        /// @throws std::invalid_argument when the request's own headers name Authorization.
        HttpResult fetch(
            const HttpRequest& request,
            const SinkCallback& sink,
            const ProgressCallback& progress = {} ) const;

        /**
         * @brief Fetch a small resource into a string.
         *
         * For manifests, pages and API replies. Not for weights -- it buffers the whole body.
         *
         * @throws std::invalid_argument when the request's own headers name Authorization.
         */
        HttpResult fetchString( const HttpRequest& request, std::string& body ) const;

    private:

        static bool isRedirect( long http_code );

        static HttpResult classify(
            const HttpRequest& request,
            const HttpResponse& response,
            HttpResult result );

        std::shared_ptr<const IHttpTransport> transport_;
    };
}
