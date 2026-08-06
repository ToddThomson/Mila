/**
 * @file HttpClient.ixx
 * @brief How to ask, and how to read the answer: redirects, the token rule, resume, status.
 *
 * Every decision that used to live inside the libcurl implementation, now written once above
 * any transport and depending on nothing. That matters most for one rule: the authorization
 * header is dropped when the host changes, so no transport can leak a bearer token to a CDN,
 * because no transport is ever told one. See Specifications/HttpClient.md.
 */

module;
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

export module Distribution.HttpClient;

import Distribution.HttpTransport;

namespace Mila::Distribution
{
    /**
     * @brief Outcome of a transfer attempt, distinguishing failures that need different fixes.
     */
    export enum class HttpStatus
    {
        Ok,               ///< Transfer completed.
        Unauthorized,     ///< 401 -- no token, or the token is not valid.
        Forbidden,        ///< 403 -- token valid, but the repository's terms are not accepted.
        NotFound,         ///< 404 -- wrong coordinate, revision or file path.
        RangeIgnored,     ///< A range request drew a 200; the partial must be discarded.
        TransportError,   ///< Connection, TLS or I/O failure.
        ServerError       ///< 5xx.
    };

    export inline std::string toString( HttpStatus status )
    {
        switch ( status )
        {
            case HttpStatus::Ok:             return "Ok";
            case HttpStatus::Unauthorized:   return "Unauthorized";
            case HttpStatus::Forbidden:      return "Forbidden";
            case HttpStatus::NotFound:       return "NotFound";
            case HttpStatus::RangeIgnored:   return "RangeIgnored";
            case HttpStatus::TransportError: return "TransportError";
            case HttpStatus::ServerError:    return "ServerError";
            default:                         return "Unknown";
        }
    }

    /**
     * @brief Called as bytes arrive. Return false to abort the transfer.
     *
     * total_bytes is zero when the server does not report a length.
     */
    export using ProgressCallback =
        std::function<bool( uint64_t bytes_so_far, uint64_t total_bytes )>;

    export struct HttpResult
    {
        HttpStatus status{ HttpStatus::TransportError };
        long http_code{ 0 };
        uint64_t bytes_received{ 0 };

        /// Server-reported total for this transfer, excluding any resumed prefix. Zero if absent.
        uint64_t content_length{ 0 };

        /// Populated on failure. Never contains the authorization token.
        std::string message;

        /// The URL actually being fetched when the result was produced, which after a
        /// redirect is not the one requested. Reporting the requested URL instead hides
        /// which hop failed.
        std::string final_url;

        bool ok() const noexcept { return status == HttpStatus::Ok; }
    };

    export struct HttpRequest
    {
        std::string url;

        /// Bearer token, or empty for an anonymous request. Never logged, and never sent to
        /// a host other than the one the request started on.
        std::string token;

        /// Byte offset to resume from. Non-zero sends a Range header; a 200 response then
        /// means the server ignored it and the caller must restart.
        uint64_t resume_from{ 0 };

        /// Redirect hops permitted before giving up.
        int maximum_redirects{ 5 };

        /// Seconds with no data before the transfer is abandoned. Zero disables.
        long low_speed_timeout_seconds{ 60 };
    };

    /**
     * @brief Extract the host from a URL, for deciding whether auth may cross a redirect.
     */
    export inline std::string hostOf( const std::string& url )
    {
        const auto scheme_end = url.find( "://" );
        const size_t start = ( scheme_end == std::string::npos ) ? 0 : scheme_end + 3;
        const auto host_end = url.find_first_of( "/?#", start );

        std::string authority = ( host_end == std::string::npos )
            ? url.substr( start )
            : url.substr( start, host_end - start );

        // Strip any userinfo and port so only the host compares.
        const auto at = authority.rfind( '@' );

        if ( at != std::string::npos )
        {
            authority = authority.substr( at + 1 );
        }

        const auto colon = authority.rfind( ':' );

        if ( colon != std::string::npos && authority.find( ']' ) == std::string::npos )
        {
            authority = authority.substr( 0, colon );
        }

        return authority;
    }

    /**
     * @brief Resolve a Location header against the URL it came from.
     *
     * A Location may be relative -- RFC 7231 permits it and HuggingFace uses it, answering
     * a manifest request with a 307 to `/api/resolve-cache/...`. Handing that to a transport
     * verbatim yields a malformed URL, since a bare path has no scheme.
     *
     * Handles the four forms: absolute, protocol-relative, root-relative, and path-relative.
     */
    export std::string resolveRedirect( const std::string& base, const std::string& location )
    {
        if ( location.empty() )
        {
            return base;
        }

        // Absolute: has a scheme.
        if ( location.find( "://" ) != std::string::npos )
        {
            return location;
        }

        const auto scheme_end = base.find( "://" );
        const std::string scheme = ( scheme_end == std::string::npos )
            ? std::string( "https" )
            : base.substr( 0, scheme_end );

        // Protocol-relative: //host/path
        if ( location.starts_with( "//" ) )
        {
            return scheme + ":" + location;
        }

        const size_t authority_start = ( scheme_end == std::string::npos )
            ? 0
            : scheme_end + 3;
        const auto authority_end = base.find( '/', authority_start );

        const std::string origin = ( authority_end == std::string::npos )
            ? base
            : base.substr( 0, authority_end );

        // Root-relative: /path
        if ( location.front() == '/' )
        {
            return origin + location;
        }

        // Path-relative: resolve against the base's directory.
        std::string path = ( authority_end == std::string::npos )
            ? std::string( "/" )
            : base.substr( authority_end );

        const auto query = path.find_first_of( "?#" );

        if ( query != std::string::npos )
        {
            path = path.substr( 0, query );
        }

        const auto last_slash = path.rfind( '/' );

        path = ( last_slash == std::string::npos ) ? "/" : path.substr( 0, last_slash + 1 );

        return origin + path + location;
    }

    /**
     * @brief GET a URL, streaming the body to a sink, over whichever transport it is given.
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
     */
    export class HttpClient
    {
    public:

        explicit HttpClient( std::shared_ptr<const IHttpTransport> transport )
            : transport_( std::move( transport ) )
        {
            // Guarded here rather than at each consumer: makeDefaultHttpTransport never
            // returns null, so a null can only come from a caller passing one, and a crash
            // deep inside a redirect loop would say nothing about where it came from.
            if ( transport_ == nullptr )
            {
                throw std::runtime_error( "HttpClient requires a transport." );
            }
        }

        /// The transport underneath, for messages. "none" when this build has no client.
        std::string transportName() const { return transport_->name(); }

        HttpResult get(
            const HttpRequest& request,
            const SinkCallback& sink,
            const ProgressCallback& progress = {} ) const
        {
            HttpResult result;

            std::string url = request.url;
            const std::string origin_host = hostOf( url );

            uint64_t bytes_received = 0;
            uint64_t content_length = 0;
            bool sink_failed = false;
            bool aborted = false;

            const auto on_headers = [&content_length]( long, uint64_t length )
                {
                    content_length = length;
                };

            const auto counting_sink =
                [&]( const char* data, size_t length ) -> bool
                {
                    if ( !sink( data, length ) )
                    {
                        sink_failed = true;

                        return false;
                    }

                    bytes_received += length;

                    if ( progress )
                    {
                        // The total is the whole file, not this response: a resumed transfer
                        // reports a content-length covering only the remaining bytes.
                        const uint64_t total = ( content_length == 0 )
                            ? 0
                            : request.resume_from + content_length;

                        if ( !progress( request.resume_from + bytes_received, total ) )
                        {
                            aborted = true;

                            return false;
                        }
                    }

                    return true;
                };

            for ( int hop = 0; hop <= request.maximum_redirects; ++hop )
            {
                HttpFetch fetch;
                fetch.url = url;
                fetch.low_speed_timeout_seconds = request.low_speed_timeout_seconds;

                // THE RULE: auth travels only to the host the request started on.
                if ( !request.token.empty() && hostOf( url ) == origin_host )
                {
                    fetch.headers.push_back( { "Authorization", "Bearer " + request.token } );
                }

                if ( request.resume_from > 0 )
                {
                    fetch.headers.push_back(
                        { "Range", std::format( "bytes={}-", request.resume_from ) } );
                }

                const HttpResponse response = transport_->fetch( fetch, counting_sink, on_headers );

                result.http_code = response.http_code;
                result.bytes_received = bytes_received;
                result.content_length = content_length;
                result.final_url = url;

                if ( aborted )
                {
                    result.status = HttpStatus::TransportError;
                    result.message = "transfer aborted by progress callback";

                    return result;
                }

                if ( sink_failed )
                {
                    result.status = HttpStatus::TransportError;
                    result.message = "sink rejected data (write failed)";

                    return result;
                }

                if ( response.transport_failed )
                {
                    result.status = HttpStatus::TransportError;
                    result.message = response.message;

                    return result;
                }

                if ( isRedirect( response.http_code ) )
                {
                    if ( response.location.empty() )
                    {
                        result.status = HttpStatus::TransportError;
                        result.message = std::format(
                            "{} redirect without a Location header", response.http_code );

                        return result;
                    }

                    url = resolveRedirect( url, response.location );

                    continue;
                }

                return classify( request, response, std::move( result ) );
            }

            result.status = HttpStatus::TransportError;
            result.message = std::format( "exceeded {} redirects", request.maximum_redirects );

            return result;
        }

        /**
         * @brief GET a small resource into a string.
         *
         * For manifests and API responses. Not for artifacts -- it buffers the whole body.
         */
        HttpResult getString( const HttpRequest& request, std::string& body ) const
        {
            body.clear();

            return get( request, [&body]( const char* data, size_t length )
                {
                    body.append( data, length );

                    return true;
                } );
        }

    private:

        static bool isRedirect( long http_code )
        {
            return http_code == 301 || http_code == 302 || http_code == 303
                || http_code == 307 || http_code == 308;
        }

        static HttpResult classify(
            const HttpRequest& request,
            const HttpResponse& response,
            HttpResult result )
        {
            switch ( response.http_code )
            {
                case 200:
                    // A range was requested and the server answered with the whole file.
                    // Appending would concatenate, so the caller must discard and restart.
                    result.status = ( request.resume_from > 0 )
                        ? HttpStatus::RangeIgnored
                        : HttpStatus::Ok;

                    if ( result.status == HttpStatus::RangeIgnored )
                    {
                        result.message = "server ignored Range and sent the whole file";
                    }

                    return result;

                case 206:
                    result.status = HttpStatus::Ok;

                    return result;

                case 401:
                    result.status = HttpStatus::Unauthorized;
                    result.message = "401: no valid HuggingFace token";

                    return result;

                case 403:
                    result.status = HttpStatus::Forbidden;
                    result.message = "403: token is valid but the repository terms are not accepted";

                    return result;

                case 404:
                    result.status = HttpStatus::NotFound;
                    result.message = "404: no such repository, revision or file";

                    return result;

                default:
                    result.status = ( response.http_code >= 500 )
                        ? HttpStatus::ServerError
                        : HttpStatus::TransportError;
                    result.message = response.message.empty()
                        ? std::format( "unexpected HTTP status {}", response.http_code )
                        : std::format( "unexpected HTTP status {}: {}",
                            response.http_code, response.message );

                    return result;
            }
        }

        std::shared_ptr<const IHttpTransport> transport_;
    };
}
