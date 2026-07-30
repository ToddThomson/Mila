/**
 * @file HttpClient.ixx
 * @brief libcurl-backed HTTP GET with resume, manual redirects and bearer auth.
 *
 * Sized for one job: pulling multi-gigabyte model artifacts from HuggingFace. Redirects are
 * handled by hand rather than by CURLOPT_FOLLOWLOCATION because a cross-host redirect would
 * otherwise forward the authorization header to a CDN. See
 * Specifications/ModelDistribution.md.
 */

module;
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <curl/curl.h>

export module Distribution.HttpClient;

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

    /**
     * @brief Called with each chunk of body as it arrives.
     *
     * The client itself neither buffers nor hashes; a caller writing a 6 GB artifact hashes
     * and writes in this callback so the bytes are touched exactly once.
     */
    export using SinkCallback =
        std::function<bool( const char* data, size_t length )>;

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

        /// Bearer token, or empty for an anonymous request. Never logged, never forwarded
        /// across a host change.
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
     * @brief One-time libcurl global initialization.
     *
     * curl_global_init is not thread-safe and must run once before any handle exists.
     */
    inline void ensureCurlInitialized()
    {
        static const bool initialized = []
            {
                return curl_global_init( CURL_GLOBAL_DEFAULT ) == CURLE_OK;
            }();

        if ( !initialized )
        {
            throw std::runtime_error( "libcurl global initialization failed" );
        }
    }

    /**
     * @brief Extract the host from a URL, for deciding whether auth may cross a redirect.
     */
    inline std::string hostOf( const std::string& url )
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
     * a manifest request with a 307 to `/api/resolve-cache/...`. Handing that to libcurl
     * verbatim yields CURLE_URL_MALFORMAT, since a bare path has no scheme.
     *
     * Handles the four forms: absolute, protocol-relative, root-relative, and path-relative.
     */
    export std::string resolveRedirect( const std::string& base, const std::string& location )
    {
        if ( location.find( "://" ) != std::string::npos )
        {
            return location;
        }

        const auto scheme_end = base.find( "://" );

        if ( scheme_end == std::string::npos )
        {
            return location;
        }

        const std::string scheme = base.substr( 0, scheme_end );

        if ( location.starts_with( "//" ) )
        {
            return scheme + ":" + location;
        }

        const size_t authority_start = scheme_end + 3;
        const auto authority_end = base.find( '/', authority_start );

        const std::string origin = ( authority_end == std::string::npos )
            ? base
            : base.substr( 0, authority_end );

        if ( location.starts_with( "/" ) )
        {
            return origin + location;
        }

        // Path-relative: replace the last segment of the base path, ignoring any query.
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

    namespace
    {
        struct TransferState
        {
            SinkCallback* sink{ nullptr };
            ProgressCallback* progress{ nullptr };
            uint64_t bytes_received{ 0 };
            uint64_t resume_offset{ 0 };
            uint64_t content_length{ 0 };
            std::string location;
            bool sink_failed{ false };
            bool aborted{ false };

            /// True while the response being received is a redirect. Its body is a short
            /// human-readable note, and passing it to the sink would prepend it to the
            /// content fetched from the next hop.
            bool discarding{ false };
        };

        size_t writeBody( char* data, size_t size, size_t count, void* user_data )
        {
            auto* state = static_cast<TransferState*>( user_data );
            const size_t length = size * count;

            if ( state->discarding )
            {
                return length;
            }

            if ( state->sink && !( *state->sink )( data, length ) )
            {
                state->sink_failed = true;

                return 0;   // Any short count aborts the transfer.
            }

            state->bytes_received += length;

            if ( state->progress )
            {
                const uint64_t total = state->content_length == 0
                    ? 0
                    : state->resume_offset + state->content_length;

                if ( !( *state->progress )( state->resume_offset + state->bytes_received, total ) )
                {
                    state->aborted = true;

                    return 0;
                }
            }

            return length;
        }

        size_t readHeader( char* buffer, size_t size, size_t count, void* user_data )
        {
            auto* state = static_cast<TransferState*>( user_data );
            const size_t length = size * count;

            std::string_view line( buffer, length );

            while ( !line.empty() && ( line.back() == '\r' || line.back() == '\n' ) )
            {
                line.remove_suffix( 1 );
            }

            // The status line has no colon, and it is the only place the response code is
            // visible early enough to suppress a redirect's body before the sink sees it.
            if ( line.starts_with( "HTTP/" ) )
            {
                const auto space = line.find( ' ' );

                if ( space != std::string_view::npos )
                {
                    const long code = std::strtol( std::string( line.substr( space + 1 ) ).c_str(),
                        nullptr, 10 );

                    state->discarding = ( code >= 300 && code < 400 );
                }

                return length;
            }

            const auto colon = line.find( ':' );

            if ( colon == std::string_view::npos )
            {
                return length;
            }

            std::string_view name = line.substr( 0, colon );
            std::string_view value = line.substr( colon + 1 );

            while ( !value.empty() && value.front() == ' ' )
            {
                value.remove_prefix( 1 );
            }

            // Header names are case-insensitive and servers vary.
            const auto equalsIgnoringCase = []( std::string_view a, std::string_view b )
                {
                    if ( a.size() != b.size() )
                    {
                        return false;
                    }

                    for ( size_t index = 0; index < a.size(); ++index )
                    {
                        const char left = static_cast<char>( std::tolower( a[ index ] ) );
                        const char right = static_cast<char>( std::tolower( b[ index ] ) );

                        if ( left != right )
                        {
                            return false;
                        }
                    }

                    return true;
                };

            if ( equalsIgnoringCase( name, "location" ) )
            {
                state->location.assign( value );
            }
            else if ( equalsIgnoringCase( name, "content-length" ) )
            {
                state->content_length = std::strtoull( std::string( value ).c_str(), nullptr, 10 );
            }

            return length;
        }
    }

    /**
     * @brief GET a URL, streaming the body to a sink.
     *
     * Redirects are followed manually so the authorization header can be dropped when the host
     * changes: HuggingFace redirects LFS files to a pre-signed CDN URL, and libcurl forwards a
     * header set through CURLOPT_HTTPHEADER across a cross-host redirect. The pre-signed URL
     * carries its own authorization and needs no token.
     *
     * A resume request that draws 200 rather than 206 is reported as RangeIgnored rather than
     * treated as success: the server is sending the whole file, so appending it to an existing
     * partial would silently concatenate.
     *
     * @param request  What to fetch, and from which offset.
     * @param sink     Receives body bytes. Called on the calling thread.
     * @param progress Optional; return false to abort.
     */
    export HttpResult httpGet(
        const HttpRequest& request,
        SinkCallback sink,
        ProgressCallback progress = {} )
    {
        ensureCurlInitialized();

        HttpResult result;

        std::string url = request.url;
        std::string origin_host = hostOf( url );

        for ( int hop = 0; hop <= request.maximum_redirects; ++hop )
        {
            std::unique_ptr<CURL, void( * )( CURL* )> handle(
                curl_easy_init(), []( CURL* easy ) { curl_easy_cleanup( easy ); } );

            if ( handle == nullptr )
            {
                result.status = HttpStatus::TransportError;
                result.message = "curl_easy_init failed";

                return result;
            }

            TransferState state;
            state.sink = &sink;
            state.progress = progress ? &progress : nullptr;
            state.resume_offset = request.resume_from;

            curl_easy_setopt( handle.get(), CURLOPT_URL, url.c_str() );
            curl_easy_setopt( handle.get(), CURLOPT_WRITEFUNCTION, writeBody );
            curl_easy_setopt( handle.get(), CURLOPT_WRITEDATA, &state );
            curl_easy_setopt( handle.get(), CURLOPT_HEADERFUNCTION, readHeader );
            curl_easy_setopt( handle.get(), CURLOPT_HEADERDATA, &state );

            // Deliberately absent: CURLOPT_FOLLOWLOCATION. See the note above.
            curl_easy_setopt( handle.get(), CURLOPT_FOLLOWLOCATION, 0L );
            curl_easy_setopt( handle.get(), CURLOPT_NOSIGNAL, 1L );
            curl_easy_setopt( handle.get(), CURLOPT_USERAGENT, "Mila/0.20" );

            if ( request.low_speed_timeout_seconds > 0 )
            {
                curl_easy_setopt( handle.get(), CURLOPT_LOW_SPEED_LIMIT, 1L );
                curl_easy_setopt( handle.get(), CURLOPT_LOW_SPEED_TIME,
                    request.low_speed_timeout_seconds );
            }

            std::string range_header;
            curl_slist* headers = nullptr;

            // Auth travels only to the host the request started on.
            const bool same_host = ( hostOf( url ) == origin_host );

            if ( !request.token.empty() && same_host )
            {
                headers = curl_slist_append(
                    headers, ( "Authorization: Bearer " + request.token ).c_str() );
            }

            if ( request.resume_from > 0 )
            {
                range_header = std::format( "Range: bytes={}-", request.resume_from );
                headers = curl_slist_append( headers, range_header.c_str() );
            }

            if ( headers != nullptr )
            {
                curl_easy_setopt( handle.get(), CURLOPT_HTTPHEADER, headers );
            }

            const CURLcode code = curl_easy_perform( handle.get() );

            long http_code = 0;
            curl_easy_getinfo( handle.get(), CURLINFO_RESPONSE_CODE, &http_code );

            if ( headers != nullptr )
            {
                curl_slist_free_all( headers );
            }

            result.http_code = http_code;
            result.bytes_received = state.bytes_received;
            result.content_length = state.content_length;
            result.final_url = url;

            if ( state.aborted )
            {
                result.status = HttpStatus::TransportError;
                result.message = "transfer aborted by progress callback";

                return result;
            }

            if ( state.sink_failed )
            {
                result.status = HttpStatus::TransportError;
                result.message = "sink rejected data (write failed)";

                return result;
            }

            if ( code != CURLE_OK )
            {
                result.status = HttpStatus::TransportError;
                result.message = curl_easy_strerror( code );

                return result;
            }

            if ( http_code == 301 || http_code == 302 || http_code == 303
                || http_code == 307 || http_code == 308 )
            {
                if ( state.location.empty() )
                {
                    result.status = HttpStatus::TransportError;
                    result.message = std::format( "{} redirect without a Location header", http_code );

                    return result;
                }

                url = resolveRedirect( url, state.location );

                continue;
            }

            switch ( http_code )
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
                    result.status = ( http_code >= 500 )
                        ? HttpStatus::ServerError
                        : HttpStatus::TransportError;
                    result.message = std::format( "unexpected HTTP status {}", http_code );

                    return result;
            }
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
    export HttpResult httpGetString( const HttpRequest& request, std::string& body )
    {
        body.clear();

        return httpGet( request, [&body]( const char* data, size_t length )
            {
                body.append( data, length );

                return true;
            } );
    }
}
