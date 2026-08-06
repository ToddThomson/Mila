/**
 * @file CurlHttpTransport.ixx
 * @brief libcurl as an IHttpTransport: one request, exactly the headers given.
 *
 * Everything that used to make this file interesting -- following redirects, dropping the
 * authorization header across a host change, building a Range, mapping a status -- moved up
 * into HttpClient, where it is written once for every transport. What is left is the part
 * that genuinely needs libcurl. The only module in the library that does, and therefore the
 * only one MILA_ENABLE_LIBCURL gates. See Specifications/HttpClient.md.
 */

module;
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>

#include <curl/curl.h>

export module Distribution.CurlHttpTransport;

import Distribution.HttpTransport;

namespace Mila::Distribution
{
    namespace
    {
        /**
         * @brief One-time libcurl global initialization.
         *
         * curl_global_init is not thread-safe and must run once before any handle exists.
         */
        bool ensureCurlInitialized()
        {
            static const bool initialized = []
                {
                    return curl_global_init( CURL_GLOBAL_DEFAULT ) == CURLE_OK;
                }();

            return initialized;
        }

        /// A non-2xx body is captured to this many bytes and no further: enough to carry an
        /// error page's useful line, bounded enough that a hostile server cannot grow it.
        constexpr size_t kMaximumCapturedErrorBytes = 4096;

        struct TransferState
        {
            const SinkCallback* sink{ nullptr };
            const HeadersCallback* on_headers{ nullptr };

            long http_code{ 0 };
            uint64_t content_length{ 0 };
            std::string location;
            std::string captured;

            bool sink_failed{ false };
            bool headers_reported{ false };

            /// Set once the status line is seen and the response is not a 2xx. Its body is
            /// diagnostic text, and passing it to the sink would prepend it to real content.
            bool capturing{ false };
        };

        size_t writeBody( char* data, size_t size, size_t count, void* user_data )
        {
            auto* state = static_cast<TransferState*>( user_data );
            const size_t length = size * count;

            if ( state->capturing )
            {
                const size_t room = ( state->captured.size() >= kMaximumCapturedErrorBytes )
                    ? 0
                    : kMaximumCapturedErrorBytes - state->captured.size();

                state->captured.append( data, ( length < room ) ? length : room );

                return length;
            }

            if ( state->sink && !( *state->sink )( data, length ) )
            {
                state->sink_failed = true;

                return 0;   // Any short count aborts the transfer.
            }

            return length;
        }

        bool equalsIgnoringCase( std::string_view left, std::string_view right )
        {
            if ( left.size() != right.size() )
            {
                return false;
            }

            for ( size_t index = 0; index < left.size(); ++index )
            {
                if ( std::tolower( left[ index ] ) != std::tolower( right[ index ] ) )
                {
                    return false;
                }
            }

            return true;
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
            // visible early enough to suppress a non-2xx body before the sink sees it.
            if ( line.starts_with( "HTTP/" ) )
            {
                const auto space = line.find( ' ' );

                if ( space != std::string_view::npos )
                {
                    state->http_code = std::strtol(
                        std::string( line.substr( space + 1 ) ).c_str(), nullptr, 10 );

                    state->capturing = !( state->http_code >= 200 && state->http_code < 300 );
                }

                // A redirect chain within one handle would re-enter here; the last status
                // line wins, which is also what CURLINFO_RESPONSE_CODE reports.
                state->headers_reported = false;
                state->content_length = 0;
                state->location.clear();

                return length;
            }

            const auto colon = line.find( ':' );

            if ( colon == std::string_view::npos )
            {
                // The blank line ending the header block: everything is known now.
                if ( line.empty() && !state->headers_reported )
                {
                    state->headers_reported = true;

                    if ( state->on_headers )
                    {
                        ( *state->on_headers )( state->http_code, state->content_length );
                    }
                }

                return length;
            }

            std::string_view name = line.substr( 0, colon );
            std::string_view value = line.substr( colon + 1 );

            while ( !value.empty() && value.front() == ' ' )
            {
                value.remove_prefix( 1 );
            }

            // Header names are case-insensitive and servers vary.
            if ( equalsIgnoringCase( name, "location" ) )
            {
                state->location.assign( value );
            }
            else if ( equalsIgnoringCase( name, "content-length" ) )
            {
                state->content_length = std::strtoull(
                    std::string( value ).c_str(), nullptr, 10 );
            }

            return length;
        }
    }

    /**
     * @brief libcurl as an IHttpTransport.
     *
     * CURLOPT_FOLLOWLOCATION stays off. That was once about protecting the token; now it is
     * simply the contract -- the client follows hops, because only the client knows which
     * host the request started on.
     */
    export class CurlHttpTransport : public IHttpTransport
    {
    public:

        std::string name() const override { return "libcurl"; }

        HttpResponse fetch(
            const HttpFetch& request,
            const SinkCallback& sink,
            const HeadersCallback& on_headers ) const override
        {
            HttpResponse response;

            if ( !ensureCurlInitialized() )
            {
                response.transport_failed = true;
                response.message = "libcurl global initialization failed";

                return response;
            }

            std::unique_ptr<CURL, void( * )( CURL* )> handle(
                curl_easy_init(), []( CURL* easy ) { curl_easy_cleanup( easy ); } );

            if ( handle == nullptr )
            {
                response.transport_failed = true;
                response.message = "curl_easy_init failed";

                return response;
            }

            TransferState state;
            state.sink = &sink;
            state.on_headers = on_headers ? &on_headers : nullptr;

            curl_easy_setopt( handle.get(), CURLOPT_URL, request.url.c_str() );
            curl_easy_setopt( handle.get(), CURLOPT_WRITEFUNCTION, writeBody );
            curl_easy_setopt( handle.get(), CURLOPT_WRITEDATA, &state );
            curl_easy_setopt( handle.get(), CURLOPT_HEADERFUNCTION, readHeader );
            curl_easy_setopt( handle.get(), CURLOPT_HEADERDATA, &state );

            curl_easy_setopt( handle.get(), CURLOPT_FOLLOWLOCATION, 0L );
            curl_easy_setopt( handle.get(), CURLOPT_NOSIGNAL, 1L );
            curl_easy_setopt( handle.get(), CURLOPT_USERAGENT, "Mila/0.20" );

            if ( request.low_speed_timeout_seconds > 0 )
            {
                curl_easy_setopt( handle.get(), CURLOPT_LOW_SPEED_LIMIT, 1L );
                curl_easy_setopt( handle.get(), CURLOPT_LOW_SPEED_TIME,
                    request.low_speed_timeout_seconds );
            }

            curl_slist* headers = nullptr;

            for ( const auto& header : request.headers )
            {
                headers = curl_slist_append(
                    headers, ( header.name + ": " + header.value ).c_str() );
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

            response.http_code = http_code;
            response.location = state.location;
            response.content_length = state.content_length;
            response.message = state.captured;

            if ( state.sink_failed )
            {
                // The client owns this diagnosis: it knows what its own sink was doing.
                response.transport_failed = false;

                return response;
            }

            if ( code != CURLE_OK )
            {
                response.transport_failed = true;
                response.message = curl_easy_strerror( code );
            }

            return response;
        }
    };
}
