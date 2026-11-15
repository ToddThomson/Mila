/**
 * @file UrlSessionHttpTransport.cpp
 * @brief The bodies of UrlSessionHttpTransport's members; the interface is UrlSessionHttpTransport.ixx.
 */

module;

#include "UrlSessionFetch.h"

module Distribution.UrlSessionHttpTransport;

import std;
import Distribution.HttpTransport;

namespace Mila::Distribution
{
    namespace
    {
        /// What one transfer's callbacks read and write; they run one at a time on the session's queue.
        struct Transfer
        {
            const SinkCallback* sink{ nullptr };
            const HeadersCallback* on_headers{ nullptr };
            HttpResponse response;
            std::string captured;
            bool capturing{ false };
        };

        void onHead( void* context, long status, const char* location, std::uint64_t content_length )
        {
            auto* transfer = static_cast<Transfer*>( context );
            transfer->response.http_code = status;
            transfer->response.location = location;
            transfer->response.content_length = content_length;

            // A redirect or error body is diagnostic text, never content.
            transfer->capturing = !( status >= 200 && status < 300 );

            if ( transfer->on_headers != nullptr && *transfer->on_headers )
            {
                ( *transfer->on_headers )( status, content_length );
            }
        }

        int onData( void* context, const char* data, std::size_t length )
        {
            auto* transfer = static_cast<Transfer*>( context );

            if ( transfer->capturing )
            {
                const std::size_t room = transfer->captured.size() >= kMaximumCapturedErrorBytes
                    ? 0
                    : kMaximumCapturedErrorBytes - transfer->captured.size();
                transfer->captured.append( data, std::min( length, room ) );

                return 1;
            }

            return transfer->sink != nullptr && *transfer->sink && !( *transfer->sink )( data, length ) ? 0 : 1;
        }
    }

    std::string UrlSessionHttpTransport::name() const
    {
        return "URLSession";
    }

    HttpResponse UrlSessionHttpTransport::fetch(
        const HttpFetch& request,
        const SinkCallback& sink,
        const HeadersCallback& on_headers ) const
    {
        std::vector<MilaUrlSessionHeader> headers;
        headers.reserve( request.headers.size() );

        for ( const auto& header : request.headers )
        {
            headers.push_back( { header.name.c_str(), header.value.c_str() } );
        }

        MilaUrlSessionRequest loading{};
        loading.url = request.url.c_str();
        loading.headers = headers.data();
        loading.header_count = headers.size();
        loading.has_body = request.body.has_value() ? 1 : 0;
        loading.body = request.body ? request.body->data() : nullptr;
        loading.body_length = request.body ? request.body->size() : 0;
        loading.idle_timeout_seconds = request.low_speed_timeout_seconds;

        Transfer transfer;
        transfer.sink = &sink;
        transfer.on_headers = &on_headers;

        MilaUrlSessionOutcome outcome{};
        milaUrlSessionFetch( &loading, &transfer, onHead, onData, &outcome );

        HttpResponse response = std::move( transfer.response );
        response.message = std::move( transfer.captured );

        // The sink stopped it: the client owns that diagnosis, since it knows what its own sink was doing.
        if ( outcome.stopped != 0 )
        {
            return response;
        }

        if ( outcome.transport_failed != 0 )
        {
            response.transport_failed = true;
            response.message = outcome.message;
        }

        return response;
    }
}
