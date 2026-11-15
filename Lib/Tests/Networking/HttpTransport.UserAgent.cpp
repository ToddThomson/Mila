/**
 * @file HttpTransport.UserAgent.cpp
 * @brief The User-Agent each of this platform's transports sends, read from the request a loopback server received
 *        (Network.md 4.3): curl's on Windows and Linux, URLSession's and Network.framework's on a Mac.
 */

#include "Common/LoopbackServer.h"

#include <gtest/gtest.h>
#include <cctype>
#include <format>
#include <memory>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Networking
{
    using namespace Mila::Networking;

    namespace
    {
        /// Every transport this build makes: the plain one, and the one held to a check on each connection.
        std::vector<std::shared_ptr<const IHttpTransport>> transports()
        {
            return { makeDefaultHttpTransport(), makeCheckedHttpTransport( isLoopback ) };
        }

        /// The values of every User-Agent header in a request's head, matched without regard to case.
        std::vector<std::string> userAgentsIn( const std::string& request )
        {
            std::string lowered = request;

            for ( char& character : lowered )
            {
                character = static_cast<char>( std::tolower( static_cast<unsigned char>( character ) ) );
            }

            const std::string name = "\r\nuser-agent:";
            std::vector<std::string> values;

            for ( std::size_t at = lowered.find( name ); at != std::string::npos; at = lowered.find( name, at + 1 ) )
            {
                std::size_t start = at + name.size();

                while ( start < request.size() && request[ start ] == ' ' )
                {
                    ++start;
                }

                values.push_back( request.substr( start, request.find( "\r\n", start ) - start ) );
            }

            return values;
        }

        std::string requestReceived( const IHttpTransport& transport, std::vector<HttpHeader> headers = {} )
        {
            LoopbackServer server;
            HttpFetch request;
            request.url = std::format( "http://127.0.0.1:{}/", server.port() );
            request.headers = std::move( headers );
            request.low_speed_timeout_seconds = 10;

            const HttpResponse response = transport.fetch( request, []( const char*, std::size_t ) { return true; }, {} );

            EXPECT_EQ( response.http_code, 200 ) << transport.name() << ": " << response.message;

            return server.lastRequest();
        }
    }

    TEST( HttpTransportUserAgent, IsMilaAndTheLibrarysFullVersion )
    {
        EXPECT_EQ( userAgent(), "Mila/" + getAPIVersion().toString() );
    }

    TEST( HttpTransportUserAgent, EveryTransportSendsItWhenTheRequestNamesNone )
    {
        for ( const auto& transport : transports() )
        {
            const std::vector<std::string> agents = userAgentsIn( requestReceived( *transport ) );

            ASSERT_EQ( agents.size(), 1u ) << transport->name();
            EXPECT_EQ( agents.front(), userAgent() ) << transport->name();
        }
    }

    TEST( HttpTransportUserAgent, ARequestsOwnUserAgentIsSentInItsPlace )
    {
        for ( const auto& transport : transports() )
        {
            const std::vector<std::string> agents = userAgentsIn( requestReceived( *transport, { { "User-Agent", "Caller/1.0" } } ) );

            ASSERT_EQ( agents.size(), 1u ) << transport->name();
            EXPECT_EQ( agents.front(), "Caller/1.0" ) << transport->name();
        }
    }
}
