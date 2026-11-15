/**
 * @file ToolGrant.Network.cpp
 * @brief What a grant's network reaches: each connection the grant refuses beside the same connection made without
 *        it, reaching what was there (ToolCalling.md 3.4), on every platform's own transports (HttpClient.md 9).
 *
 * A server on a loopback address stands in for anything on the machine or its network. One test reads a public
 * page and skips when the machine is offline.
 */

#include "Common/LoopbackServer.h"

#include <gtest/gtest.h>
#include <cstdlib>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>

import Mila;

namespace Mila::Tests::AI
{
    using namespace Mila::Agents;
    using namespace Mila::Networking;

    namespace
    {
        constexpr const char* kRefusal = "not a public one";

        struct Fetched
        {
            HttpResponse response;
            std::string body;
        };

        Fetched fetch( const IHttpTransport& transport, const std::string& url )
        {
            Fetched fetched;
            HttpFetch request;
            request.url = url;
            request.low_speed_timeout_seconds = 10;

            fetched.response = transport.fetch( request, [&fetched]( const char* data, std::size_t length )
                {
                    fetched.body.append( data, length );

                    return true;
                }, {} );

            return fetched;
        }

        std::shared_ptr<const IHttpTransport> grantedTransport()
        {
            return ToolGrant( {}, true ).transport();
        }

        /// Names a proxy in the environment for one test, as a person's machine may.
        class ProxyInEnvironment
        {
        public:

            explicit ProxyInEnvironment( int port )
            {
                const std::string proxy = std::format( "http://127.0.0.1:{}", port );
#ifdef _WIN32
                _putenv_s( "http_proxy", proxy.c_str() );
#else
                setenv( "http_proxy", proxy.c_str(), 1 );
#endif
            }

            ~ProxyInEnvironment()
            {
#ifdef _WIN32
                _putenv_s( "http_proxy", "" );
#else
                unsetenv( "http_proxy" );
#endif
            }
        };
    }

    TEST( ToolGrantNetwork, ALoopbackServerIsRefusedThroughTheGrant )
    {
        LoopbackServer server;
        const Fetched fetched = fetch( *grantedTransport(), std::format( "http://127.0.0.1:{}/", server.port() ) );

        EXPECT_TRUE( fetched.response.transport_failed );
        EXPECT_NE( fetched.response.message.find( kRefusal ), std::string::npos ) << fetched.response.message;
        EXPECT_EQ( server.connections(), 0 );
    }

    TEST( ToolGrantNetwork, TheSameServerIsReachedWithoutTheGrant )
    {
        LoopbackServer server;
        const Fetched fetched = fetch( *makeDefaultHttpTransport(), std::format( "http://127.0.0.1:{}/", server.port() ) );

        EXPECT_EQ( fetched.response.http_code, 200 ) << fetched.response.message;
        EXPECT_EQ( fetched.body, "ok" );
        EXPECT_EQ( server.connections(), 1 );
    }

    TEST( ToolGrantNetwork, TheSameServerIsReachedThroughACheckThatAllowsIt )
    {
        LoopbackServer server;
        const Fetched fetched = fetch( *makeCheckedHttpTransport( isLoopback ), std::format( "http://127.0.0.1:{}/", server.port() ) );

        EXPECT_EQ( fetched.response.http_code, 200 ) << fetched.response.message;
        EXPECT_EQ( fetched.body, "ok" );
        EXPECT_EQ( server.connections(), 1 );
    }

    TEST( ToolGrantNetwork, ANameThatResolvesToLoopbackIsRefused )
    {
        LoopbackServer server;
        const Fetched fetched = fetch( *grantedTransport(), std::format( "http://localhost:{}/", server.port() ) );

        EXPECT_TRUE( fetched.response.transport_failed );
        EXPECT_EQ( server.connections(), 0 );
    }

    namespace
    {
        /// A server on 127.0.0.1 redirecting to one on ::1, followed as a client follows a redirect.
        struct RedirectFollowed
        {
            int redirecting_connections;
            int target_connections;
            HttpResult result;
            std::string body;
        };

        RedirectFollowed followRedirect( ConnectionCheck check )
        {
            LoopbackServer target( kPageReadingOk, true );
            LoopbackServer redirecting( std::format(
                "HTTP/1.1 302 Found\r\nLocation: http://[::1]:{}/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", target.port() ) );

            const HttpClient client( makeCheckedHttpTransport( std::move( check ) ) );
            HttpRequest request;
            request.url = std::format( "http://127.0.0.1:{}/", redirecting.port() );

            RedirectFollowed followed{};
            followed.result = client.fetchString( request, followed.body );
            followed.redirecting_connections = redirecting.connections();
            followed.target_connections = target.connections();

            return followed;
        }
    }

    TEST( ToolGrantNetwork, ARedirectToAnAddressRefusedIsNotReached )
    {
        // The check stands in for isPublicAddress: the first address is allowed and the one redirected to is not.
        const RedirectFollowed followed = followRedirect( isIpv4Loopback );

        EXPECT_EQ( followed.redirecting_connections, 1 );
        EXPECT_EQ( followed.target_connections, 0 ) << "a redirect's connection is checked as the first was";
        EXPECT_EQ( followed.result.status, HttpStatus::TransportError );
    }

    TEST( ToolGrantNetwork, ARedirectToAnAddressAllowedIsReached )
    {
        const RedirectFollowed followed = followRedirect( isLoopback );

        EXPECT_EQ( followed.redirecting_connections, 1 );
        EXPECT_EQ( followed.target_connections, 1 );
        EXPECT_EQ( followed.body, "ok" ) << followed.result.message;
    }

    TEST( ToolGrantNetwork, AProxyNamedInTheEnvironmentIsNotUsedByTheGrant )
    {
        LoopbackServer proxy;
        const ProxyInEnvironment environment( proxy.port() );

        // Reached directly when the machine is online; either way never through the proxy.
        (void)fetch( *grantedTransport(), "http://example.com/" );

        EXPECT_EQ( proxy.connections(), 0 );
    }

    TEST( ToolGrantNetwork, AProxyNamedInTheEnvironmentIsUsedWithoutTheGrant )
    {
        if ( makeDefaultHttpTransport()->name() != "libcurl" )
        {
            GTEST_SKIP() << "the environment's proxy is curl's to use; a Mac takes its own proxy settings";
        }

        LoopbackServer proxy;
        const ProxyInEnvironment environment( proxy.port() );
        const Fetched fetched = fetch( *makeDefaultHttpTransport(), "http://example.com/" );

        EXPECT_EQ( proxy.connections(), 1 );
        EXPECT_EQ( fetched.body, "ok" ) << "the proxy answered, so the grant's refusal of it is a real one";
    }

    TEST( ToolGrantNetwork, AGrantWithoutTheNetworkGivesNoTransport )
    {
        EXPECT_THROW( (void)ToolGrant( {}, false ).transport(), std::runtime_error );

        ToolNeeds files;
        files.reads_files = true;

        EXPECT_THROW( (void)ToolGrant( {}, true ).narrowedTo( files ).transport(), std::runtime_error )
            << "a tool that declares no network is given none";
    }

    TEST( ToolGrantNetwork, AGrantWithTheNetworkGivesATransport )
    {
        ToolNeeds network;
        network.uses_network = true;

        EXPECT_NE( ToolGrant( {}, true ).transport(), nullptr );
        EXPECT_TRUE( ToolGrant( {}, true ).narrowedTo( network ).network() );
    }

    namespace
    {
        bool offline( const Fetched& fetched )
        {
            return fetched.response.transport_failed && fetched.response.message.find( kRefusal ) == std::string::npos
                && fetched.response.http_code == 0 && fetched.response.message.find( "certificate" ) == std::string::npos
                && fetched.response.message.find( "TLS" ) == std::string::npos;
        }
    }

    TEST( ToolGrantNetwork, APublicPageIsReached )
    {
        const Fetched fetched = fetch( *grantedTransport(), "https://example.com/" );

        if ( offline( fetched ) )
        {
            GTEST_SKIP() << "offline: " << fetched.response.message;
        }

        EXPECT_FALSE( fetched.response.transport_failed ) << fetched.response.message;
        EXPECT_EQ( fetched.response.http_code, 200 );
        EXPECT_FALSE( fetched.body.empty() );
    }

    TEST( ToolGrantNetwork, ACertificateForAnotherNameIsRefusedThroughEitherTransport )
    {
        for ( const auto& transport : { grantedTransport(), makeDefaultHttpTransport() } )
        {
            const Fetched fetched = fetch( *transport, "https://wrong.host.badssl.com/" );

            if ( fetched.response.transport_failed && fetched.response.message.find( "resolve" ) != std::string::npos )
            {
                GTEST_SKIP() << "offline: " << fetched.response.message;
            }

            EXPECT_TRUE( fetched.response.transport_failed ) << transport->name() << " reached a host whose certificate names another";
            EXPECT_TRUE( fetched.body.empty() ) << transport->name();
        }
    }
}
