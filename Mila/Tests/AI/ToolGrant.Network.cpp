/**
 * @file ToolGrant.Network.cpp
 * @brief What a grant's network reaches: each connection the grant refuses beside the same connection made without
 *        it, reaching what was there (ToolCalling.md 3.4). Windows only, for the local server's sockets.
 *
 * A server on the loopback address stands in for anything on the machine or its network. One test reads a public
 * page and skips when the machine is offline.
 */

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>

#include <gtest/gtest.h>
#include <atomic>
#include <cstdlib>
#include <format>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>

import Mila;

namespace Mila::Tests::AI
{
    using namespace Mila::AI;
    using namespace Mila::Distribution;

    namespace
    {
        constexpr const char* kRefusal = "not a public one";

        constexpr const char* kPageReadingOk = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";

        /// 127.0.0.2: loopback too, and a second address for a redirect to lead to.
        constexpr std::uint32_t kSecondLoopback = 0x7f000002u;

        /// Answers every connection on a loopback address with one response, a page reading "ok" unless given
        /// another, and counts them.
        class LoopbackServer
        {
        public:

            explicit LoopbackServer( std::string response = kPageReadingOk, std::uint32_t host = INADDR_LOOPBACK )
                : response_( std::move( response ) )
            {
                WSADATA data{};
                WSAStartup( MAKEWORD( 2, 2 ), &data );

                listener_ = socket( AF_INET, SOCK_STREAM, IPPROTO_TCP );

                sockaddr_in address{};
                address.sin_family = AF_INET;
                address.sin_addr.s_addr = htonl( host );
                address.sin_port = 0;

                bind( listener_, reinterpret_cast<const sockaddr*>( &address ), sizeof( address ) );
                listen( listener_, 8 );

                int length = sizeof( address );
                getsockname( listener_, reinterpret_cast<sockaddr*>( &address ), &length );
                port_ = ntohs( address.sin_port );

                thread_ = std::thread( [this] { serve(); } );
            }

            ~LoopbackServer()
            {
                closesocket( listener_ );
                thread_.join();
                WSACleanup();
            }

            [[nodiscard]] int port() const noexcept
            {
                return port_;
            }

            [[nodiscard]] int connections() const noexcept
            {
                return connections_.load();
            }

        private:

            void serve()
            {
                for ( ;; )
                {
                    const SOCKET client = accept( listener_, nullptr, nullptr );

                    if ( client == INVALID_SOCKET )
                    {
                        return;
                    }

                    ++connections_;

                    char request[ 4096 ];
                    recv( client, request, sizeof( request ), 0 );

                    send( client, response_.data(), static_cast<int>( response_.size() ), 0 );
                    closesocket( client );
                }
            }

            std::string response_;
            SOCKET listener_{ INVALID_SOCKET };
            int port_{ 0 };
            std::atomic<int> connections_{ 0 };
            std::thread thread_;
        };

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
                _putenv_s( "http_proxy", std::format( "http://127.0.0.1:{}", port ).c_str() );
            }

            ~ProxyInEnvironment()
            {
                _putenv_s( "http_proxy", "" );
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

        EXPECT_EQ( fetched.response.http_code, 200 );
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
        /// A server on 127.0.0.1 redirecting to one on 127.0.0.2, followed as a client follows a redirect.
        struct RedirectFollowed
        {
            int redirecting_connections;
            int target_connections;
            HttpResult result;
            std::string body;
        };

        RedirectFollowed followRedirect( ConnectionCheck check )
        {
            LoopbackServer target( kPageReadingOk, kSecondLoopback );
            LoopbackServer redirecting( std::format(
                "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.2:{}/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", target.port() ) );

            const HttpClient client( makeCheckedHttpTransport( std::move( check ) ) );
            HttpRequest request;
            request.url = std::format( "http://127.0.0.1:{}/", redirecting.port() );

            RedirectFollowed followed{};
            followed.result = client.getString( request, followed.body );
            followed.redirecting_connections = redirecting.connections();
            followed.target_connections = target.connections();

            return followed;
        }

        bool isFirstLoopback( std::span<const std::uint8_t> address )
        {
            return address.size() == 4 && address[ 0 ] == 127 && address[ 1 ] == 0 && address[ 2 ] == 0 && address[ 3 ] == 1;
        }
    }

    TEST( ToolGrantNetwork, ARedirectToAnAddressRefusedIsNotReached )
    {
        // The check stands in for isPublicAddress: the first address is allowed and the one redirected to is not.
        const RedirectFollowed followed = followRedirect( isFirstLoopback );

        EXPECT_EQ( followed.redirecting_connections, 1 );
        EXPECT_EQ( followed.target_connections, 0 ) << "a redirect's connection is checked as the first was";
        EXPECT_EQ( followed.result.status, HttpStatus::TransportError );
    }

    TEST( ToolGrantNetwork, ARedirectToAnAddressAllowedIsReached )
    {
        const RedirectFollowed followed = followRedirect( []( std::span<const std::uint8_t> address )
            {
                return address.size() == 4 && address[ 0 ] == 127;
            } );

        EXPECT_EQ( followed.redirecting_connections, 1 );
        EXPECT_EQ( followed.target_connections, 1 );
        EXPECT_EQ( followed.body, "ok" );
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

    TEST( ToolGrantNetwork, APublicPageIsReached )
    {
        const Fetched fetched = fetch( *grantedTransport(), "https://example.com/" );

        if ( fetched.response.transport_failed && fetched.response.message.find( kRefusal ) == std::string::npos )
        {
            GTEST_SKIP() << "offline: " << fetched.response.message;
        }

        EXPECT_FALSE( fetched.response.transport_failed ) << fetched.response.message;
        EXPECT_EQ( fetched.response.http_code, 200 );
    }
}
