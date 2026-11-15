/**
 * @file LoopbackServer.h
 * @brief A server on a loopback address that answers every connection with one response and keeps the last request
 *        it read, for the tests of a transport and of a grant's network.
 */

#pragma once

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <utility>

namespace Mila::Tests
{
#ifdef _WIN32
    using Socket = SOCKET;
    inline constexpr Socket kNoSocket = INVALID_SOCKET;
    inline void closeSocket( Socket socket ) { closesocket( socket ); }
    using AddressLength = int;
#else
    using Socket = int;
    inline constexpr Socket kNoSocket = -1;
    inline void closeSocket( Socket socket ) { close( socket ); }
    using AddressLength = socklen_t;
#endif

    inline constexpr const char* kPageReadingOk = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";

    /// Answers every connection on a loopback address, 127.0.0.1 or ::1, with one response, a page reading "ok"
    /// unless given another, counts them, and keeps the last request read.
    class LoopbackServer
    {
    public:

        explicit LoopbackServer( std::string response = kPageReadingOk, bool ipv6 = false )
            : response_( std::move( response ) ), ipv6_( ipv6 )
        {
#ifdef _WIN32
            WSADATA data{};
            WSAStartup( MAKEWORD( 2, 2 ), &data );
#endif
            listener_ = socket( ipv6_ ? AF_INET6 : AF_INET, SOCK_STREAM, IPPROTO_TCP );

            sockaddr_storage address = loopback( 0 );
            AddressLength length = addressLength();
            bind( listener_, reinterpret_cast<const sockaddr*>( &address ), length );
            listen( listener_, 8 );

            getsockname( listener_, reinterpret_cast<sockaddr*>( &address ), &length );
            port_ = ipv6_
                ? ntohs( reinterpret_cast<const sockaddr_in6*>( &address )->sin6_port )
                : ntohs( reinterpret_cast<const sockaddr_in*>( &address )->sin_port );

            thread_ = std::thread( [this] { serve(); } );
        }

        ~LoopbackServer()
        {
            // Closing a socket does not wake an accept on every platform; a last connection of its own does.
            stopping_ = true;
            const Socket waking = socket( ipv6_ ? AF_INET6 : AF_INET, SOCK_STREAM, IPPROTO_TCP );
            const sockaddr_storage address = loopback( port_ );
            connect( waking, reinterpret_cast<const sockaddr*>( &address ), addressLength() );
            thread_.join();
            closeSocket( waking );
            closeSocket( listener_ );
#ifdef _WIN32
            WSACleanup();
#endif
        }

        LoopbackServer( const LoopbackServer& ) = delete;
        LoopbackServer& operator=( const LoopbackServer& ) = delete;

        [[nodiscard]] int port() const noexcept
        {
            return port_;
        }

        [[nodiscard]] int connections() const noexcept
        {
            return connections_.load();
        }

        /// The bytes of the last request read, up to its first 4096: the request line and headers of any request a
        /// test sends.
        [[nodiscard]] std::string lastRequest() const
        {
            const std::lock_guard lock( request_mutex_ );

            return last_request_;
        }

    private:

        sockaddr_storage loopback( int port ) const
        {
            sockaddr_storage storage{};

            if ( ipv6_ )
            {
                auto* address = reinterpret_cast<sockaddr_in6*>( &storage );
                address->sin6_family = AF_INET6;
                address->sin6_addr = in6addr_loopback;
                address->sin6_port = htons( static_cast<std::uint16_t>( port ) );
            }
            else
            {
                auto* address = reinterpret_cast<sockaddr_in*>( &storage );
                address->sin_family = AF_INET;
                address->sin_addr.s_addr = htonl( INADDR_LOOPBACK );
                address->sin_port = htons( static_cast<std::uint16_t>( port ) );
            }

            return storage;
        }

        AddressLength addressLength() const
        {
            return static_cast<AddressLength>( ipv6_ ? sizeof( sockaddr_in6 ) : sizeof( sockaddr_in ) );
        }

        void serve()
        {
            for ( ;; )
            {
                const Socket client = accept( listener_, nullptr, nullptr );

                if ( client == kNoSocket )
                {
                    return;
                }

                if ( stopping_ )
                {
                    closeSocket( client );

                    return;
                }

                ++connections_;

#ifdef SO_NOSIGPIPE
                // A client that has gone must not end the test process with SIGPIPE.
                const int on = 1;
                setsockopt( client, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof( on ) );
#endif
#ifdef MSG_NOSIGNAL
                constexpr int kSendFlags = MSG_NOSIGNAL;
#else
                constexpr int kSendFlags = 0;
#endif

                char request[ 4096 ];
                const auto received = recv( client, request, sizeof( request ), 0 );

                if ( received > 0 )
                {
                    const std::lock_guard lock( request_mutex_ );
                    last_request_.assign( request, static_cast<std::size_t>( received ) );
                }

                send( client, response_.data(), static_cast<int>( response_.size() ), kSendFlags );
                closeSocket( client );
            }
        }

        std::string response_;
        bool ipv6_{ false };
        Socket listener_{ kNoSocket };
        int port_{ 0 };
        std::atomic<int> connections_{ 0 };
        std::atomic<bool> stopping_{ false };
        mutable std::mutex request_mutex_;
        std::string last_request_;
        std::thread thread_;
    };

    inline bool isIpv4Loopback( std::span<const std::uint8_t> address )
    {
        return address.size() == 4 && address[ 0 ] == 127 && address[ 1 ] == 0 && address[ 2 ] == 0 && address[ 3 ] == 1;
    }

    inline bool isLoopback( std::span<const std::uint8_t> address )
    {
        if ( isIpv4Loopback( address ) )
        {
            return true;
        }

        static constexpr std::uint8_t kIpv6Loopback[ 16 ] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

        return address.size() == 16 && std::memcmp( address.data(), kIpv6Loopback, 16 ) == 0;
    }
}
