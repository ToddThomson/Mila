/**
 * @file NetworkHttpTransport.cpp
 * @brief The bodies of NetworkHttpTransport's members; the interface is NetworkHttpTransport.ixx.
 */

module;

#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include "NetworkConnection.h"

module Networking.NetworkHttpTransport;

import std;
import Networking.HttpTransport;
import Networking.HttpReplyReader;

namespace Mila::Networking
{
    namespace
    {
        constexpr const char* kRefusedAddress =
            "the address it leads to is not a public one, and only public addresses may be reached";

        constexpr std::size_t kReceiveBytes = 64 * 1024;

        /// Where a URL leads, read for one request.
        struct Target
        {
            bool secure{ false };
            bool literal_ipv6{ false };
            std::string host;
            std::string port;
            std::string path;
        };

        std::string lowered( std::string_view text )
        {
            std::string result( text );

            for ( char& character : result )
            {
                character = static_cast<char>( std::tolower( static_cast<unsigned char>( character ) ) );
            }

            return result;
        }

        std::expected<Target, std::string> readUrl( const std::string& url )
        {
            const std::size_t scheme_end = url.find( "://" );

            if ( scheme_end == std::string::npos )
                return std::unexpected( "'" + url + "' is not a URL" );

            Target target;
            const std::string scheme = lowered( std::string_view( url ).substr( 0, scheme_end ) );

            if ( scheme != "http" && scheme != "https" )
                return std::unexpected( "only http and https are spoken, and the URL names " + scheme );

            target.secure = scheme == "https";

            const std::string_view rest = std::string_view( url ).substr( scheme_end + 3 );
            const std::size_t authority_end = rest.find_first_of( "/?#" );
            const std::string_view authority = rest.substr( 0, authority_end );
            std::string_view remainder = authority_end == std::string_view::npos ? std::string_view{} : rest.substr( authority_end );

            if ( authority.find( '@' ) != std::string_view::npos )
                return std::unexpected( "a URL that carries a user name or password is not fetched" );

            std::string_view port;

            if ( authority.starts_with( '[' ) )
            {
                const std::size_t close = authority.find( ']' );

                if ( close == std::string_view::npos )
                    return std::unexpected( "the URL's address is not closed by ]" );

                target.host = authority.substr( 1, close - 1 );
                target.literal_ipv6 = true;
                const std::string_view after = authority.substr( close + 1 );

                if ( !after.empty() && !after.starts_with( ':' ) )
                    return std::unexpected( "the URL's address is followed by something other than a port" );

                port = after.empty() ? std::string_view{} : after.substr( 1 );
            }
            else
            {
                const std::size_t colon = authority.rfind( ':' );
                target.host = authority.substr( 0, colon );
                port = colon == std::string_view::npos ? std::string_view{} : authority.substr( colon + 1 );
            }

            if ( target.host.empty() )
                return std::unexpected( "the URL names no host" );

            if ( port.empty() )
            {
                target.port = target.secure ? "443" : "80";
            }
            else
            {
                int number = 0;
                const auto [ end, error ] = std::from_chars( port.data(), port.data() + port.size(), number );

                if ( error != std::errc() || end != port.data() + port.size() || number < 1 || number > 65535 )
                    return std::unexpected( "the URL's port is not one" );

                target.port = std::string( port );
            }

            remainder = remainder.substr( 0, remainder.find( '#' ) );
            target.path = remainder.empty() ? "/" : remainder.starts_with( '?' ) ? "/" + std::string( remainder ) : std::string( remainder );

            return target;
        }

        /// The Host header's value: the port only when it is not the scheme's own.
        std::string hostHeader( const Target& target )
        {
            std::string value = target.literal_ipv6 ? "[" + target.host + "]" : target.host;
            const bool default_port = target.port == ( target.secure ? "443" : "80" );

            return default_port ? value : value + ":" + target.port;
        }

        /// A line break in a header would let it write headers of its own, so such a request is never sent.
        bool safeHeader( const HttpHeader& header )
        {
            const auto breaks = []( std::string_view text )
                {
                    return text.find_first_of( "\r\n" ) != std::string_view::npos;
                };

            return !header.name.empty() && header.name.find( ':' ) == std::string::npos && !breaks( header.name ) && !breaks( header.value );
        }

        std::string requestText( const HttpFetch& request, const Target& target )
        {
            bool host = false;
            bool agent = false;
            bool accept = false;

            for ( const auto& header : request.headers )
            {
                const std::string name = lowered( header.name );
                host = host || name == "host";
                agent = agent || name == "user-agent";
                accept = accept || name == "accept";
            }

            std::string text = std::string( request.body ? "POST " : "GET " ) + target.path + " HTTP/1.1\r\n";

            if ( !host )
            {
                text += "Host: " + hostHeader( target ) + "\r\n";
            }

            for ( const auto& header : request.headers )
            {
                text += header.name + ": " + header.value + "\r\n";
            }

            // What curl's request carries beside the given headers, so a server sees the same request from either.
            if ( !agent )
            {
                text += "User-Agent: Mila/0.20\r\n";
            }

            if ( !accept )
            {
                text += "Accept: */*\r\n";
            }

            text += "Connection: close\r\n";

            if ( request.body )
            {
                text += "Content-Length: " + std::to_string( request.body->size() ) + "\r\n";
            }

            text += "\r\n";

            if ( request.body )
            {
                text += *request.body;
            }

            return text;
        }

        std::span<const std::uint8_t> addressBytes( const addrinfo& address )
        {
            if ( address.ai_family == AF_INET )
            {
                const auto* ipv4 = reinterpret_cast<const sockaddr_in*>( address.ai_addr );

                return { reinterpret_cast<const std::uint8_t*>( &ipv4->sin_addr ), 4 };
            }

            if ( address.ai_family == AF_INET6 )
            {
                const auto* ipv6 = reinterpret_cast<const sockaddr_in6*>( address.ai_addr );

                return { reinterpret_cast<const std::uint8_t*>( &ipv6->sin6_addr ), 16 };
            }

            return {};
        }

        struct CloseConnection
        {
            void operator()( MilaNetworkConnection* connection ) const
            {
                milaNetworkClose( connection );
            }
        };

        HttpResponse failed( std::string message )
        {
            HttpResponse response;
            response.transport_failed = true;
            response.message = std::move( message );

            return response;
        }
    }

    NetworkHttpTransport::NetworkHttpTransport( ConnectionCheck check )
        : check_( std::move( check ) )
    {
    }

    std::string NetworkHttpTransport::name() const
    {
        return "Network.framework";
    }

    HttpResponse NetworkHttpTransport::fetch(
        const HttpFetch& request,
        const SinkCallback& sink,
        const HeadersCallback& on_headers ) const
    {
        const auto target = readUrl( request.url );

        if ( !target )
            return failed( target.error() );

        if ( !std::ranges::all_of( request.headers, safeHeader ) )
            return failed( "a header carries a line break or no name, so the request is not sent" );

        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;
        addrinfo* found = nullptr;
        const int resolved = getaddrinfo( target->host.c_str(), target->port.c_str(), &hints, &found );

        if ( resolved != 0 )
            return failed( std::string( "the host's name did not resolve: " ) + gai_strerror( resolved ) );

        const std::unique_ptr<addrinfo, decltype( &freeaddrinfo )> addresses( found, freeaddrinfo );
        std::unique_ptr<MilaNetworkConnection, CloseConnection> connection;
        bool refused = false;
        std::string failure;

        // Every address the name leads to is checked before it is connected to, in the order the resolver gave.
        for ( const addrinfo* address = found; address != nullptr && !connection; address = address->ai_next )
        {
            const std::span<const std::uint8_t> bytes = addressBytes( *address );

            if ( bytes.empty() )
                continue;

            if ( !check_( bytes ) )
            {
                refused = true;

                continue;
            }

            MilaNetworkOutcome outcome{};
            connection.reset( milaNetworkConnect( address->ai_addr, target->secure ? target->host.c_str() : nullptr,
                request.low_speed_timeout_seconds, &outcome ) );

            if ( !connection )
            {
                failure = outcome.message;

                if ( outcome.result == MilaNetworkNotDirect )
                    break;
            }
        }

        if ( !connection )
            return failed( !failure.empty() ? failure : refused ? std::string( kRefusedAddress ) : "the host's name led to no address" );

        const std::string text = requestText( request, *target );
        MilaNetworkOutcome sent{};

        if ( milaNetworkSend( connection.get(), text.data(), text.size(), request.low_speed_timeout_seconds, &sent ) != MilaNetworkOk )
            return failed( sent.message );

        HttpReplyReader reader;
        std::string captured;
        bool capturing = false;
        bool reported = false;
        bool sink_failed = false;
        std::optional<std::string> receive_failure;

        const auto report = [&]
            {
                if ( reported || !reader.headComplete() )
                    return;

                reported = true;

                // A redirect or error body is diagnostic text, never content.
                capturing = !( reader.status() >= 200 && reader.status() < 300 );

                if ( on_headers )
                {
                    on_headers( reader.status(), reader.contentLength() );
                }
            };

        const auto body = [&]( const char* data, std::size_t length )
            {
                report();

                if ( capturing )
                {
                    const std::size_t room = captured.size() >= kMaximumCapturedErrorBytes ? 0 : kMaximumCapturedErrorBytes - captured.size();
                    captured.append( data, std::min( length, room ) );

                    return true;
                }

                if ( sink && !sink( data, length ) )
                {
                    sink_failed = true;

                    return false;
                }

                return true;
            };

        std::vector<char> buffer( kReceiveBytes );

        while ( !reader.complete() )
        {
            std::size_t received = 0;
            MilaNetworkOutcome outcome{};
            const MilaNetworkResult result = milaNetworkReceive( connection.get(), buffer.data(), buffer.size(), &received,
                request.low_speed_timeout_seconds, &outcome );

            if ( result == MilaNetworkOk )
            {
                if ( !reader.read( std::string_view( buffer.data(), received ), body ) )
                    break;

                report();

                continue;
            }

            if ( result == MilaNetworkClosed )
            {
                reader.finish();

                break;
            }

            receive_failure = outcome.message;

            break;
        }

        report();

        HttpResponse response;
        response.http_code = reader.status();
        response.location = reader.location();
        response.content_length = reader.contentLength();
        response.message = std::move( captured );

        // The sink stopped it: the client owns that diagnosis, since it knows what its own sink was doing.
        if ( sink_failed )
            return response;

        if ( receive_failure )
        {
            response.transport_failed = true;
            response.message = *receive_failure;
        }
        else if ( !reader.complete() )
        {
            response.transport_failed = true;
            response.message = "the reply could not be read: " + reader.error();
        }

        return response;
    }
}
