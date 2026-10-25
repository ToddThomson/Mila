/**
 * @file BrokerConnection.Windows.cpp
 * @brief The client's end of the broker's named pipe on Windows.
 */

module;
#define NOMINMAX
#include <windows.h>

module AI.Tools.BrokerConnection;

import std;

namespace Mila::AI
{
    namespace
    {
        constexpr std::size_t kWriteChunk = 1u << 20;
        constexpr std::size_t kReadChunk = 64u * 1024u;
        constexpr DWORD kBusyWaitMilliseconds = 2000;

        [[noreturn]] void throwLastError( const char* what )
        {
            throw std::system_error( static_cast<int>( ::GetLastError() ), std::system_category(), what );
        }
    }

    BrokerConnection::BrokerConnection( const std::wstring& pipe_name )
    {
        // Identification level only, so the broker can read who is calling but cannot act as them.
        constexpr DWORD kFlags = SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION;

        HANDLE pipe = ::CreateFileW( pipe_name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, kFlags, nullptr );

        if ( pipe == INVALID_HANDLE_VALUE && ::GetLastError() == ERROR_PIPE_BUSY
            && ::WaitNamedPipeW( pipe_name.c_str(), kBusyWaitMilliseconds ) )
        {
            pipe = ::CreateFileW( pipe_name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, kFlags, nullptr );
        }

        if ( pipe == INVALID_HANDLE_VALUE )
        {
            throwLastError( "Opening the tool broker's pipe failed" );
        }

        pipe_ = pipe;
    }

    BrokerConnection::~BrokerConnection()
    {
        if ( pipe_ != nullptr )
        {
            ::CloseHandle( pipe_ );
        }
    }

    BrokerConnection::BrokerConnection( BrokerConnection&& other ) noexcept
        : pipe_( std::exchange( other.pipe_, nullptr ) ), buffer_( std::move( other.buffer_ ) ),
          end_of_stream_( other.end_of_stream_ )
    {
    }

    BrokerConnection& BrokerConnection::operator=( BrokerConnection&& other ) noexcept
    {
        if ( this != &other )
        {
            if ( pipe_ != nullptr )
            {
                ::CloseHandle( pipe_ );
            }

            pipe_ = std::exchange( other.pipe_, nullptr );
            buffer_ = std::move( other.buffer_ );
            end_of_stream_ = other.end_of_stream_;
        }

        return *this;
    }

    void BrokerConnection::send( std::string_view message )
    {
        if ( message.find( '\n' ) != std::string_view::npos )
        {
            throw std::invalid_argument( "An MCP message cannot contain a newline" );
        }

        std::string line;
        line.reserve( message.size() + 1 );
        line.append( message );
        line.push_back( '\n' );

        std::size_t written = 0;

        while ( written < line.size() )
        {
            const auto count = static_cast<DWORD>( std::min( line.size() - written, kWriteChunk ) );
            DWORD done = 0;

            if ( !::WriteFile( pipe_, line.data() + written, count, &done, nullptr ) )
            {
                throwLastError( "Writing to the tool broker failed" );
            }

            written += done;
        }
    }

    std::optional<std::string> BrokerConnection::receive()
    {
        std::array<char, kReadChunk> chunk{};

        while ( true )
        {
            const auto newline = buffer_.find( '\n' );

            if ( newline != std::string::npos )
            {
                if ( newline > kMessageLimit )
                {
                    throw std::length_error( "A message from the tool broker exceeded the limit" );
                }

                std::string message = buffer_.substr( 0, newline );
                buffer_.erase( 0, newline + 1 );

                if ( !message.empty() && message.back() == '\r' )
                {
                    message.pop_back();
                }

                return message;
            }

            if ( buffer_.size() > kMessageLimit )
            {
                throw std::length_error( "A message from the tool broker exceeded the limit" );
            }

            if ( end_of_stream_ )
            {
                return std::nullopt;
            }

            DWORD count = 0;

            if ( !::ReadFile( pipe_, chunk.data(), static_cast<DWORD>( chunk.size() ), &count, nullptr ) )
            {
                const auto error = ::GetLastError();

                if ( error != ERROR_BROKEN_PIPE && error != ERROR_PIPE_NOT_CONNECTED )
                {
                    throw std::system_error( static_cast<int>( error ), std::system_category(), "Reading from the tool broker failed" );
                }

                end_of_stream_ = true;
                continue;
            }

            if ( count == 0 )
            {
                end_of_stream_ = true;
                continue;
            }

            buffer_.append( chunk.data(), count );
        }
    }
}
