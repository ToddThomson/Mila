/**
 * @file LineChannel.Windows.cpp
 * @brief The line channel over Windows file handles: anonymous pipes to a worker, a named pipe to a client.
 */

module;
#define NOMINMAX
#include <windows.h>

module ToolBroker.LineChannel;

import std;

namespace Mila::ToolBroker
{
    namespace
    {
        // Larger writes are split, since WriteFile takes a 32-bit count.
        constexpr std::size_t kWriteChunk = 1u << 20;
        constexpr std::size_t kReadChunk = 64u * 1024u;

        void closeHandle( void*& handle ) noexcept
        {
            if ( handle != nullptr && handle != INVALID_HANDLE_VALUE )
            {
                ::CloseHandle( handle );
            }

            handle = nullptr;
        }

        [[noreturn]] void throwLastError( const char* what )
        {
            throw std::system_error( static_cast<int>( ::GetLastError() ), std::system_category(), what );
        }
    }

    LineChannel::LineChannel( void* read_handle, void* write_handle, std::size_t message_limit )
        : read_handle_( read_handle ), write_handle_( write_handle ), message_limit_( message_limit )
    {
    }

    LineChannel::~LineChannel()
    {
        close();
    }

    LineChannel::LineChannel( LineChannel&& other ) noexcept
        : read_handle_( std::exchange( other.read_handle_, nullptr ) ),
          write_handle_( std::exchange( other.write_handle_, nullptr ) ),
          message_limit_( other.message_limit_ ),
          buffer_( std::move( other.buffer_ ) ),
          end_of_stream_( other.end_of_stream_ )
    {
    }

    LineChannel& LineChannel::operator=( LineChannel&& other ) noexcept
    {
        if ( this != &other )
        {
            close();
            read_handle_ = std::exchange( other.read_handle_, nullptr );
            write_handle_ = std::exchange( other.write_handle_, nullptr );
            message_limit_ = other.message_limit_;
            buffer_ = std::move( other.buffer_ );
            end_of_stream_ = other.end_of_stream_;
        }

        return *this;
    }

    void LineChannel::send( std::string_view message )
    {
        if ( message.find( '\n' ) != std::string_view::npos )
        {
            throw std::invalid_argument( "An MCP message cannot contain a newline" );
        }

        if ( write_handle_ == nullptr )
        {
            throw std::logic_error( "The sending side of this channel is closed" );
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

            if ( !::WriteFile( write_handle_, line.data() + written, count, &done, nullptr ) )
            {
                throwLastError( "Writing a message failed" );
            }

            written += done;
        }
    }

    std::optional<std::string> LineChannel::receive()
    {
        std::array<char, kReadChunk> chunk{};

        while ( true )
        {
            const auto newline = buffer_.find( '\n' );

            if ( newline != std::string::npos )
            {
                if ( newline > message_limit_ )
                {
                    throw std::length_error( "A message exceeded the channel's limit" );
                }

                std::string message = buffer_.substr( 0, newline );
                buffer_.erase( 0, newline + 1 );

                // A worker writing through the C runtime's text mode ends its lines with a carriage return.
                if ( !message.empty() && message.back() == '\r' )
                {
                    message.pop_back();
                }

                return message;
            }

            if ( buffer_.size() > message_limit_ )
            {
                throw std::length_error( "A message exceeded the channel's limit" );
            }

            if ( end_of_stream_ || read_handle_ == nullptr )
            {
                return std::nullopt;
            }

            DWORD count = 0;

            if ( !::ReadFile( read_handle_, chunk.data(), static_cast<DWORD>( chunk.size() ), &count, nullptr ) )
            {
                const auto error = ::GetLastError();

                if ( error != ERROR_BROKEN_PIPE && error != ERROR_PIPE_NOT_CONNECTED )
                {
                    throw std::system_error( static_cast<int>( error ), std::system_category(), "Reading a message failed" );
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

    void LineChannel::closeSending() noexcept
    {
        closeHandle( write_handle_ );
    }

    void LineChannel::close() noexcept
    {
        closeHandle( read_handle_ );
        closeHandle( write_handle_ );
    }
}
