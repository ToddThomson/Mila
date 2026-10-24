/**
 * @file PipeServer.Windows.cpp
 * @brief The broker's named pipe on Windows, its access list built from the current user's token.
 */

module;
#define NOMINMAX
#include <windows.h>
#include <sddl.h>

module ToolBroker.PipeServer;

import std;
import ToolBroker.LineChannel;

namespace Mila::ToolBroker
{
    namespace
    {
        constexpr DWORD kPipeBuffer = 64u * 1024u;

        [[noreturn]] void throwLastError( const char* what )
        {
            throw std::system_error( static_cast<int>( ::GetLastError() ), std::system_category(), what );
        }

        /** The current user's SID as a string, read from this process's token. */
        std::wstring currentUserSid()
        {
            HANDLE token = nullptr;

            if ( !::OpenProcessToken( ::GetCurrentProcess(), TOKEN_QUERY, &token ) )
            {
                throwLastError( "Reading the broker's token failed" );
            }

            DWORD size = 0;
            ::GetTokenInformation( token, TokenUser, nullptr, 0, &size );
            std::vector<std::byte> storage( size );

            if ( !::GetTokenInformation( token, TokenUser, storage.data(), size, &size ) )
            {
                ::CloseHandle( token );
                throwLastError( "Reading the broker's user failed" );
            }

            ::CloseHandle( token );

            LPWSTR text = nullptr;

            if ( !::ConvertSidToStringSidW( reinterpret_cast<TOKEN_USER*>( storage.data() )->User.Sid, &text ) )
            {
                throwLastError( "Converting the broker's user failed" );
            }

            std::wstring sid( text );
            ::LocalFree( text );

            return sid;
        }
    }

    PipeServer::PipeServer( std::wstring name )
        : name_( std::move( name ) )
    {
        listening_ = createInstance( true );
    }

    PipeServer::~PipeServer()
    {
        if ( listening_ != nullptr )
        {
            ::CloseHandle( listening_ );
        }
    }

    PipeServer::PipeServer( PipeServer&& other ) noexcept
        : name_( std::move( other.name_ ) ), listening_( std::exchange( other.listening_, nullptr ) )
    {
    }

    PipeServer& PipeServer::operator=( PipeServer&& other ) noexcept
    {
        if ( this != &other )
        {
            if ( listening_ != nullptr )
            {
                ::CloseHandle( listening_ );
            }

            name_ = std::move( other.name_ );
            listening_ = std::exchange( other.listening_, nullptr );
        }

        return *this;
    }

    const std::wstring& PipeServer::name() const noexcept
    {
        return name_;
    }

    LineChannel PipeServer::accept()
    {
        if ( !::ConnectNamedPipe( listening_, nullptr ) && ::GetLastError() != ERROR_PIPE_CONNECTED )
        {
            throwLastError( "Waiting for a client failed" );
        }

        HANDLE connected = std::exchange( listening_, nullptr );
        HANDLE duplicate = nullptr;

        if ( !::DuplicateHandle( ::GetCurrentProcess(), connected, ::GetCurrentProcess(), &duplicate, 0, FALSE,
                DUPLICATE_SAME_ACCESS ) )
        {
            ::CloseHandle( connected );
            throwLastError( "Duplicating a client's pipe failed" );
        }

        LineChannel channel( connected, duplicate );
        listening_ = createInstance( false );

        return channel;
    }

    void* PipeServer::createInstance( bool first )
    {
        // Protected (P), so nothing is inherited from the pipe namespace: the user alone, with full access.
        const auto descriptor_text = std::format( L"D:P(A;;GA;;;{})", currentUserSid() );
        PSECURITY_DESCRIPTOR descriptor = nullptr;

        if ( !::ConvertStringSecurityDescriptorToSecurityDescriptorW( descriptor_text.c_str(), SDDL_REVISION_1,
                &descriptor, nullptr ) )
        {
            throwLastError( "Building the pipe's access list failed" );
        }

        SECURITY_ATTRIBUTES attributes{ sizeof( SECURITY_ATTRIBUTES ), descriptor, FALSE };

        HANDLE pipe = ::CreateNamedPipeW( name_.c_str(),
            PIPE_ACCESS_DUPLEX | ( first ? FILE_FLAG_FIRST_PIPE_INSTANCE : 0 ),
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
            PIPE_UNLIMITED_INSTANCES, kPipeBuffer, kPipeBuffer, 0, &attributes );

        const auto error = ::GetLastError();
        ::LocalFree( descriptor );

        if ( pipe == INVALID_HANDLE_VALUE )
        {
            throw std::system_error( static_cast<int>( error ), std::system_category(),
                first ? "Creating the broker's pipe failed; another program may hold its name"
                      : "Creating the broker's next pipe instance failed" );
        }

        return pipe;
    }
}
