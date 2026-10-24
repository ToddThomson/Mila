#define NOMINMAX
#include <windows.h>
#include <aclapi.h>
#include <gtest/gtest.h>

import std;
import ToolBroker.LineChannel;
import ToolBroker.PipeServer;

namespace Mila::Tests::Apps::ToolBroker
{
    using Mila::ToolBroker::LineChannel;
    using Mila::ToolBroker::PipeServer;

    namespace
    {
        std::wstring uniquePipeName()
        {
            static int counter = 0;

            return std::format( L"\\\\.\\pipe\\mila-tool-broker-test-{}-{}", ::GetCurrentProcessId(), counter++ );
        }

        HANDLE openClient( const std::wstring& name, DWORD access = GENERIC_READ | GENERIC_WRITE )
        {
            return ::CreateFileW( name.c_str(), access, 0, nullptr, OPEN_EXISTING, 0, nullptr );
        }

        LineChannel clientChannel( const std::wstring& name )
        {
            HANDLE pipe = openClient( name );

            if ( pipe == INVALID_HANDLE_VALUE )
            {
                throw std::system_error( static_cast<int>( ::GetLastError() ), std::system_category(), "Opening the pipe failed" );
            }

            HANDLE duplicate = nullptr;
            ::DuplicateHandle( ::GetCurrentProcess(), pipe, ::GetCurrentProcess(), &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS );

            return LineChannel( pipe, duplicate );
        }

        std::vector<std::byte> currentUserSid()
        {
            HANDLE token = nullptr;
            ::OpenProcessToken( ::GetCurrentProcess(), TOKEN_QUERY, &token );
            DWORD size = 0;
            ::GetTokenInformation( token, TokenUser, nullptr, 0, &size );
            std::vector<std::byte> storage( size );
            ::GetTokenInformation( token, TokenUser, storage.data(), size, &size );
            ::CloseHandle( token );

            const PSID sid = reinterpret_cast<TOKEN_USER*>( storage.data() )->User.Sid;
            std::vector<std::byte> copy( ::GetLengthSid( sid ) );
            ::CopySid( static_cast<DWORD>( copy.size() ), copy.data(), sid );

            return copy;
        }
    }

    TEST( PipeServerTests, CarriesAMessageEachWay )
    {
        PipeServer server( uniquePipeName() );

        std::jthread serving( [&server]
            {
                auto channel = server.accept();
                const auto request = channel.receive();
                channel.send( "reply to " + request.value_or( "<closed>" ) );
            } );

        auto client = clientChannel( server.name() );
        client.send( "request" );

        EXPECT_EQ( client.receive(), "reply to request" );
    }

    TEST( PipeServerTests, ServesASecondClientAfterTheFirst )
    {
        PipeServer server( uniquePipeName() );

        std::jthread serving( [&server]
            {
                for ( int client = 0; client < 2; ++client )
                {
                    auto channel = server.accept();
                    channel.send( std::format( "client {}", client ) );
                }
            } );

        auto first = clientChannel( server.name() );
        EXPECT_EQ( first.receive(), "client 0" );

        auto second = clientChannel( server.name() );
        EXPECT_EQ( second.receive(), "client 1" );
    }

    TEST( PipeServerTests, RefusesANameAlreadyInUse )
    {
        PipeServer server( uniquePipeName() );

        EXPECT_THROW( PipeServer( server.name() ), std::system_error );
    }

    TEST( PipeServerTests, GrantsTheCurrentUserAlone )
    {
        PipeServer server( uniquePipeName() );
        HANDLE client = openClient( server.name(), GENERIC_READ | READ_CONTROL );
        ASSERT_NE( client, INVALID_HANDLE_VALUE );

        PACL dacl = nullptr;
        PSECURITY_DESCRIPTOR descriptor = nullptr;
        ASSERT_EQ( ::GetSecurityInfo( client, SE_KERNEL_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, &dacl,
            nullptr, &descriptor ), static_cast<DWORD>( ERROR_SUCCESS ) );
        ASSERT_NE( dacl, nullptr );

        ACL_SIZE_INFORMATION size{};
        ASSERT_TRUE( ::GetAclInformation( dacl, &size, sizeof( size ), AclSizeInformation ) );
        EXPECT_EQ( size.AceCount, 1u );

        void* ace = nullptr;
        ASSERT_TRUE( ::GetAce( dacl, 0, &ace ) );
        auto* allowed = static_cast<ACCESS_ALLOWED_ACE*>( ace );
        EXPECT_EQ( allowed->Header.AceType, ACCESS_ALLOWED_ACE_TYPE );

        auto user = currentUserSid();
        EXPECT_TRUE( ::EqualSid( &allowed->SidStart, user.data() ) );

        ::LocalFree( descriptor );
        ::CloseHandle( client );
    }
}
