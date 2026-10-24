#define NOMINMAX
#include <windows.h>
#include <gtest/gtest.h>

import std;
import ToolBroker.LineChannel;

namespace Mila::Tests::Apps::ToolBroker
{
    using Mila::ToolBroker::LineChannel;

    namespace
    {
        /** Two channels joined by anonymous pipes, so each test is both ends of one connection. */
        struct ChannelPair
        {
            LineChannel local;
            LineChannel remote;
        };

        ChannelPair makePair( std::size_t limit = LineChannel::kDefaultMessageLimit )
        {
            HANDLE local_read = nullptr;
            HANDLE remote_write = nullptr;
            HANDLE remote_read = nullptr;
            HANDLE local_write = nullptr;

            if ( !::CreatePipe( &local_read, &remote_write, nullptr, 0 ) || !::CreatePipe( &remote_read, &local_write, nullptr, 0 ) )
            {
                throw std::runtime_error( "CreatePipe failed" );
            }

            return { LineChannel( local_read, local_write, limit ), LineChannel( remote_read, remote_write, limit ) };
        }

        void writeRaw( HANDLE handle, std::string_view bytes )
        {
            DWORD done = 0;
            ASSERT_TRUE( ::WriteFile( handle, bytes.data(), static_cast<DWORD>( bytes.size() ), &done, nullptr ) );
        }
    }

    TEST( LineChannelTests, RoundTripsOneMessage )
    {
        auto pair = makePair();

        pair.local.send( R"({"jsonrpc":"2.0","id":1,"method":"tools/list"})" );

        EXPECT_EQ( pair.remote.receive(), R"({"jsonrpc":"2.0","id":1,"method":"tools/list"})" );
    }

    TEST( LineChannelTests, ReturnsMessagesInOrderWhenOneReadHoldsSeveral )
    {
        auto pair = makePair();

        pair.local.send( "first" );
        pair.local.send( "second" );
        pair.local.send( "" );

        EXPECT_EQ( pair.remote.receive(), "first" );
        EXPECT_EQ( pair.remote.receive(), "second" );
        EXPECT_EQ( pair.remote.receive(), "" );
    }

    TEST( LineChannelTests, AssemblesAMessageSplitAcrossWrites )
    {
        HANDLE read = nullptr;
        HANDLE write = nullptr;
        ASSERT_TRUE( ::CreatePipe( &read, &write, nullptr, 0 ) );
        LineChannel channel( read, nullptr );

        writeRaw( write, "par" );
        writeRaw( write, "tial\nnext\n" );

        EXPECT_EQ( channel.receive(), "partial" );
        EXPECT_EQ( channel.receive(), "next" );
        ::CloseHandle( write );
    }

    TEST( LineChannelTests, DropsTheCarriageReturnOfATextModeLine )
    {
        HANDLE read = nullptr;
        HANDLE write = nullptr;
        ASSERT_TRUE( ::CreatePipe( &read, &write, nullptr, 0 ) );
        LineChannel channel( read, nullptr );

        writeRaw( write, "text mode\r\n" );

        EXPECT_EQ( channel.receive(), "text mode" );
        ::CloseHandle( write );
    }

    TEST( LineChannelTests, RefusesToSendAMessageHoldingANewline )
    {
        auto pair = makePair();

        EXPECT_THROW( pair.local.send( "one\ntwo" ), std::invalid_argument );
    }

    TEST( LineChannelTests, ReturnsNothingOnceTheOtherEndCloses )
    {
        auto pair = makePair();

        pair.local.send( "last" );
        pair.local.closeSending();

        EXPECT_EQ( pair.remote.receive(), "last" );
        EXPECT_EQ( pair.remote.receive(), std::nullopt );
        EXPECT_EQ( pair.remote.receive(), std::nullopt );
    }

    TEST( LineChannelTests, DropsALineTheOtherEndNeverFinished )
    {
        HANDLE read = nullptr;
        HANDLE write = nullptr;
        ASSERT_TRUE( ::CreatePipe( &read, &write, nullptr, 0 ) );
        LineChannel channel( read, nullptr );

        writeRaw( write, "unfinished" );
        ::CloseHandle( write );

        EXPECT_EQ( channel.receive(), std::nullopt );
    }

    TEST( LineChannelTests, RefusesAMessageLongerThanItsLimit )
    {
        auto pair = makePair( 8 );

        pair.local.send( "123456789" );

        EXPECT_THROW( pair.remote.receive(), std::length_error );
    }

    TEST( LineChannelTests, RefusesAnUnendedMessageOnceItPassesTheLimit )
    {
        HANDLE read = nullptr;
        HANDLE write = nullptr;
        ASSERT_TRUE( ::CreatePipe( &read, &write, nullptr, 0 ) );
        LineChannel channel( read, nullptr, 8 );

        writeRaw( write, "a line that never ends" );

        EXPECT_THROW( channel.receive(), std::length_error );
        ::CloseHandle( write );
    }

    TEST( LineChannelTests, RefusesToSendAfterClosingTheSendingSide )
    {
        auto pair = makePair();

        pair.local.closeSending();

        EXPECT_THROW( pair.local.send( "late" ), std::logic_error );
    }
}
