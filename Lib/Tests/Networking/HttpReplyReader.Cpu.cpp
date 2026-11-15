/**
 * @file HttpReplyReader.Cpu.cpp
 * @brief An HTTP/1.1 reply read as it arrives: each way a body is framed, a reply split anywhere, and each reply
 *        refused rather than misread (HttpClient.md section 9).
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <string>
#include <string_view>

import Mila;

namespace Mila::Tests::Networking
{
    using namespace Mila::Networking;

    namespace
    {
        struct Read
        {
            bool accepted{ false };
            bool ended{ false };
            HttpReplyReader reader;
            std::string body;
        };

        /// Reads a reply in pieces of `piece` bytes, then the close.
        Read readReply( std::string_view reply, std::size_t piece = 1 << 20, bool close = true )
        {
            Read result;
            result.accepted = true;

            for ( std::size_t offset = 0; offset < reply.size() && result.accepted; offset += piece )
            {
                result.accepted = result.reader.read( reply.substr( offset, piece ), [&result]( const char* data, std::size_t length )
                    {
                        result.body.append( data, length );

                        return true;
                    } );
            }

            result.ended = result.accepted && close ? result.reader.finish() : result.reader.complete();

            return result;
        }

        constexpr std::string_view kChunked =
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
            "5;name=value\r\nhello\r\n1A\r\nabcdefghijklmnopqrstuvwxyz\r\n0\r\nTrailer: kept out\r\n\r\n";
    }

    TEST( HttpReplyReader, ABodyByLengthEndsAtItsLength )
    {
        const Read read = readReply( "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello and what follows", 1 << 20, false );

        EXPECT_TRUE( read.ended );
        EXPECT_EQ( read.reader.status(), 200 );
        EXPECT_EQ( read.reader.contentLength(), 5u );
        EXPECT_EQ( read.body, "hello" );
    }

    TEST( HttpReplyReader, AChunkedBodyIsJoinedWithoutItsFraming )
    {
        const Read read = readReply( kChunked, 1 << 20, false );

        EXPECT_TRUE( read.ended );
        EXPECT_EQ( read.body, "helloabcdefghijklmnopqrstuvwxyz" );
        EXPECT_EQ( read.reader.contentLength(), 0u );
    }

    TEST( HttpReplyReader, AReplySplitAtEveryByteReadsAsOne )
    {
        const Read whole = readReply( kChunked );
        const Read split = readReply( kChunked, 1 );

        EXPECT_TRUE( split.ended );
        EXPECT_EQ( split.body, whole.body );
    }

    TEST( HttpReplyReader, ABodyWithNoLengthRunsToTheClose )
    {
        const Read read = readReply( "HTTP/1.0 200 OK\r\n\r\nto the end" );

        EXPECT_TRUE( read.ended );
        EXPECT_EQ( read.body, "to the end" );
    }

    TEST( HttpReplyReader, AnInterimReplyIsPassedOver )
    {
        const Read read = readReply( "HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 201 Created\r\nContent-Length: 2\r\n\r\nok" );

        EXPECT_TRUE( read.ended );
        EXPECT_EQ( read.reader.status(), 201 );
        EXPECT_EQ( read.body, "ok" );
    }

    TEST( HttpReplyReader, ARedirectNamesWhereItLeads )
    {
        const Read read = readReply( "HTTP/1.1 302 Found\r\nLOCATION:   /elsewhere  \r\ncontent-length: 0\r\n\r\n" );

        EXPECT_TRUE( read.ended );
        EXPECT_EQ( read.reader.status(), 302 );
        EXPECT_EQ( read.reader.location(), "/elsewhere" );
    }

    TEST( HttpReplyReader, NoContentHasNoBody )
    {
        const Read read = readReply( "HTTP/1.1 204 No Content\r\n\r\n", 1 << 20, false );

        EXPECT_TRUE( read.ended );
        EXPECT_TRUE( read.body.empty() );
    }

    TEST( HttpReplyReader, ATransferCodingOverridesALength )
    {
        const Read read = readReply( "HTTP/1.1 200 OK\r\nContent-Length: 999\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nok\r\n0\r\n\r\n", 1 << 20, false );

        EXPECT_TRUE( read.ended );
        EXPECT_EQ( read.body, "ok" );
    }

    TEST( HttpReplyReader, AReplyCutShortIsNotEnded )
    {
        const Read read = readReply( "HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nhalf" );

        EXPECT_FALSE( read.ended );
        EXPECT_TRUE( read.reader.headComplete() );
        EXPECT_FALSE( read.reader.error().empty() );
    }

    TEST( HttpReplyReader, ACloseBeforeAnyReplyIsNotEnded )
    {
        const Read read = readReply( "" );

        EXPECT_FALSE( read.ended );
        EXPECT_FALSE( read.reader.headComplete() );
    }

    TEST( HttpReplyReader, TwoLengthsAreRefused )
    {
        EXPECT_FALSE( readReply( "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\nok" ).accepted );
        EXPECT_TRUE( readReply( "HTTP/1.1 200 OK\r\nContent-Length: 2, 2\r\n\r\nok" ).ended );
        EXPECT_FALSE( readReply( "HTTP/1.1 200 OK\r\nContent-Length: -2\r\n\r\nok" ).accepted );
    }

    TEST( HttpReplyReader, WhatIsNotAnHttpReplyIsRefused )
    {
        EXPECT_FALSE( readReply( "SSH-2.0-OpenSSH\r\n\r\n" ).accepted );
        EXPECT_FALSE( readReply( "HTTP/1.1 2000 OK\r\n\r\n" ).accepted );
        EXPECT_FALSE( readReply( "HTTP/1.1 200 OK\r\nno colon here\r\n\r\n" ).accepted );
        EXPECT_FALSE( readReply( "HTTP/1.1 200 OK\r\nA: b\r\n folded\r\n\r\n" ).accepted );
        EXPECT_FALSE( readReply( "HTTP/1.1 101 Switching Protocols\r\n\r\n" ).accepted );
        EXPECT_FALSE( readReply( "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n" ).accepted );
        EXPECT_FALSE( readReply( "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nokay\r\n" ).accepted );
    }

    TEST( HttpReplyReader, AHeadPastItsBoundIsRefused )
    {
        const std::string reply = "HTTP/1.1 200 OK\r\nX: " + std::string( HttpReplyReader::kMaximumHeadBytes, 'a' ) + "\r\n\r\n";

        EXPECT_FALSE( readReply( reply ).accepted );
    }

    TEST( HttpReplyReader, TheBodyCallbackStopsTheReading )
    {
        HttpReplyReader reader;
        const bool accepted = reader.read( "HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nstop", []( const char*, std::size_t )
            {
                return false;
            } );

        EXPECT_FALSE( accepted );
        EXPECT_TRUE( reader.stopped() );
        EXPECT_TRUE( reader.error().empty() );
    }
}
