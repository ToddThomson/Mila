// Contract tests for the reply reader every chat template reads its replies through
// (Dnn.Models.Conversation.ReplyReader).
//
// The reader is what makes a streamed reply and a parsed one the same reply, so the central test cuts each
// family's replies into pieces every way that matters -- a byte at a time, three bytes at a time, whole --
// and requires one reading of all of them: the same Reply, and events whose text is that Reply's text.

#include <gtest/gtest.h>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Models
{
    namespace Conversation = Mila::Dnn::Conversation;

    using Conversation::ReplyEvent;

    namespace
    {
        struct Streamed
        {
            Conversation::Reply reply;
            std::string reasoning;
            std::string answer;
            int calls_opened{ 0 };
            int calls_closed{ 0 };
            int reasoning_opened{ 0 };
            std::vector<ReplyEvent::Kind> kinds;
        };

        void collect( Streamed& streamed, std::span<const ReplyEvent> events )
        {
            for ( const auto& event : events )
            {
                streamed.kinds.push_back( event.kind );

                switch ( event.kind )
                {
                    case ReplyEvent::Kind::Reasoning:
                        streamed.reasoning += event.text;
                        break;

                    case ReplyEvent::Kind::Answer:
                        streamed.answer += event.text;
                        break;

                    case ReplyEvent::Kind::ToolCallOpened:
                        ++streamed.calls_opened;
                        break;

                    case ReplyEvent::Kind::ToolCallClosed:
                        ++streamed.calls_closed;
                        break;

                    case ReplyEvent::Kind::ReasoningOpened:
                        ++streamed.reasoning_opened;
                        break;

                    case ReplyEvent::Kind::ReasoningClosed:
                        break;
                }
            }
        }

        /// The reply fed in pieces of @p size bytes; zero feeds it whole.
        Streamed stream( const Conversation::Protocol& protocol, std::string_view text, bool begins_in_reasoning, std::size_t size )
        {
            Streamed streamed;
            Conversation::ReplyReader reader = protocol.readReply( { {}, begins_in_reasoning } );

            if ( size == 0 )
            {
                collect( streamed, reader.feed( text ) );
            } else {
                for ( std::size_t at = 0; at < text.size(); at += size )
                {
                    collect( streamed, reader.feed( text.substr( at, size ) ) );
                }
            }

            collect( streamed, reader.finish() );
            streamed.reply = reader.reply();

            return streamed;
        }

        void expectSameReply( const Conversation::Reply& actual, const Conversation::Reply& expected )
        {
            EXPECT_EQ( actual.reasoning, expected.reasoning );
            EXPECT_EQ( actual.content, expected.content );
            EXPECT_EQ( actual.malformed_tool_calls, expected.malformed_tool_calls );
            ASSERT_EQ( actual.tool_calls.size(), expected.tool_calls.size() );

            for ( std::size_t index = 0; index < actual.tool_calls.size(); ++index )
            {
                EXPECT_EQ( actual.tool_calls[ index ].name, expected.tool_calls[ index ].name );
                EXPECT_EQ( actual.tool_calls[ index ].arguments, expected.tool_calls[ index ].arguments );
            }
        }

        struct Case
        {
            std::string_view family;
            std::string_view text;
            bool begins_in_reasoning{ false };
        };

        std::unique_ptr<Conversation::Protocol> protocolOf( std::string_view family )
        {
            if ( family == "gemma" )
            {
                return std::make_unique<Mila::Dnn::Gemma::GemmaChatProtocol>();
            }

            if ( family == "qwen" )
            {
                return std::make_unique<Mila::Dnn::Qwen::QwenChatProtocol>();
            }

            return std::make_unique<Mila::Dnn::Llama::LlamaChatProtocol>( Mila::Dnn::Llama::LlamaChatProtocol::llama31() );
        }

        const Case kCases[] = {
            { "gemma", "<|channel>thought\nThe capital is Canberra.<channel|>Canberra." },
            { "gemma", "The tool said sunny.<channel|>It is sunny.", true },
            { "gemma", "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_call>call:get_time{city:<|\"|>Paris<|\"|>}<tool_call|>" },
            { "gemma", "Sure.<|tool_call>get_weather{city:1}<tool_call|>" },
            { "gemma", "<|tool_call>call:run:cmd=ls" },
            { "gemma", "<|channel>thought\nFirst.<channel|>  <|channel>thought\n  Second.  <channel|>Text <|turn> and <|odd|>more <|>end.  " },
            { "gemma", "<|channel>Not a label line<channel|>Answer" },
            { "gemma", "<|channel>thought\nCut off mid" },
            { "gemma", "Before <|tool_response>response:x{value:1}<tool_response|> after <channel|> stray" },
            { "gemma", "  h\xC3\xA9llo w\xC3\xB6rld \xE6\x97\xA5\xE6\x9C\xAC  " },
            { "gemma", "a < b and a <| b, then <|tool_cal" },
            { "qwen", "The capital is Canberra.\n</think>\n\nCanberra.", true },
            { "qwen", "Still thinking about", true },
            { "qwen", "Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n"
                "<tool_call>\n<function=get_time>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>" },
            { "qwen", "Sure. <tool_call>\nget_weather\n</tool_call>" },
            { "qwen", "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar" },
            { "qwen", "Plan.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>", true },
            { "llama", "Canberra is the capital." },
            { "llama", R"({"name": "get_weather", "parameters": {"city": "Paris"}})" },
            { "llama", R"(Checking.<|python_tag|>{"name": "get_weather", "arguments": {"city": "Paris"}}<|eom_id|>ignored)" },
            { "llama", R"({"name": "get_weather", "parameters": "Paris"})" },
            { "llama", "  \n{\"name\": \"get_time\", \"parameters\": {}}  \n" },
            { "llama", "The set {1, 2} is small." },
        };
    }

    TEST( ReplyReader, AReplyReadsTheSameHoweverItIsCut )
    {
        for ( const auto& reply_case : kCases )
        {
            SCOPED_TRACE( std::string( reply_case.family ) + ": " + std::string( reply_case.text ) );

            const auto protocol = protocolOf( reply_case.family );
            const Streamed whole = stream( *protocol, reply_case.text, reply_case.begins_in_reasoning, 0 );

            for ( const std::size_t size : { std::size_t{ 1 }, std::size_t{ 2 }, std::size_t{ 3 }, std::size_t{ 7 } } )
            {
                SCOPED_TRACE( std::format( "pieces of {}", size ) );

                const Streamed cut = stream( *protocol, reply_case.text, reply_case.begins_in_reasoning, size );

                expectSameReply( cut.reply, whole.reply );
            }
        }
    }

    TEST( ReplyReader, WhatStreamsIsWhatIsParsed )
    {
        for ( const auto& reply_case : kCases )
        {
            SCOPED_TRACE( std::string( reply_case.family ) + ": " + std::string( reply_case.text ) );

            const auto protocol = protocolOf( reply_case.family );
            const Streamed streamed = stream( *protocol, reply_case.text, reply_case.begins_in_reasoning, 1 );

            EXPECT_EQ( streamed.reasoning, streamed.reply.reasoning );
            EXPECT_EQ( streamed.answer, streamed.reply.content );
            EXPECT_EQ( streamed.calls_opened, streamed.calls_closed );
            EXPECT_EQ( static_cast<std::size_t>( streamed.calls_closed ),
                streamed.reply.tool_calls.size() + streamed.reply.malformed_tool_calls.size() );

            expectSameReply( streamed.reply, protocol->parseReply( reply_case.text, { {}, reply_case.begins_in_reasoning } ) );
        }
    }

    TEST( ReplyReader, ReasoningSpansAreTrimmedAndJoined )
    {
        const auto reply = Mila::Dnn::Gemma::GemmaChatProtocol{}.parseReply(
            "<|channel>thought\nFirst.<channel|>  <|channel>thought\n  Second.  <channel|>Text <|turn> and <|odd|>more <|>end.  ", {} );

        EXPECT_EQ( reply.reasoning, "First.\nSecond." );
        EXPECT_EQ( reply.content, "Text  and more end." );
    }

    TEST( ReplyReader, AModelWrittenResultIsNotPartOfTheReply )
    {
        const auto reply = Mila::Dnn::Gemma::GemmaChatProtocol{}.parseReply(
            "Before <|tool_response>response:x{value:1}<tool_response|> after <channel|> stray", {} );

        EXPECT_EQ( reply.content, "Before  after  stray" );
    }

    TEST( ReplyReader, AMarkerCutAcrossPiecesIsNeverShown )
    {
        Mila::Dnn::Gemma::GemmaChatProtocol protocol;
        Conversation::ReplyReader reader = protocol.readReply( {} );

        Streamed streamed;
        collect( streamed, reader.feed( "Hi <|chan" ) );

        EXPECT_EQ( streamed.answer, "Hi" );

        collect( streamed, reader.feed( "nel>thought\nWhy.<channel|> there" ) );
        collect( streamed, reader.finish() );

        // Each side of the span keeps its own space.
        EXPECT_EQ( streamed.answer, "Hi  there" );
        EXPECT_EQ( streamed.reasoning, "Why." );
        EXPECT_EQ( streamed.answer.find( '<' ), std::string::npos );
    }

    TEST( ReplyReader, AChannelLabelIsHeldUntilItsLineEnds )
    {
        Mila::Dnn::Gemma::GemmaChatProtocol protocol;
        Conversation::ReplyReader reader = protocol.readReply( {} );

        Streamed streamed;
        collect( streamed, reader.feed( "<|channel>tho" ) );

        EXPECT_TRUE( streamed.reasoning.empty() );
        EXPECT_EQ( streamed.reasoning_opened, 1 );

        collect( streamed, reader.feed( "ught\nBecause." ) );

        EXPECT_EQ( streamed.reasoning, "Because." );
    }

    TEST( ReplyReader, AnIncompleteCharacterIsHeldForTheNextPiece )
    {
        Mila::Dnn::Gemma::GemmaChatProtocol protocol;
        Conversation::ReplyReader reader = protocol.readReply( {} );

        Streamed first;
        collect( first, reader.feed( "caf\xC3" ) );

        EXPECT_EQ( first.answer, "caf" );

        Streamed second;
        collect( second, reader.feed( "\xA9!" ) );

        EXPECT_EQ( second.answer, "\xC3\xA9!" );
    }

    TEST( ReplyReader, TrailingWhitespaceWaitsForTextAfterIt )
    {
        Mila::Dnn::Qwen::QwenChatProtocol protocol;
        Conversation::ReplyReader reader = protocol.readReply( {} );

        Streamed streamed;
        collect( streamed, reader.feed( "Hello  " ) );

        EXPECT_EQ( streamed.answer, "Hello" );

        collect( streamed, reader.finish() );

        EXPECT_EQ( streamed.answer, "Hello" );
        EXPECT_EQ( reader.reply().content, "Hello" );
    }

    TEST( ReplyReader, EachCallIsReadAsItCloses )
    {
        Mila::Dnn::Qwen::QwenChatProtocol protocol;
        Conversation::ReplyReader reader = protocol.readReply( {} );

        Streamed streamed;
        collect( streamed, reader.feed( "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n" ) );

        EXPECT_EQ( streamed.calls_opened, 1 );
        EXPECT_TRUE( reader.reply().tool_calls.empty() );

        collect( streamed, reader.feed( "</tool_call>\n<tool_call>" ) );

        ASSERT_EQ( reader.reply().tool_calls.size(), 1u );
        EXPECT_EQ( reader.reply().tool_calls[ 0 ].name, "get_weather" );
        EXPECT_EQ( streamed.calls_opened, 2 );
        EXPECT_EQ( streamed.calls_closed, 1 );
    }

    TEST( ReplyReader, AReplyThatOpensWithACallShowsNoText )
    {
        const auto protocol = Mila::Dnn::Llama::LlamaChatProtocol::llama31();
        const Streamed streamed = stream( protocol, R"( {"name": "get_time", "parameters": {}})", false, 1 );

        EXPECT_TRUE( streamed.answer.empty() );
        ASSERT_EQ( streamed.reply.tool_calls.size(), 1u );
        EXPECT_EQ( streamed.reply.tool_calls[ 0 ].name, "get_time" );
    }

    TEST( ReplyReader, ABraceAfterTextIsNotACall )
    {
        const auto reply = Mila::Dnn::Llama::LlamaChatProtocol::llama31().parseReply( "The set {1, 2} is small.", {} );

        EXPECT_EQ( reply.content, "The set {1, 2} is small." );
        EXPECT_TRUE( reply.tool_calls.empty() );
        EXPECT_TRUE( reply.malformed_tool_calls.empty() );
    }

    TEST( ReplyReader, AReplyThatBeginsInReasoningSaysSoFirst )
    {
        Mila::Dnn::Qwen::QwenChatProtocol protocol;
        Conversation::ReplyReader reader = protocol.readReply( { {}, true } );

        const auto events = reader.feed( "Hmm" );

        ASSERT_FALSE( events.empty() );
        EXPECT_EQ( events.front().kind, ReplyEvent::Kind::ReasoningOpened );
    }

    TEST( ReplyReader, NothingIsReadAfterTheReplyFinished )
    {
        Mila::Dnn::Qwen::QwenChatProtocol protocol;
        Conversation::ReplyReader reader = protocol.readReply( {} );
        reader.finish();

        EXPECT_THROW( (void)reader.feed( "more" ), std::logic_error );
    }
}
