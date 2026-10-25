#include <gtest/gtest.h>

import std;
import Chat.Message;
import Chat.ToolCallParser;

namespace Mila::Tests::Apps::Chat
{
    using Mila::ChatApp::ToolCallParser;

    TEST( ToolCallParserTests, ReadsAReplyThatIsAPythonicCall )
    {
        const auto call = ToolCallParser::parse( R"([get_weather(location="Paris")])" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( call->name, "get_weather" );
        EXPECT_EQ( call->arguments, R"({"location":"Paris"})" );
    }

    TEST( ToolCallParserTests, ReadsACallAfterLeadingWhitespace )
    {
        const auto call = ToolCallParser::parse( "\n  [get_weather(location=\"Paris\")]" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( call->name, "get_weather" );
    }

    TEST( ToolCallParserTests, ReadsASingleQuotedArgumentWhole )
    {
        const auto call = ToolCallParser::parse( "[get_weather(location='Paris, France', unit='celsius')]" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( call->arguments, R"({"location":"Paris, France","unit":"celsius"})" );
    }

    TEST( ToolCallParserTests, ReadsAReplyThatIsAJsonCall )
    {
        const auto call = ToolCallParser::parse( R"({"name": "get_weather", "arguments": {"location": "Paris"}})" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( call->name, "get_weather" );
        EXPECT_EQ( call->arguments, R"({"location":"Paris"})" );
    }

    TEST( ToolCallParserTests, ReadsATaggedCall )
    {
        const auto call = ToolCallParser::parse( R"(<|python_tag|>{"name": "get_weather", "arguments": {"location": "Paris"}})" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( call->name, "get_weather" );
    }

    TEST( ToolCallParserTests, LeavesABracketInProseAlone )
    {
        EXPECT_FALSE( ToolCallParser::parse( "In Python, [print(x=1)] builds a list from one call." ).has_value() );
        EXPECT_FALSE( ToolCallParser::parse( "Paris is the capital of France [1]." ).has_value() );
    }

    TEST( ToolCallParserTests, LeavesJsonInProseAlone )
    {
        EXPECT_FALSE( ToolCallParser::parse(
            R"(A call looks like {"name": "get_weather", "arguments": {"location": "Paris"}} on the wire.)" ).has_value() );
    }

    TEST( ToolCallParserTests, ReadsNothingFromAnEmptyReply )
    {
        EXPECT_FALSE( ToolCallParser::parse( "" ).has_value() );
        EXPECT_FALSE( ToolCallParser::parse( " \n " ).has_value() );
    }
}
