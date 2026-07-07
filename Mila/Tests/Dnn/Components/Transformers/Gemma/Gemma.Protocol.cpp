// Unit tests for the canonical Gemma 4 grammar module (Dnn.Components.GemmaProtocol).
// This is the union-of-both-implementations coverage the drought never had: it pins the
// spec-verified behaviors the runtime module now owns -- the <|"|> string delimiter on both
// parse and format, integer-preserving argument coercion, namespace stripping, and the
// tool-response output-field distillation with failed-tool error surfacing.

#include <gtest/gtest.h>
#include <string>
#include <optional>

import Mila;
import nlohmann.json;

namespace Dnn::Components::Transformers::Tests
{
    using namespace Mila::Dnn::Gemma;

    // ---- parseToolCall: detection + malformed input -------------------------

    TEST( GemmaProtocolParse, NoToolCallOpen_ReturnsNullopt )
    {
        EXPECT_FALSE( parseToolCall( "just some model prose with no call" ).has_value() );
    }

    TEST( GemmaProtocolParse, MissingCallPrefix_ReturnsNullopt )
    {
        EXPECT_FALSE( parseToolCall( "<|tool_call>get_weather{location: \"x\"}<tool_call|>" ).has_value() );
    }

    TEST( GemmaProtocolParse, MissingBraces_ReturnsNullopt )
    {
        EXPECT_FALSE( parseToolCall( "<|tool_call>call:get_weather<tool_call|>" ).has_value() );
    }

    TEST( GemmaProtocolParse, EmptyName_ReturnsNullopt )
    {
        EXPECT_FALSE( parseToolCall( "<|tool_call>call:{a: 1}<tool_call|>" ).has_value() );
    }

    // ---- parseToolCall: argument grammar ------------------------------------

    TEST( GemmaProtocolParse, PlainQuotedStringArgument )
    {
        const auto call = parseToolCall( "<|tool_call>call:get_weather{location: \"London\"}<tool_call|>" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( call->name, "get_weather" );
        EXPECT_EQ( nlohmann::json::parse( call->arguments ).at( "location" ), "London" );
    }

    TEST( GemmaProtocolParse, StringDelimiterArgument )
    {
        // The trained <|"|> delimiter -- the drift this consolidation closed. The old
        // Chat parser saw the leading token as a bare literal and mangled the value.
        const auto call = parseToolCall( "<|tool_call>call:run{cmd: <|\"|>ls -F<|\"|>}<tool_call|>" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( nlohmann::json::parse( call->arguments ).at( "cmd" ), "ls -F" );
    }

    TEST( GemmaProtocolParse, QuotedStringWithEmbeddedComma )
    {
        const auto call = parseToolCall( "<|tool_call>call:get_weather{location: \"Toronto, Canada\"}<tool_call|>" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( nlohmann::json::parse( call->arguments ).at( "location" ), "Toronto, Canada" );
    }

    TEST( GemmaProtocolParse, IntegerArgumentStaysInteger )
    {
        // Spec-verified behavior: 42 is an integer, not 42.0. The retired C++ parser
        // coerced every bare literal through strtod and lost the distinction.
        const auto call = parseToolCall( "<|tool_call>call:set{count: 42}<tool_call|>" );

        ASSERT_TRUE( call.has_value() );
        const auto args = nlohmann::json::parse( call->arguments );
        EXPECT_TRUE( args.at( "count" ).is_number_integer() );
        EXPECT_EQ( args.at( "count" ), 42 );
    }

    TEST( GemmaProtocolParse, BoolAndFloatAndMultipleArguments )
    {
        const auto call = parseToolCall(
            "<|tool_call>call:configure{enabled: true, ratio: 0.5, label: \"hi\"}<tool_call|>" );

        ASSERT_TRUE( call.has_value() );
        const auto args = nlohmann::json::parse( call->arguments );
        EXPECT_EQ( args.at( "enabled" ), true );
        EXPECT_TRUE( args.at( "ratio" ).is_number_float() );
        EXPECT_EQ( args.at( "label" ), "hi" );
    }

    TEST( GemmaProtocolParse, NamespacedNameStrippedToBareHandler )
    {
        const auto colon = parseToolCall( "<|tool_call>call:default_api:get_weather{location: \"x\"}<tool_call|>" );
        const auto dot = parseToolCall( "<|tool_call>call:default_api.get_weather{location: \"x\"}<tool_call|>" );

        ASSERT_TRUE( colon.has_value() );
        ASSERT_TRUE( dot.has_value() );
        EXPECT_EQ( colon->name, "get_weather" );
        EXPECT_EQ( dot->name, "get_weather" );
    }

    TEST( GemmaProtocolParse, LastCallWinsWhenMultiplePresent )
    {
        const auto call = parseToolCall(
            "<|tool_call>call:first{a: 1}<tool_call|> ... <|tool_call>call:second{b: 2}<tool_call|>" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( call->name, "second" );
    }

    // ---- formatToolCall: round trip -----------------------------------------

    TEST( GemmaProtocolFormat, ToolCallRoundTripsThroughParse )
    {
        const std::string rendered = formatToolCall( "get_weather", R"({"location":"Paris"})" );

        // String values render in the trained delimiter, not plain quotes.
        EXPECT_NE( rendered.find( std::string( kStringDelimiter ) + "Paris" + std::string( kStringDelimiter ) ),
            std::string::npos );

        const auto reparsed = parseToolCall( rendered );
        ASSERT_TRUE( reparsed.has_value() );
        EXPECT_EQ( reparsed->name, "get_weather" );
        EXPECT_EQ( nlohmann::json::parse( reparsed->arguments ).at( "location" ), "Paris" );
    }

    TEST( GemmaProtocolFormat, ToolCallNonObjectArgumentsDegradeToBareCall )
    {
        EXPECT_EQ( formatToolCall( "ping", "not json" ),
            std::string( kToolCallOpen ) + "call:ping{}" + std::string( kToolCallClose ) );
    }

    // ---- formatToolResponse: envelope distillation --------------------------

    TEST( GemmaProtocolResponse, DistillsPrimaryOutputField )
    {
        const std::string out = formatToolResponse( "get_weather", R"({"output":"cloudy, 18C","chunk_id":7})" );

        // The metadata sibling is dropped; the output is surfaced as `result`.
        EXPECT_NE( out.find( "result:" ), std::string::npos );
        EXPECT_NE( out.find( "cloudy, 18C" ), std::string::npos );
        EXPECT_EQ( out.find( "chunk_id" ), std::string::npos );
    }

    TEST( GemmaProtocolResponse, FailedToolSurfacesErrorField )
    {
        // Empty content must not shadow the error; without the error the model
        // sees an empty result and blind-retries.
        const std::string out = formatToolResponse( "run", R"({"content":"","error":"command not found"})" );

        EXPECT_NE( out.find( "error:" ), std::string::npos );
        EXPECT_NE( out.find( "command not found" ), std::string::npos );
    }

    TEST( GemmaProtocolResponse, NonJsonResultPassesThroughAsString )
    {
        const std::string out = formatToolResponse( "echo", "plain text result" );

        EXPECT_NE( out.find( std::string( kToolResponseOpen ) + "response:echo{" ), std::string::npos );
        EXPECT_NE( out.find( "plain text result" ), std::string::npos );
        EXPECT_NE( out.find( std::string( kStringDelimiter ) ), std::string::npos );
    }

    // ---- Control-token constants --------------------------------------------

    TEST( GemmaProtocolConstants, StringDelimiterIsTheRegisteredToken )
    {
        EXPECT_EQ( std::string( kStringDelimiter ), "<|\"|>" );
        EXPECT_EQ( std::string( kToolCallOpen ), "<|tool_call>" );
        EXPECT_EQ( std::string( kToolCallClose ), "<tool_call|>" );
    }
}
