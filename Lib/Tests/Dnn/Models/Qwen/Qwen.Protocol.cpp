// Unit tests for the canonical Qwen 3.8 chat protocol (Dnn.Models.QwenProtocol).
//
// The grammar shipped in the Chat adaptor with no coverage in either language. These pin the
// three places the checkpoint's own chat template differs from the Qwen 3 conventions it
// resembles -- each of which fails SILENTLY, which is why they are worth a test rather than a
// comment: the call grammar is nested tags and not a JSON object, reasoning effort is a trained
// parameter whose middle level emits nothing, and the generation primer leaves the reasoning
// span OPEN when thinking is on.
//
// Read from tokenizer_config.json in the checkpoint, not from a vendor summary.

#include <gtest/gtest.h>
#include <string>
#include <vector>

import Mila;
import nlohmann.json;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn::Qwen;

    namespace Conversation = Mila::Dnn::Conversation;

    // ---- parseToolCall: detection ------------------------------------------

    TEST( QwenProtocolParse, NoToolCallSpan_ReturnsNullopt )
    {
        EXPECT_FALSE( parseToolCall( "just some model prose with no call" ).has_value() );
    }

    TEST( QwenProtocolParse, ProseContainingBracketsIsNotACall )
    {
        // The span anchors the parse. A parser keyed on anything looser routes ordinary prose
        // -- markdown links, footnotes, array literals -- into the call path.
        EXPECT_FALSE( parseToolCall( "see [1] and the array [1, 2, 3] above" ).has_value() );
    }

    TEST( QwenProtocolParse, UnterminatedSpan_ReturnsNullopt )
    {
        // Generation stopped inside the call. Half a call must not be dispatched.
        EXPECT_FALSE( parseToolCall(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n" ).has_value() );
    }

    TEST( QwenProtocolParse, ReasoningBeforeTheCallIsAllowed )
    {
        // The template explicitly permits natural language BEFORE a call.
        const auto call = parseToolCall(
            "I should check the weather first.\n"
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
            "</function>\n</tool_call>" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( call->name, "get_weather" );
    }

    // ---- parseToolCall: malformed spans surface rather than degrade ---------

    TEST( QwenProtocolParse, SpanWithoutFunctionBlock_Throws )
    {
        EXPECT_THROW( (void)parseToolCall( "<tool_call>\nget_weather\n</tool_call>" ),
            std::runtime_error );
    }

    TEST( QwenProtocolParse, UnnamedFunction_Throws )
    {
        EXPECT_THROW( (void)parseToolCall( "<tool_call>\n<function=>\n</function>\n</tool_call>" ),
            std::runtime_error );
    }

    // ---- parseToolCall: parameter typing ------------------------------------

    TEST( QwenProtocolParse, ParameterValuesAreTypedWhenTheyParseAsJson )
    {
        // The grammar carries no types: `4096` and the word `high` look identical. JSON-when-
        // valid recovers numbers and booleans, and prose is never valid JSON by accident.
        const auto call = parseToolCall(
            "<tool_call>\n<function=configure>\n"
            "<parameter=tokens>\n4096\n</parameter>\n"
            "<parameter=enabled>\ntrue\n</parameter>\n"
            "<parameter=mode>\nhigh\n</parameter>\n"
            "</function>\n</tool_call>" );

        ASSERT_TRUE( call.has_value() );

        const auto arguments = nlohmann::json::parse( call->arguments );

        EXPECT_TRUE( arguments[ "tokens" ].is_number_integer() );
        EXPECT_EQ( arguments[ "tokens" ].get<int>(), 4096 );
        EXPECT_TRUE( arguments[ "enabled" ].is_boolean() );
        EXPECT_TRUE( arguments[ "mode" ].is_string() );
        EXPECT_EQ( arguments[ "mode" ].get<std::string>(), "high" );
    }

    TEST( QwenProtocolParse, FramingNewlinesAreNotPartOfTheValue )
    {
        // The template writes a newline after the tag and before the closing one. A value that
        // kept them would differ from the same value passed any other way.
        const auto call = parseToolCall(
            "<tool_call>\n<function=echo>\n<parameter=text>\nhello\n</parameter>\n"
            "</function>\n</tool_call>" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( nlohmann::json::parse( call->arguments )[ "text" ].get<std::string>(), "hello" );
    }

    TEST( QwenProtocolParse, AMultiLineValueKeepsItsInteriorNewlines )
    {
        const auto call = parseToolCall(
            "<tool_call>\n<function=write>\n<parameter=body>\nline one\nline two\n</parameter>\n"
            "</function>\n</tool_call>" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( nlohmann::json::parse( call->arguments )[ "body" ].get<std::string>(),
            "line one\nline two" );
    }

    // ---- formatToolCall ------------------------------------------------------

    TEST( QwenProtocolFormat, RendersNestedTagsNotJson )
    {
        // The single most consequential difference from Qwen 3 convention: a JSON-object call
        // body is a different grammar and the model does not emit it.
        const std::string rendered = formatToolCall( "get_weather", R"({"city":"Paris"})" );

        EXPECT_EQ( rendered,
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
            "</function>\n</tool_call>" );
    }

    TEST( QwenProtocolFormat, StringValuesAreSplicedRawAndOthersReEncoded )
    {
        // The template's own rule, and it is what lets a parameter hold prose containing
        // quotes or newlines with no escaping.
        const std::string rendered = formatToolCall( "f", R"({"n":42,"s":"a \"quoted\" word"})" );

        EXPECT_NE( rendered.find( "<parameter=n>\n42\n</parameter>" ), std::string::npos );
        EXPECT_NE( rendered.find( "<parameter=s>\na \"quoted\" word\n</parameter>" ),
            std::string::npos );
    }

    TEST( QwenProtocolFormat, RoundTripsThroughParse )
    {
        const auto call = parseToolCall( formatToolCall( "f", R"({"a":1,"b":"two"})" ) );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( call->name, "f" );

        const auto arguments = nlohmann::json::parse( call->arguments );

        EXPECT_EQ( arguments[ "a" ].get<int>(), 1 );
        EXPECT_EQ( arguments[ "b" ].get<std::string>(), "two" );
    }

    // ---- reasoning effort ----------------------------------------------------

    TEST( QwenProtocolReasoning, MediumEmitsNothing )
    {
        // Deliberate: the template has no instruction text for medium, so the model runs at its
        // trained default. Inventing wording would be a prompt it was never tuned against.
        EXPECT_TRUE( reasoningInstruction( ReasoningEffort::Medium ).empty() );
        EXPECT_FALSE( reasoningInstruction( ReasoningEffort::Low ).empty() );
        EXPECT_FALSE( reasoningInstruction( ReasoningEffort::Xhigh ).empty() );
    }

    TEST( QwenProtocolReasoning, ScaleEndsMapToTheEnds )
    {
        EXPECT_EQ( reasoningEffortFromScale( 1 ), ReasoningEffort::Low );
        EXPECT_EQ( reasoningEffortFromScale( 2 ), ReasoningEffort::Low );
        EXPECT_EQ( reasoningEffortFromScale( 3 ), ReasoningEffort::Medium );
        EXPECT_EQ( reasoningEffortFromScale( 4 ), ReasoningEffort::Xhigh );
        EXPECT_EQ( reasoningEffortFromScale( 5 ), ReasoningEffort::Xhigh );
    }

    // ---- formatPrompt --------------------------------------------------------

    TEST( QwenProtocolPrompt, MinimalTurnCarriesNoBos )
    {
        // add_bos_token is false. A leading marker the model was not trained with costs a
        // position and shifts every one after it.
        const std::vector<Conversation::Turn> history{ { Conversation::Role::User, "hello", {} } };

        EXPECT_EQ( formatPrompt( history, false ),
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n" );
    }

    TEST( QwenProtocolPrompt, ThinkingOnLeavesTheSpanOpen )
    {
        // The primer ends ON the opening marker, so the response carries a CLOSE with no OPEN.
        // A parser that assumed a balanced pair reads the whole answer as reasoning.
        const std::vector<Conversation::Turn> history{ { Conversation::Role::User, "hello", {} } };

        EXPECT_TRUE( formatPrompt( history, true ).ends_with( "assistant\n<think>\n" ) );
    }

    TEST( QwenProtocolPrompt, AssistantHistoryTurnCarriesAnEmptyReasoningSpan )
    {
        const std::vector<Conversation::Turn> history{
            { Conversation::Role::User, "2+2?", {} },
            { Conversation::Role::Assistant, "4", {} },
            { Conversation::Role::User, "and 3+3?", {} },
        };

        EXPECT_NE( formatPrompt( history, false ).find(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n4<|im_end|>\n" ), std::string::npos );
    }

    TEST( QwenProtocolPrompt, ToolResultIsAUserTurnCarryingAResponseSpan )
    {
        // Not a role of its own: rendering Tool by name would open a role the model was never
        // trained to read.
        const std::vector<Conversation::Turn> history{
            { Conversation::Role::User, "weather?", {} },
            { Conversation::Role::Tool, "sunny", {} },
        };

        EXPECT_NE( formatPrompt( history, false ).find(
            "<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n" ),
            std::string::npos );
    }

    TEST( QwenProtocolPrompt, SystemTurnIsAssembledInTheTemplatesOrder )
    {
        // Reasoning instruction, then tools, then the caller's own text. A caller that had
        // already concatenated them could not produce that order.
        const std::vector<Conversation::Turn> history{
            { Conversation::Role::System, "Be terse.", {} },
            { Conversation::Role::User, "hi", {} },
        };

        const std::string prompt = formatPrompt(
            history, true, ReasoningEffort::Low, R"({"name":"get_weather"})" );

        const auto instruction = prompt.find( "Reasoning effort is set to low" );
        const auto tools = prompt.find( "# Tools" );
        const auto configured = prompt.find( "Be terse." );

        ASSERT_NE( instruction, std::string::npos );
        ASSERT_NE( tools, std::string::npos );
        ASSERT_NE( configured, std::string::npos );
        EXPECT_LT( instruction, tools );
        EXPECT_LT( tools, configured );
    }

    TEST( QwenProtocolPrompt, EffortIsOmittedWhenThinkingIsOff )
    {
        // With the span closed the instruction would describe a channel that never opens.
        const std::vector<Conversation::Turn> history{ { Conversation::Role::User, "hi", {} } };

        EXPECT_EQ( formatPrompt( history, false, ReasoningEffort::Xhigh ).find(
            "Reasoning effort" ), std::string::npos );
    }

    TEST( QwenProtocolPrompt, NoToolsMeansNoToolsSection )
    {
        // The absence of the section is what tells the model there are none.
        const std::vector<Conversation::Turn> history{ { Conversation::Role::User, "hi", {} } };

        EXPECT_EQ( formatPrompt( history, false ).find( "# Tools" ), std::string::npos );
    }

    TEST( QwenProtocolPrompt, EmptyHistoryThrows )
    {
        EXPECT_THROW( (void)formatPrompt( std::vector<Conversation::Turn>{}, false ),
            std::invalid_argument );
    }

    TEST( QwenProtocolPrompt, TrailingAssistantTurnThrows )
    {
        // The primer would open a second consecutive assistant turn.
        const std::vector<Conversation::Turn> history{
            { Conversation::Role::User, "hi", {} },
            { Conversation::Role::Assistant, "hello", {} },
        };

        EXPECT_THROW( (void)formatPrompt( history, false ), std::invalid_argument );
    }

    // ---- serializeToolSignatures ---------------------------------------------

    TEST( QwenProtocolTools, SignaturesRenderOnePerLineWithNoArray )
    {
        // The template writes the objects newline separated inside <tools>, with no array
        // around them -- not the pretty-printed array a host is likely to be holding.
        EXPECT_EQ( serializeToolSignatures( R"([{"name":"a"},{"name":"b"}])" ),
            "{\"name\":\"a\"}\n{\"name\":\"b\"}" );
    }

    TEST( QwenProtocolTools, MalformedOrEmptyInputOmitsTheSection )
    {
        EXPECT_TRUE( serializeToolSignatures( "" ).empty() );
        EXPECT_TRUE( serializeToolSignatures( "[]" ).empty() );
        EXPECT_TRUE( serializeToolSignatures( "not json" ).empty() );
        EXPECT_TRUE( serializeToolSignatures( R"({"name":"a"})" ).empty() );
    }
}
