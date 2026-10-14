// Parity and contract tests for Gemma 4's chat template behind the Protocol interface
// (Dnn.Models.GemmaChatProtocol).
//
// The goldens are google/gemma-4-12b-it's own chat_template.jinja -- the file the 12B, its QAT build
// and the 26B-A4B all ship -- rendered by transformers' render_jinja_template, the renderer
// apply_chat_template calls. Recorded by Mila/Tools/ChatTemplates/record_template_goldens.py, whose
// cases these histories mirror one for one. Re-record rather than edit: a golden that disagrees with
// this renderer is a finding about the renderer, never about the template (ModelHandle.md 3.4).

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <string_view>
#include <vector>

import Mila;
import nlohmann.json;

namespace Mila::Tests::Dnn::Models
{
    namespace Conversation = Mila::Dnn::Conversation;

    using Mila::Dnn::Gemma::GemmaChatProtocol;

    namespace
    {
        // ---- Goldens: record_template_goldens.py gemma --------------------------
        constexpr std::string_view kPlain =
            "<bos><|turn>user\n"
            "What is the capital of Australia?<turn|>\n"
            "<|turn>model\n"
            "<|channel>thought\n"
            "<channel|>";

        constexpr std::string_view kSystemGolden =
            "<bos><|turn>system\n"
            "You are a concise assistant.<turn|>\n"
            "<|turn>user\n"
            "What is the capital of Australia?<turn|>\n"
            "<|turn>model\n"
            "<|channel>thought\n"
            "<channel|>";

        constexpr std::string_view kTrimmed =
            "<bos><|turn>system\n"
            "Be brief.<turn|>\n"
            "<|turn>user\n"
            "hi<turn|>\n"
            "<|turn>model\n"
            "<|channel>thought\n"
            "<channel|>";

        constexpr std::string_view kMultiTurn =
            "<bos><|turn>user\n"
            "Capital of Australia?<turn|>\n"
            "<|turn>model\n"
            "Canberra.<turn|>\n"
            "<|turn>user\n"
            "And of Canada?<turn|>\n"
            "<|turn>model\n"
            "<|channel>thought\n"
            "<channel|>";

        constexpr std::string_view kTools =
            "<bos><|turn>system\n"
            "You are a concise assistant.<|tool>declaration:get_weather{description:<|\"|>Get the weather for a city<|\"|>,parameters:{properties:{city:{description:<|\"|>The city<|\"|>,type:<|\"|>STRING<|\"|>},units:{enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><|tool>declaration:get_time{description:<|\"|>Get the local time in a city<|\"|>,parameters:{properties:{city:{type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n"
            "<|turn>user\n"
            "What is the weather in Paris?<turn|>\n"
            "<|turn>model\n"
            "<|channel>thought\n"
            "<channel|>";

        constexpr std::string_view kToolsNoSystem =
            "<bos><|turn>system\n"
            "<|tool>declaration:get_weather{description:<|\"|>Get the weather for a city<|\"|>,parameters:{properties:{city:{description:<|\"|>The city<|\"|>,type:<|\"|>STRING<|\"|>},units:{enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n"
            "<|turn>user\n"
            "What is the weather in Paris?<turn|>\n"
            "<|turn>model\n"
            "<|channel>thought\n"
            "<channel|>";

        constexpr std::string_view kOneCallResume =
            "<bos><|turn>system\n"
            "You are a concise assistant.<|tool>declaration:get_weather{description:<|\"|>Get the weather for a city<|\"|>,parameters:{properties:{city:{description:<|\"|>The city<|\"|>,type:<|\"|>STRING<|\"|>},units:{enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n"
            "<|turn>user\n"
            "What is the weather in Paris?<turn|>\n"
            "<|turn>model\n"
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_response>response:get_weather{value:<|\"|>Sunny, 21 C<|\"|>}<tool_response|>";

        constexpr std::string_view kTwoCallsResume =
            "<bos><|turn>system\n"
            "<|tool>declaration:get_weather{description:<|\"|>Get the weather for a city<|\"|>,parameters:{properties:{city:{description:<|\"|>The city<|\"|>,type:<|\"|>STRING<|\"|>},units:{enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><|tool>declaration:get_time{description:<|\"|>Get the local time in a city<|\"|>,parameters:{properties:{city:{type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n"
            "<|turn>user\n"
            "Weather and time in Paris?<turn|>\n"
            "<|turn>model\n"
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_call>call:get_time{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_response>response:get_weather{value:<|\"|>Sunny, 21 C<|\"|>}<tool_response|><|tool_response>response:get_time{value:<|\"|>14:05<|\"|>}<tool_response|>Let me check both.<turn|>\n";

        constexpr std::string_view kTwoCallsOutOfOrder =
            "<bos><|turn>system\n"
            "<|tool>declaration:get_weather{description:<|\"|>Get the weather for a city<|\"|>,parameters:{properties:{city:{description:<|\"|>The city<|\"|>,type:<|\"|>STRING<|\"|>},units:{enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><|tool>declaration:get_time{description:<|\"|>Get the local time in a city<|\"|>,parameters:{properties:{city:{type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n"
            "<|turn>user\n"
            "Weather and time in Paris?<turn|>\n"
            "<|turn>model\n"
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_call>call:get_time{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_response>response:get_time{value:<|\"|>14:05<|\"|>}<tool_response|><|tool_response>response:get_weather{value:<|\"|>Sunny, 21 C<|\"|>}<tool_response|>";

        constexpr std::string_view kClosedCycle =
            "<bos><|turn>system\n"
            "<|tool>declaration:get_weather{description:<|\"|>Get the weather for a city<|\"|>,parameters:{properties:{city:{description:<|\"|>The city<|\"|>,type:<|\"|>STRING<|\"|>},units:{enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n"
            "<|turn>user\n"
            "What is the weather in Paris?<turn|>\n"
            "<|turn>model\n"
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_response>response:get_weather{value:<|\"|>Sunny, 21 C<|\"|>}<tool_response|>It is sunny and 21 C.<turn|>\n"
            "<|turn>user\n"
            "And tomorrow?<turn|>\n"
            "<|turn>model\n"
            "<|channel>thought\n"
            "<channel|>";

        constexpr std::string_view kTypedArguments =
            "<bos><|turn>user\n"
            "Configure it.<turn|>\n"
            "<|turn>model\n"
            "<|tool_call>call:configure{enabled:true,mode:<|\"|>high<|\"|>,opts:{a:[1,2],z:1},tokens:4096}<tool_call|><|tool_response>response:configure{value:<|\"|>ok<|\"|>}<tool_response|>";

        constexpr std::string_view kJsonResult =
            "<bos><|turn>system\n"
            "<|tool>declaration:get_weather{description:<|\"|>Get the weather for a city<|\"|>,parameters:{properties:{city:{description:<|\"|>The city<|\"|>,type:<|\"|>STRING<|\"|>},units:{enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n"
            "<|turn>user\n"
            "What is the weather in Paris?<turn|>\n"
            "<|turn>model\n"
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_response>response:get_weather{value:<|\"|>{\"output\": \"Sunny\", \"chunk_id\": 7}<|\"|>}<tool_response|>";

        constexpr std::string_view kThinkingOn =
            "<bos><|turn>system\n"
            "<|think|>\n"
            "You are a concise assistant.<turn|>\n"
            "<|turn>user\n"
            "What is the capital of Australia?<turn|>\n"
            "<|turn>model\n";

        constexpr std::string_view kThinkingOnNoSystem =
            "<bos><|turn>system\n"
            "<|think|>\n"
            "<turn|>\n"
            "<|turn>user\n"
            "What is the capital of Australia?<turn|>\n"
            "<|turn>model\n";

        constexpr std::string_view kToolsThinkingOn =
            "<bos><|turn>system\n"
            "<|think|>\n"
            "You are a concise assistant.<|tool>declaration:get_weather{description:<|\"|>Get the weather for a city<|\"|>,parameters:{properties:{city:{description:<|\"|>The city<|\"|>,type:<|\"|>STRING<|\"|>},units:{enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><|tool>declaration:get_time{description:<|\"|>Get the local time in a city<|\"|>,parameters:{properties:{city:{type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n"
            "<|turn>user\n"
            "What is the weather in Paris?<turn|>\n"
            "<|turn>model\n";

        constexpr std::string_view kOneCallResumeThinkingOn =
            "<bos><|turn>system\n"
            "<|think|>\n"
            "You are a concise assistant.<|tool>declaration:get_weather{description:<|\"|>Get the weather for a city<|\"|>,parameters:{properties:{city:{description:<|\"|>The city<|\"|>,type:<|\"|>STRING<|\"|>},units:{enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n"
            "<|turn>user\n"
            "What is the weather in Paris?<turn|>\n"
            "<|turn>model\n"
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_response>response:get_weather{value:<|\"|>Sunny, 21 C<|\"|>}<tool_response|><|channel>thought\n";

        constexpr std::string_view kClosedCyclePreserveOn =
            "<bos><|turn>system\n"
            "<|tool>declaration:get_weather{description:<|\"|>Get the weather for a city<|\"|>,parameters:{properties:{city:{description:<|\"|>The city<|\"|>,type:<|\"|>STRING<|\"|>},units:{enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n"
            "<|turn>user\n"
            "What is the weather in Paris?<turn|>\n"
            "<|turn>model\n"
            "<|channel>thought\n"
            "I need the weather tool.\n"
            "<channel|><|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_response>response:get_weather{value:<|\"|>Sunny, 21 C<|\"|>}<tool_response|>It is sunny and 21 C.<turn|>\n"
            "<|turn>user\n"
            "And tomorrow?<turn|>\n"
            "<|turn>model\n"
            "<|channel>thought\n"
            "<channel|>";

        // ---- The recorder's cases ------------------------------------------------

        const std::string kWeather = R"({"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city"}, "units": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}})";
        const std::string kTime = R"({"type": "function", "function": {"name": "get_time", "description": "Get the local time in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})";

        std::string tools( std::initializer_list<std::string> schemas )
        {
            std::string json = "[";

            for ( const auto& schema : schemas )
            {
                json += json.size() == 1 ? "" : ",";
                json += schema;
            }

            return json + "]";
        }

        Conversation::Turn system( std::string content )
        {
            return { Conversation::Role::System, std::move( content ) };
        }

        Conversation::Turn user( std::string content )
        {
            return { Conversation::Role::User, std::move( content ) };
        }

        Conversation::Turn assistant( std::string content, std::vector<Conversation::ToolCall> calls = {}, std::string reasoning = {} )
        {
            return { Conversation::Role::Assistant, std::move( content ), std::move( calls ), {}, std::move( reasoning ) };
        }

        Conversation::Turn result( std::string name, std::string content )
        {
            return { Conversation::Role::Tool, std::move( content ), {}, std::move( name ) };
        }

        const Conversation::ToolCall kWeatherCall{ "get_weather", R"({"city": "Paris"})" };
        const Conversation::ToolCall kTimeCall{ "get_time", R"({"city": "Paris"})" };

        const std::string kSystem = "You are a concise assistant.";
        const std::string kCapital = "What is the capital of Australia?";
        const std::string kAsk = "What is the weather in Paris?";
        const std::string kAskBoth = "Weather and time in Paris?";

        std::vector<Conversation::Turn> oneCallResume()
        {
            return { system( kSystem ), user( kAsk ), assistant( "", { kWeatherCall } ), result( "get_weather", "Sunny, 21 C" ) };
        }

        std::vector<Conversation::Turn> closedCycle()
        {
            return {
                user( kAsk ),
                assistant( "", { kWeatherCall }, "I need the weather tool." ),
                result( "get_weather", "Sunny, 21 C" ),
                assistant( "It is sunny and 21 C.", {}, "The tool said sunny." ),
                user( "And tomorrow?" ) };
        }

        Conversation::PromptOptions options( std::string tools_json = {}, std::map<std::string, std::string> variables = {} )
        {
            return { std::move( variables ), std::move( tools_json ) };
        }

        std::string render( const std::vector<Conversation::Turn>& history, const Conversation::PromptOptions& prompt_options = {} )
        {
            return GemmaChatProtocol{}.formatPrompt( history, prompt_options ).text;
        }
    }

    // ---- P1: byte-identical to the template ------------------------------------

    TEST( GemmaChatProtocolParity, Plain )
    {
        EXPECT_EQ( render( { user( kCapital ) } ), kPlain );
    }

    TEST( GemmaChatProtocolParity, System )
    {
        EXPECT_EQ( render( { system( kSystem ), user( kCapital ) } ), kSystemGolden );
    }

    TEST( GemmaChatProtocolParity, ContentIsTrimmed )
    {
        EXPECT_EQ( render( { system( "  Be brief.\n" ), user( "  hi  \n" ) } ), kTrimmed );
    }

    TEST( GemmaChatProtocolParity, MultiTurn )
    {
        EXPECT_EQ( render( { user( "Capital of Australia?" ), assistant( "Canberra." ), user( "And of Canada?" ) } ), kMultiTurn );
    }

    TEST( GemmaChatProtocolParity, Tools )
    {
        EXPECT_EQ( render( { system( kSystem ), user( kAsk ) }, options( tools( { kWeather, kTime } ) ) ), kTools );
    }

    TEST( GemmaChatProtocolParity, ToolsWithNoSystemTurn )
    {
        EXPECT_EQ( render( { user( kAsk ) }, options( tools( { kWeather } ) ) ), kToolsNoSystem );
    }

    TEST( GemmaChatProtocolParity, OneCallResumeLeavesTheModelTurnOpen )
    {
        EXPECT_EQ( render( oneCallResume(), options( tools( { kWeather } ) ) ), kOneCallResume );
    }

    TEST( GemmaChatProtocolParity, ContentOfACallingTurnFollowsItsResults )
    {
        // The template writes a turn's content after its calls and results, and then closes the turn
        // with no prompt after it. Rendered as the template renders it.
        EXPECT_EQ( render( {
            user( kAskBoth ),
            assistant( "Let me check both.", { kWeatherCall, kTimeCall } ),
            result( "get_weather", "Sunny, 21 C" ),
            result( "get_time", "14:05" ) }, options( tools( { kWeather, kTime } ) ) ), kTwoCallsResume );
    }

    TEST( GemmaChatProtocolParity, ResultsRenderInTheOrderGivenUnderTheirOwnNames )
    {
        EXPECT_EQ( render( {
            user( kAskBoth ),
            assistant( "", { kWeatherCall, kTimeCall } ),
            result( "get_time", "14:05" ),
            result( "get_weather", "Sunny, 21 C" ) }, options( tools( { kWeather, kTime } ) ) ), kTwoCallsOutOfOrder );
    }

    TEST( GemmaChatProtocolParity, ClosedCycleIsOneModelTurn )
    {
        EXPECT_EQ( render( closedCycle(), options( tools( { kWeather } ) ) ), kClosedCycle );
    }

    TEST( GemmaChatProtocolParity, TypedArgumentsRenderSorted )
    {
        EXPECT_EQ( render( {
            user( "Configure it." ),
            assistant( "", { { "configure", R"({"tokens": 4096, "enabled": true, "mode": "high", "opts": {"z": 1, "a": [1, 2]}})" } } ),
            result( "configure", "ok" ) } ), kTypedArguments );
    }

    TEST( GemmaChatProtocolParity, JsonResultIsRenderedAsTextNotDistilled )
    {
        EXPECT_EQ( render( {
            user( kAsk ),
            assistant( "", { kWeatherCall } ),
            result( "get_weather", R"({"output": "Sunny", "chunk_id": 7})" ) }, options( tools( { kWeather } ) ) ), kJsonResult );
    }

    TEST( GemmaChatProtocolParity, ThinkingOn )
    {
        EXPECT_EQ( render( { system( kSystem ), user( kCapital ) }, options( {}, { { "enable_thinking", "true" } } ) ), kThinkingOn );
    }

    TEST( GemmaChatProtocolParity, ThinkingOnOpensASystemTurnOfItsOwn )
    {
        EXPECT_EQ( render( { user( kCapital ) }, options( {}, { { "enable_thinking", "true" } } ) ), kThinkingOnNoSystem );
    }

    TEST( GemmaChatProtocolParity, ToolsThinkingOn )
    {
        EXPECT_EQ( render( { system( kSystem ), user( kAsk ) },
            options( tools( { kWeather, kTime } ), { { "enable_thinking", "true" } } ) ), kToolsThinkingOn );
    }

    TEST( GemmaChatProtocolParity, OneCallResumeThinkingOnOpensTheChannel )
    {
        EXPECT_EQ( render( oneCallResume(), options( tools( { kWeather } ), { { "enable_thinking", "true" } } ) ), kOneCallResumeThinkingOn );
    }

    TEST( GemmaChatProtocolParity, PreserveThinkingKeepsTheReasoningOfACallingTurn )
    {
        EXPECT_EQ( render( closedCycle(), options( tools( { kWeather } ), { { "preserve_thinking", "true" } } ) ), kClosedCyclePreserveOn );
    }

    // ---- Template variables -------------------------------------------------------

    TEST( GemmaChatProtocolVariables, AreTheTemplatesOwnWithItsDefaults )
    {
        const GemmaChatProtocol protocol;
        std::map<std::string, std::string> defaults;

        for ( const auto& variable : protocol.templateVariables() )
        {
            defaults[ variable.name ] = variable.default_value;
        }

        EXPECT_EQ( defaults, ( std::map<std::string, std::string>{ { "enable_thinking", "false" }, { "preserve_thinking", "false" } } ) );
    }

    TEST( GemmaChatProtocolVariables, EffortIsNotAVariableOfThisTemplate )
    {
        // Gemma 4's thinking is a switch with no budget; an effort level would be text Mila invented.
        EXPECT_THROW( (void)render( { user( kCapital ) }, options( {}, { { "reasoning_effort", "low" } } ) ), std::invalid_argument );
    }

    TEST( GemmaChatProtocolVariables, TheReplyBeginsInReasoningOnlyOnAThinkingResume )
    {
        const GemmaChatProtocol protocol;
        const auto thinking = options( tools( { kWeather } ), { { "enable_thinking", "true" } } );

        EXPECT_TRUE( protocol.formatPrompt( oneCallResume(), thinking ).reply_begins_in_reasoning );
        EXPECT_FALSE( protocol.formatPrompt( oneCallResume(), options( tools( { kWeather } ) ) ).reply_begins_in_reasoning );
        EXPECT_FALSE( protocol.formatPrompt( std::vector{ user( kCapital ) }, thinking ).reply_begins_in_reasoning );
    }

    // ---- Histories refused rather than rendered as a prompt that looks valid --------

    TEST( GemmaChatProtocolRefusal, AResultWithNoCallBeforeItIsRefused )
    {
        // The template would drop it without a trace.
        EXPECT_THROW( (void)render( { user( kAsk ), result( "get_weather", "Sunny" ) } ), std::invalid_argument );
    }

    TEST( GemmaChatProtocolRefusal, AResultThatNamesNoToolIsRefused )
    {
        // The template would name it `unknown`.
        EXPECT_THROW( (void)render( { user( kAsk ), assistant( "", { kWeatherCall } ), result( "", "Sunny" ) } ), std::invalid_argument );
    }

    TEST( GemmaChatProtocolRefusal, ArgumentsThatAreNotAnObjectAreRefused )
    {
        EXPECT_THROW( (void)render( { user( kAsk ), assistant( "", { { "get_weather", "Paris" } } ) } ), std::invalid_argument );
    }

    // ---- parseReply ----------------------------------------------------------------

    namespace
    {
        Conversation::Reply parse( std::string_view text, bool begins_in_reasoning = false )
        {
            return GemmaChatProtocol{}.parseReply( text, { {}, begins_in_reasoning } );
        }
    }

    TEST( GemmaChatProtocolReply, EveryCallIsReturnedInOrder )
    {
        // The free parseToolCall keeps only the last of them.
        const auto reply = parse(
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_call>call:get_time{city:<|\"|>Paris<|\"|>}<tool_call|>" );

        ASSERT_EQ( reply.tool_calls.size(), 2u );
        EXPECT_EQ( reply.tool_calls[ 0 ].name, "get_weather" );
        EXPECT_EQ( reply.tool_calls[ 1 ].name, "get_time" );
        EXPECT_TRUE( reply.content.empty() );
    }

    TEST( GemmaChatProtocolReply, ChannelsAreReasoningWithoutTheirLabel )
    {
        const auto reply = parse( "<|channel>thought\nThe capital is Canberra.<channel|>Canberra." );

        EXPECT_EQ( reply.reasoning, "The capital is Canberra." );
        EXPECT_EQ( reply.content, "Canberra." );
    }

    TEST( GemmaChatProtocolReply, AReplyThatBeganInsideTheChannelSplitsAtItsClose )
    {
        const auto reply = parse( "The tool said sunny.<channel|>It is sunny.", true );

        EXPECT_EQ( reply.reasoning, "The tool said sunny." );
        EXPECT_EQ( reply.content, "It is sunny." );
    }

    TEST( GemmaChatProtocolReply, AMalformedCallIsReported )
    {
        const auto reply = parse( "Sure.<|tool_call>get_weather{city:1}<tool_call|>" );

        ASSERT_EQ( reply.malformed_tool_calls.size(), 1u );
        EXPECT_EQ( reply.malformed_tool_calls[ 0 ], "<|tool_call>get_weather{city:1}<tool_call|>" );
        EXPECT_TRUE( reply.tool_calls.empty() );
        EXPECT_EQ( reply.content, "Sure." );
    }

    TEST( GemmaChatProtocolReply, OffGrammarArgumentsAreReportedNotDispatchedEmpty )
    {
        const std::string span = R"(<|tool_call>call:exec_command{cmd="cat line_count.txt"}<tool_call|>)";
        const auto reply = parse( span );

        EXPECT_TRUE( reply.tool_calls.empty() );
        ASSERT_EQ( reply.malformed_tool_calls.size(), 1u );
        EXPECT_EQ( reply.malformed_tool_calls[ 0 ], span );
    }

    TEST( GemmaChatProtocolReply, TheOffSpecUnclosedFormIsReportedAsMalformed )
    {
        // The 12B's `call:name:key=value` with no closing marker.
        const auto reply = parse( "<|tool_call>call:run:cmd=ls" );

        ASSERT_EQ( reply.malformed_tool_calls.size(), 1u );
        EXPECT_EQ( reply.malformed_tool_calls[ 0 ], "<|tool_call>call:run:cmd=ls" );
        EXPECT_TRUE( reply.content.empty() );
    }

    TEST( GemmaChatProtocolReply, AParsedCallRendersBackAsTheTemplateWritesIt )
    {
        const auto reply = parse( "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>" );

        ASSERT_EQ( reply.tool_calls.size(), 1u );

        auto history = oneCallResume();
        history[ 2 ].tool_calls = reply.tool_calls;

        EXPECT_EQ( render( history, options( tools( { kWeather } ) ) ), kOneCallResume );
    }

    TEST( GemmaChatProtocolResult, IsTheTemplatesResponseBlockForText )
    {
        EXPECT_EQ( GemmaChatProtocol{}.formatToolResult( "get_weather", "Sunny, 21 C" ),
            "<|tool_response>response:get_weather{value:<|\"|>Sunny, 21 C<|\"|>}<tool_response|>" );
    }
}
