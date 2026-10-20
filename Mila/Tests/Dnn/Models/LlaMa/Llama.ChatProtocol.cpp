// Parity and contract tests for Meta's Llama 3.1 and 3.2 chat templates behind the Protocol interface
// (Dnn.Models.LlamaChatProtocol).
//
// The goldens are the chat_template in meta-llama/Llama-3.1-8B-Instruct's tokenizer_config.json rendered by
// transformers' render_jinja_template, the renderer apply_chat_template calls, and recorded by
// Mila/Tools/ChatTemplates/record_template_goldens.py, whose cases these histories mirror one for one. With
// date_string fixed, Llama-3.2-3B-Instruct's template renders every case to the same bytes (record llama32 and
// compare), so both protocols are held to one set. Re-record rather than edit: a golden that disagrees with
// this renderer is a finding about the renderer, never about the template (ModelHandle.md 3.4).

#include <gtest/gtest.h>
#include <chrono>
#include <format>
#include <map>
#include <string>
#include <string_view>
#include <vector>

import Mila;
import nlohmann.json;

namespace Mila::Tests::Dnn::Models
{
    namespace Conversation = Mila::Dnn::Conversation;

    using Mila::Dnn::Llama::LlamaChatProtocol;

    namespace
    {
        // ---- Goldens: record_template_goldens.py llama31 ------------------------
        constexpr std::string_view kPlain =
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n"
            "\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "What is the capital of Australia?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n";

        constexpr std::string_view kSystemGolden =
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n"
            "\n"
            "You are a concise assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "What is the capital of Australia?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n";

        constexpr std::string_view kTrimmed =
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n"
            "\n"
            "Be brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n";

        constexpr std::string_view kMultiTurn =
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n"
            "\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "Capital of Australia?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n"
            "Canberra.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "And of Canada?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n";

        constexpr std::string_view kTools =
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "Environment: ipython\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n"
            "\n"
            "You are a concise assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n"
            "\n"
            "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n"
            "\n"
            "{\n"
            "    \"type\": \"function\",\n"
            "    \"function\": {\n"
            "        \"name\": \"get_weather\",\n"
            "        \"description\": \"Get the weather for a city\",\n"
            "        \"parameters\": {\n"
            "            \"type\": \"object\",\n"
            "            \"properties\": {\n"
            "                \"city\": {\n"
            "                    \"type\": \"string\",\n"
            "                    \"description\": \"The city\"\n"
            "                },\n"
            "                \"units\": {\n"
            "                    \"type\": \"string\",\n"
            "                    \"enum\": [\n"
            "                        \"c\",\n"
            "                        \"f\"\n"
            "                    ]\n"
            "                }\n"
            "            },\n"
            "            \"required\": [\n"
            "                \"city\"\n"
            "            ]\n"
            "        }\n"
            "    }\n"
            "}\n"
            "\n"
            "{\n"
            "    \"type\": \"function\",\n"
            "    \"function\": {\n"
            "        \"name\": \"get_time\",\n"
            "        \"description\": \"Get the local time in a city\",\n"
            "        \"parameters\": {\n"
            "            \"type\": \"object\",\n"
            "            \"properties\": {\n"
            "                \"city\": {\n"
            "                    \"type\": \"string\"\n"
            "                }\n"
            "            },\n"
            "            \"required\": [\n"
            "                \"city\"\n"
            "            ]\n"
            "        }\n"
            "    }\n"
            "}\n"
            "\n"
            "What is the weather in Paris?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n";

        constexpr std::string_view kToolsNoSystem =
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "Environment: ipython\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n"
            "\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n"
            "\n"
            "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n"
            "\n"
            "{\n"
            "    \"type\": \"function\",\n"
            "    \"function\": {\n"
            "        \"name\": \"get_weather\",\n"
            "        \"description\": \"Get the weather for a city\",\n"
            "        \"parameters\": {\n"
            "            \"type\": \"object\",\n"
            "            \"properties\": {\n"
            "                \"city\": {\n"
            "                    \"type\": \"string\",\n"
            "                    \"description\": \"The city\"\n"
            "                },\n"
            "                \"units\": {\n"
            "                    \"type\": \"string\",\n"
            "                    \"enum\": [\n"
            "                        \"c\",\n"
            "                        \"f\"\n"
            "                    ]\n"
            "                }\n"
            "            },\n"
            "            \"required\": [\n"
            "                \"city\"\n"
            "            ]\n"
            "        }\n"
            "    }\n"
            "}\n"
            "\n"
            "What is the weather in Paris?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n";

        constexpr std::string_view kOneCallResume =
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "Environment: ipython\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n"
            "\n"
            "You are a concise assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n"
            "\n"
            "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n"
            "\n"
            "{\n"
            "    \"type\": \"function\",\n"
            "    \"function\": {\n"
            "        \"name\": \"get_weather\",\n"
            "        \"description\": \"Get the weather for a city\",\n"
            "        \"parameters\": {\n"
            "            \"type\": \"object\",\n"
            "            \"properties\": {\n"
            "                \"city\": {\n"
            "                    \"type\": \"string\",\n"
            "                    \"description\": \"The city\"\n"
            "                },\n"
            "                \"units\": {\n"
            "                    \"type\": \"string\",\n"
            "                    \"enum\": [\n"
            "                        \"c\",\n"
            "                        \"f\"\n"
            "                    ]\n"
            "                }\n"
            "            },\n"
            "            \"required\": [\n"
            "                \"city\"\n"
            "            ]\n"
            "        }\n"
            "    }\n"
            "}\n"
            "\n"
            "What is the weather in Paris?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n"
            "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n"
            "\n"
            "\"Sunny, 21 C\"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n";

        constexpr std::string_view kClosedCycle =
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "Environment: ipython\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n"
            "\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n"
            "\n"
            "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n"
            "\n"
            "{\n"
            "    \"type\": \"function\",\n"
            "    \"function\": {\n"
            "        \"name\": \"get_weather\",\n"
            "        \"description\": \"Get the weather for a city\",\n"
            "        \"parameters\": {\n"
            "            \"type\": \"object\",\n"
            "            \"properties\": {\n"
            "                \"city\": {\n"
            "                    \"type\": \"string\",\n"
            "                    \"description\": \"The city\"\n"
            "                },\n"
            "                \"units\": {\n"
            "                    \"type\": \"string\",\n"
            "                    \"enum\": [\n"
            "                        \"c\",\n"
            "                        \"f\"\n"
            "                    ]\n"
            "                }\n"
            "            },\n"
            "            \"required\": [\n"
            "                \"city\"\n"
            "            ]\n"
            "        }\n"
            "    }\n"
            "}\n"
            "\n"
            "What is the weather in Paris?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n"
            "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n"
            "\n"
            "\"Sunny, 21 C\"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n"
            "It is sunny and 21 C.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "And tomorrow?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n";

        constexpr std::string_view kTypedArguments =
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n"
            "\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "Configure it.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n"
            "{\"name\": \"configure\", \"parameters\": {\"tokens\": 4096, \"enabled\": true, \"mode\": \"high\", \"opts\": {\"z\": 1, \"a\": [1, 2]}}}<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n"
            "\n"
            "\"ok\"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n";

        constexpr std::string_view kJsonResult =
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "Environment: ipython\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n"
            "\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n"
            "\n"
            "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n"
            "\n"
            "{\n"
            "    \"type\": \"function\",\n"
            "    \"function\": {\n"
            "        \"name\": \"get_weather\",\n"
            "        \"description\": \"Get the weather for a city\",\n"
            "        \"parameters\": {\n"
            "            \"type\": \"object\",\n"
            "            \"properties\": {\n"
            "                \"city\": {\n"
            "                    \"type\": \"string\",\n"
            "                    \"description\": \"The city\"\n"
            "                },\n"
            "                \"units\": {\n"
            "                    \"type\": \"string\",\n"
            "                    \"enum\": [\n"
            "                        \"c\",\n"
            "                        \"f\"\n"
            "                    ]\n"
            "                }\n"
            "            },\n"
            "            \"required\": [\n"
            "                \"city\"\n"
            "            ]\n"
            "        }\n"
            "    }\n"
            "}\n"
            "\n"
            "What is the weather in Paris?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n"
            "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n"
            "\n"
            "\"{\\\"output\\\": \\\"Sunny\\\", \\\"chunk_id\\\": 7}\"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n";

        constexpr std::string_view kToolsInSystem =
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "Environment: ipython\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n"
            "\n"
            "You have access to the following functions. To call a function, please respond with JSON for a function call.Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n"
            "\n"
            "{\n"
            "    \"type\": \"function\",\n"
            "    \"function\": {\n"
            "        \"name\": \"get_weather\",\n"
            "        \"description\": \"Get the weather for a city\",\n"
            "        \"parameters\": {\n"
            "            \"type\": \"object\",\n"
            "            \"properties\": {\n"
            "                \"city\": {\n"
            "                    \"type\": \"string\",\n"
            "                    \"description\": \"The city\"\n"
            "                },\n"
            "                \"units\": {\n"
            "                    \"type\": \"string\",\n"
            "                    \"enum\": [\n"
            "                        \"c\",\n"
            "                        \"f\"\n"
            "                    ]\n"
            "                }\n"
            "            },\n"
            "            \"required\": [\n"
            "                \"city\"\n"
            "            ]\n"
            "        }\n"
            "    }\n"
            "}\n"
            "\n"
            "{\n"
            "    \"type\": \"function\",\n"
            "    \"function\": {\n"
            "        \"name\": \"get_time\",\n"
            "        \"description\": \"Get the local time in a city\",\n"
            "        \"parameters\": {\n"
            "            \"type\": \"object\",\n"
            "            \"properties\": {\n"
            "                \"city\": {\n"
            "                    \"type\": \"string\"\n"
            "                }\n"
            "            },\n"
            "            \"required\": [\n"
            "                \"city\"\n"
            "            ]\n"
            "        }\n"
            "    }\n"
            "}\n"
            "\n"
            "You are a concise assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "What is the weather in Paris?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n";

        constexpr std::string_view kDated =
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 14 Oct 2026\n"
            "\n"
            "You are a concise assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "\n"
            "What is the capital of Australia?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "\n";

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

        /// Every case through both templates: 3.1 at its default date, 3.2 with the date the recorder fixed.
        void expectGolden( const std::vector<Conversation::Turn>& history, Conversation::PromptOptions prompt_options, std::string_view golden )
        {
            EXPECT_EQ( LlamaChatProtocol::llama31().formatPrompt( history, prompt_options ).text, golden ) << "Llama 3.1";

            prompt_options.template_variables.try_emplace( "date_string", "26 Jul 2024" );
            EXPECT_EQ( LlamaChatProtocol::llama32().formatPrompt( history, prompt_options ).text, golden ) << "Llama 3.2";
        }
    }

    // ---- P1: byte-identical to the template ------------------------------------

    TEST( LlamaChatProtocolParity, Plain )
    {
        expectGolden( { user( kCapital ) }, {}, kPlain );
    }

    TEST( LlamaChatProtocolParity, System )
    {
        expectGolden( { system( kSystem ), user( kCapital ) }, {}, kSystemGolden );
    }

    TEST( LlamaChatProtocol, ASystemTurnAfterAReplyRendersWhereItStands )
    {
        const std::vector<Conversation::Turn> history{ user( kCapital ), assistant( "Canberra." ), system( kSystem ), user( kCapital ) };

        for ( const auto& protocol : { LlamaChatProtocol::llama31(), LlamaChatProtocol::llama32() } )
        {
            const auto prompt = protocol.formatPrompt( history, {} ).text;

            EXPECT_NE( prompt.find( "Canberra.<|eot_id|><|start_header_id|>system<|end_header_id|>\n\n" + kSystem ),
                std::string::npos );
            EXPECT_TRUE( protocol.rendersLaterSystemTurns() );
        }
    }

    TEST( LlamaChatProtocolParity, ContentIsTrimmed )
    {
        expectGolden( { system( "  Be brief.\n" ), user( "  hi  \n" ) }, {}, kTrimmed );
    }

    TEST( LlamaChatProtocolParity, MultiTurn )
    {
        expectGolden( { user( "Capital of Australia?" ), assistant( "Canberra." ), user( "And of Canada?" ) }, {}, kMultiTurn );
    }

    TEST( LlamaChatProtocolParity, ToolsGoInTheFirstUserMessageIndented )
    {
        expectGolden( { system( kSystem ), user( kAsk ) }, options( tools( { kWeather, kTime } ) ), kTools );
    }

    TEST( LlamaChatProtocolParity, ToolsWithNoSystemTurn )
    {
        expectGolden( { user( kAsk ) }, options( tools( { kWeather } ) ), kToolsNoSystem );
    }

    TEST( LlamaChatProtocolParity, OneCallResumeQuotesTheResult )
    {
        expectGolden( { system( kSystem ), user( kAsk ), assistant( "", { kWeatherCall } ), result( "get_weather", "Sunny, 21 C" ) },
            options( tools( { kWeather } ) ), kOneCallResume );
    }

    TEST( LlamaChatProtocolParity, ClosedCycleDropsReasoning )
    {
        expectGolden( closedCycle(), options( tools( { kWeather } ) ), kClosedCycle );
    }

    TEST( LlamaChatProtocolParity, TypedArgumentsKeepTheirOrderThroughTojson )
    {
        expectGolden( {
            user( "Configure it." ),
            assistant( "", { { "configure", R"({"tokens": 4096, "enabled": true, "mode": "high", "opts": {"z": 1, "a": [1, 2]}})" } } ),
            result( "configure", "ok" ) }, {}, kTypedArguments );
    }

    TEST( LlamaChatProtocolParity, JsonResultIsQuotedAsText )
    {
        expectGolden( { user( kAsk ), assistant( "", { kWeatherCall } ), result( "get_weather", R"({"output": "Sunny", "chunk_id": 7})" ) },
            options( tools( { kWeather } ) ), kJsonResult );
    }

    TEST( LlamaChatProtocolParity, ToolsInTheSystemMessage )
    {
        expectGolden( { system( kSystem ), user( kAsk ) },
            options( tools( { kWeather, kTime } ), { { "tools_in_user_message", "false" } } ), kToolsInSystem );
    }

    TEST( LlamaChatProtocolParity, Dated )
    {
        expectGolden( { system( kSystem ), user( kCapital ) }, options( {}, { { "date_string", "14 Oct 2026" } } ), kDated );
    }

    // ---- Template variables -------------------------------------------------------

    TEST( LlamaChatProtocolVariables, TheDateDefaultsAsEachTemplateDefaultsIt )
    {
        const std::vector<Conversation::Turn> history{ user( kCapital ) };
        const auto now = std::chrono::current_zone()->to_local( std::chrono::system_clock::now() );
        const std::string today = std::format( "{:%d %b %Y}", std::chrono::floor<std::chrono::days>( now ) );

        EXPECT_NE( LlamaChatProtocol::llama31().formatPrompt( history, {} ).text.find( "Today Date: 26 Jul 2024\n" ), std::string::npos );
        EXPECT_NE( LlamaChatProtocol::llama32().formatPrompt( history, {} ).text.find( "Today Date: " + today + "\n" ), std::string::npos );
    }

    TEST( LlamaChatProtocolVariables, BuiltInToolsAreNotTaken )
    {
        EXPECT_THROW( (void)LlamaChatProtocol::llama31().formatPrompt( std::vector{ user( kCapital ) },
            options( {}, { { "builtin_tools", "brave_search" } } ) ), std::invalid_argument );
    }

    TEST( LlamaChatProtocolVariables, ThinkingIsNotAVariableOfThisTemplate )
    {
        EXPECT_THROW( (void)LlamaChatProtocol::llama32().formatPrompt( std::vector{ user( kCapital ) },
            options( {}, { { "enable_thinking", "true" } } ) ), std::invalid_argument );
    }

    // ---- Histories the template refuses -------------------------------------------

    TEST( LlamaChatProtocolRefusal, TwoCallsInOneTurnAreRefused )
    {
        // The recorder's two_calls_resume and two_calls_out_of_order: "This model only supports single tool-calls at once!"
        EXPECT_THROW( (void)LlamaChatProtocol::llama31().formatPrompt( std::vector{
            user( kAskBoth ), assistant( "", { kWeatherCall, kTimeCall } ), result( "get_weather", "Sunny" ), result( "get_time", "14:05" ) },
            options( tools( { kWeather, kTime } ) ) ), std::invalid_argument );
    }

    TEST( LlamaChatProtocolRefusal, ToolsWithNoMessageToCarryThemAreRefused )
    {
        EXPECT_THROW( (void)LlamaChatProtocol::llama31().formatPrompt( std::vector{ system( kSystem ) }, options( tools( { kWeather } ) ) ), std::invalid_argument );
    }

    // ---- parseReply ----------------------------------------------------------------

    namespace
    {
        Conversation::Reply parse( std::string_view text )
        {
            return LlamaChatProtocol::llama31().parseReply( text, {} );
        }
    }

    TEST( LlamaChatProtocolReply, ProseIsContent )
    {
        const auto reply = parse( "Canberra is the capital." );

        EXPECT_EQ( reply.content, "Canberra is the capital." );
        EXPECT_TRUE( reply.tool_calls.empty() );
        EXPECT_TRUE( reply.malformed_tool_calls.empty() );
    }

    TEST( LlamaChatProtocolReply, TheTemplatesCallFormatIsACall )
    {
        const auto reply = parse( R"({"name": "get_weather", "parameters": {"city": "Paris"}})" );

        ASSERT_EQ( reply.tool_calls.size(), 1u );
        EXPECT_EQ( reply.tool_calls[ 0 ].name, "get_weather" );
        EXPECT_EQ( nlohmann::json::parse( reply.tool_calls[ 0 ].arguments ).at( "city" ), "Paris" );
        EXPECT_TRUE( reply.content.empty() );
    }

    TEST( LlamaChatProtocolReply, ThePythonTagAndArgumentsKeyAreRead )
    {
        const auto reply = parse( R"(Checking.<|python_tag|>{"name": "get_weather", "arguments": {"city": "Paris"}}<|eom_id|>)" );

        ASSERT_EQ( reply.tool_calls.size(), 1u );
        EXPECT_EQ( reply.tool_calls[ 0 ].name, "get_weather" );
        EXPECT_EQ( reply.content, "Checking." );
    }

    TEST( LlamaChatProtocolReply, AnObjectThatIsNotACallIsReported )
    {
        const auto reply = parse( R"({"name": "get_weather", "parameters": "Paris"})" );

        EXPECT_TRUE( reply.tool_calls.empty() );
        ASSERT_EQ( reply.malformed_tool_calls.size(), 1u );
        EXPECT_EQ( reply.malformed_tool_calls[ 0 ], R"({"name": "get_weather", "parameters": "Paris"})" );
    }

    TEST( LlamaChatProtocolReply, AParsedCallRendersBackAsTheTemplateWritesIt )
    {
        const auto reply = parse( R"({"name": "get_weather", "parameters": {"city": "Paris"}})" );

        ASSERT_EQ( reply.tool_calls.size(), 1u );
        expectGolden( { system( kSystem ), user( kAsk ), assistant( "", reply.tool_calls ), result( "get_weather", "Sunny, 21 C" ) },
            options( tools( { kWeather } ) ), kOneCallResume );
    }

    TEST( LlamaChatProtocolResult, IsJsonQuoted )
    {
        EXPECT_EQ( LlamaChatProtocol::llama31().formatToolResult( "get_weather", "Sunny, \"21\" C" ), R"("Sunny, \"21\" C")" );
    }
}
