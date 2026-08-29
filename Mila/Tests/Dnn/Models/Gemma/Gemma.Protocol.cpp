// Unit tests for the canonical Gemma 4 grammar module (Dnn.Models.GemmaProtocol).
// This is the union-of-both-implementations coverage the drought never had: it pins the
// spec-verified behaviors the runtime module now owns -- the <|"|> string delimiter on both
// parse and format, integer-preserving argument coercion, namespace stripping, and the
// tool-response output-field distillation with failed-tool error surfacing.

#include <gtest/gtest.h>
#include <string>
#include <optional>
#include <vector>

import Mila;
import nlohmann.json;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn::Gemma;

    // The family-neutral conversation the template renders. Aliased rather than a using
    // directive: this namespace is itself called Dnn, so an unqualified one is ambiguous.
    namespace Conversation = Mila::Dnn::Conversation;

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

    // ---- Argument-order parity with gemma_protocol.py -----------------------

    // The golden strings below are asserted verbatim by the Python twin in
    // Mila/Adaptors/Inference/Server/tests/test_gemma_protocol.py -- the two suites
    // agreeing on the same literals is what makes "parity" mean something. Change a
    // golden here and the Python counterpart must change in the same commit.
    //
    // Keys render SORTED. This side got it for free (nlohmann::json is a std::map);
    // Python preserved insertion order and so emitted a different prompt for identical
    // input until it was taught to sort. Google's canonical chat_template.jinja sorts
    // too (Jinja `| dictsort`), so sorted is the spec, not a coin flip. Pinned here
    // because the ordering is implicit -- switching to nlohmann::ordered_json would
    // silently reintroduce the split.

    TEST( GemmaProtocolParity, ToolCallArgumentsRenderInSortedOrder )
    {
        // Supplied in NON-sorted order so insertion order cannot pass. No whitespace
        // around ':' or ',' -- the trained format has none, and in a tokenized grammar
        // "key: value" and "key:value" are different tokens.
        EXPECT_EQ( formatToolCall( "get_weather", R"({"units":"c","location":"Paris"})" ),
            R"(<|tool_call>call:get_weather{location:<|"|>Paris<|"|>,units:<|"|>c<|"|>}<tool_call|>)" );
    }

    TEST( GemmaProtocolParity, ToolCallArgumentOrderIsInputOrderIndependent )
    {
        EXPECT_EQ( formatToolCall( "get_weather", R"({"location":"Paris","units":"c"})" ),
            formatToolCall( "get_weather", R"({"units":"c","location":"Paris"})" ) );
    }

    TEST( GemmaProtocolParity, FailedToolResponseSortsErrorBeforeResult )
    {
        // The exact input that split the two implementations: the response fields are
        // built {result, error}, so insertion order and sorted order disagree.
        EXPECT_EQ( formatToolResponse( "run", R"({"output":"partial","error":"boom"})" ),
            R"(<|tool_response>response:run{error:<|"|>boom<|"|>,result:<|"|>partial<|"|>}<tool_response|>)" );
    }

    TEST( GemmaProtocolParity, ManyArgumentsRenderInSortedOrder )
    {
        EXPECT_EQ( formatToolCall( "configure", R"({"zeta":1,"alpha":2,"middle":3})" ),
            "<|tool_call>call:configure{alpha:2,middle:3,zeta:1}<tool_call|>" );
    }

    // ---- Trained value grammar (non-scalar arguments) -----------------------

    // Non-scalar values RECURSE into the trained DSL rather than collapsing to raw
    // JSON. These are the cases that previously emitted plain double quotes -- tokens
    // the model never trained on -- on every call carrying an object or array
    // parameter. Goldens mirrored in test_gemma_protocol.py TestTrainedValueGrammar.

    TEST( GemmaProtocolParity, ArrayOfStringsUsesTrainedDelimiterNotJson )
    {
        const std::string rendered = formatToolCall( "edit", R"({"lines":["a","b"]})" );

        EXPECT_EQ( rendered, R"(<|tool_call>call:edit{lines:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>)" );
        EXPECT_EQ( rendered.find( "\"a\"" ), std::string::npos );
    }

    TEST( GemmaProtocolParity, NestedObjectRecursesWithBareKeys )
    {
        // Argument bodies render with the template's escape_keys=False, so nested
        // keys stay bare; only declarations wrap keys in the delimiter.
        EXPECT_EQ( formatToolCall( "configure", R"({"opts":{"n":1,"deep":true}})" ),
            "<|tool_call>call:configure{opts:{deep:true,n:1}}<tool_call|>" );
    }

    TEST( GemmaProtocolParity, NullAndBoolRenderAsBareLiterals )
    {
        EXPECT_EQ( formatToolCall( "set", R"({"flag":false,"missing":null})" ),
            "<|tool_call>call:set{flag:false,missing:null}<tool_call|>" );
    }

    TEST( GemmaProtocolParity, ArrayOfObjectsRecurses )
    {
        EXPECT_EQ( formatToolCall( "todo", R"({"items":[{"id":1,"tag":"x"}]})" ),
            R"(<|tool_call>call:todo{items:[{id:1,tag:<|"|>x<|"|>}]}<tool_call|>)" );
    }

    TEST( GemmaProtocolParity, NonMappingResponseUsesValueKey )
    {
        // The canonical template emits `value:` (not `result:`) for a non-mapping response.
        EXPECT_EQ( formatToolResponse( "echo", "plain text" ),
            R"(<|tool_response>response:echo{value:<|"|>plain text<|"|>}<tool_response|>)" );
    }

    // ---- Round-trip oracle (recursive-descent parser) -----------------------

    // render -> parse identity is THE oracle for the parser: it is correct exactly when
    // it is the inverse of renderValue over the value grammar. Before the recursive
    // rewrite only flat-scalar calls round-tripped -- a container truncated at the first
    // comma inside itself AND spilled its remaining contents out as sibling arguments.
    // Mirrored in test_gemma_protocol.py TestRoundTripOracle.

    namespace
    {
        /// Render `arguments_json` as a tool call, parse it back, return the arguments.
        nlohmann::json roundTrip( const std::string& arguments_json )
        {
            const auto call = parseToolCall( formatToolCall( "t", arguments_json ) );

            if ( !call.has_value() )
                return nlohmann::json( "PARSE FAILED" );

            return nlohmann::json::parse( call->arguments );
        }
    }

    TEST( GemmaProtocolRoundTrip, FlatScalarsSurvive )
    {
        const std::string arguments = R"({"count":42,"label":"hi","ratio":0.5,"flag":false})";
        EXPECT_EQ( roundTrip( arguments ), nlohmann::json::parse( arguments ) );
    }

    TEST( GemmaProtocolRoundTrip, ContainersSurvive )
    {
        const std::string arrays = R"({"lines":["a","b"]})";
        const std::string nested = R"({"opts":{"deep":true,"n":1}})";
        const std::string objects = R"({"items":[{"id":1,"tag":"x"},{"id":2,"tag":"y"}]})";
        const std::string deep = R"({"a":{"b":{"c":["d",{"e":1}]}}})";

        EXPECT_EQ( roundTrip( arrays ), nlohmann::json::parse( arrays ) );
        EXPECT_EQ( roundTrip( nested ), nlohmann::json::parse( nested ) );
        EXPECT_EQ( roundTrip( objects ), nlohmann::json::parse( objects ) );
        EXPECT_EQ( roundTrip( deep ), nlohmann::json::parse( deep ) );
    }

    TEST( GemmaProtocolRoundTrip, NullAndEmptyContainersSurvive )
    {
        const std::string arguments = R"({"blank":{},"missing":null,"nothing":[]})";
        EXPECT_EQ( roundTrip( arguments ), nlohmann::json::parse( arguments ) );
    }

    TEST( GemmaProtocolRoundTrip, GrammarPunctuationInsideStringsSurvives )
    {
        // The case the delimiter exists for: braces/brackets/commas that ARE grammar
        // elsewhere must be inert inside a delimited string.
        const std::string arguments = R"({"code":"func() { return [1,2]; }","where":"Toronto, Canada"})";
        EXPECT_EQ( roundTrip( arguments ), nlohmann::json::parse( arguments ) );
    }

    TEST( GemmaProtocolRoundTrip, ContainerValueIsAContainerNotAString )
    {
        const auto arguments = roundTrip( R"({"lines":["a","b"]})" );

        ASSERT_TRUE( arguments.at( "lines" ).is_array() );
        EXPECT_EQ( arguments.at( "lines" ).size(), 2u );
    }

    // ---- Container-shredding regression -------------------------------------

    // Reconstructs the live failure from the 2026-07-16 Claude Code A/B: the old parser
    // returned {"metadata":"{alpha:Done","beta":"Done","gamma":"Done","},subject":...}
    // -- the container's contents leaked out as sibling arguments and destroyed the
    // model's legitimate `subject` argument. Mirrored in test_gemma_protocol.py.

    TEST( GemmaProtocolShredding, ContainerParsesAsContainerAndSiblingsSurvive )
    {
        const auto call = parseToolCall(
            R"(<|tool_call>call:TaskUpdate{metadata:{alpha:Done,beta:Done,gamma:Done},subject:<|"|>Create todo list<|"|>,taskId:1}<tool_call|>)" );

        ASSERT_TRUE( call.has_value() );
        const auto arguments = nlohmann::json::parse( call->arguments );

        EXPECT_EQ( arguments.at( "metadata" ), nlohmann::json::parse( R"({"alpha":"Done","beta":"Done","gamma":"Done"})" ) );
        EXPECT_EQ( arguments.at( "subject" ), "Create todo list" );
        EXPECT_EQ( arguments.at( "taskId" ), 1 );
        EXPECT_EQ( arguments.size(), 3u );

        // The shredded siblings the old parser invented must not exist.
        EXPECT_FALSE( arguments.contains( "beta" ) );
        EXPECT_FALSE( arguments.contains( "gamma" ) );
    }

    // ---- Parser tolerance ----------------------------------------------------

    // The parser reads what the model EMITS, which is inconsistent: it stays lenient on
    // input while the renderer stays strict on output.

    TEST( GemmaProtocolParse, WhitespaceAroundSeparatorsIsTolerated )
    {
        const auto call = parseToolCall( R"(<|tool_call>call:t{a: <|"|>x<|"|> , b: [1, 2]}<tool_call|>)" );

        ASSERT_TRUE( call.has_value() );
        const auto arguments = nlohmann::json::parse( call->arguments );
        EXPECT_EQ( arguments.at( "a" ), "x" );
        EXPECT_EQ( arguments.at( "b" ), nlohmann::json::parse( "[1,2]" ) );
    }

    TEST( GemmaProtocolParse, TruncatedBodyKeepsWhatParsed )
    {
        // A cut-off call surfaces partially rather than blanking the turn.
        const auto call = parseToolCall( R"(<|tool_call>call:t{a:<|"|>x<|"|>,b:{c:1}<tool_call|>)" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( nlohmann::json::parse( call->arguments ).at( "a" ), "x" );
    }

    // ---- Control-token constants --------------------------------------------

    TEST( GemmaProtocolConstants, StringDelimiterIsTheRegisteredToken )
    {
        EXPECT_EQ( std::string( kStringDelimiter ), "<|\"|>" );
        EXPECT_EQ( std::string( kToolCallOpen ), "<|tool_call>" );
        EXPECT_EQ( std::string( kToolCallClose ), "<tool_call|>" );
    }

    // ---- Stray registered tokens --------------------------------------------
    // The checkpoint slips pipe-bracketed tokens this grammar has not enumerated into values.
    // Left in place they ride into the client's tool arguments -- the observed case is a path
    // arriving as `foo.cpp<|>`.

    TEST( GemmaProtocolParse, StrayPipeTokenScrubbedFromStringValue )
    {
        const auto call = parseToolCall(
            R"(<|tool_call>call:read{path:<|"|>foo.cpp<|><|"|>}<tool_call|>)" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( nlohmann::json::parse( call->arguments ).at( "path" ), "foo.cpp" );
    }

    TEST( GemmaProtocolParse, StrayPipeTokenScrubbedInsideContainer )
    {
        const auto call = parseToolCall(
            R"(<|tool_call>call:edit{files:[<|"|>a.cpp<|><|"|>]}<tool_call|>)" );

        ASSERT_TRUE( call.has_value() );
        EXPECT_EQ( nlohmann::json::parse( call->arguments ).at( "files" ),
            nlohmann::json::parse( R"(["a.cpp"])" ) );
    }

    TEST( GemmaProtocolParse, AngleFormMarkersAreNotMistakenForPipeTokens )
    {
        // <|channel> and friends close with '>' alone, so the pipe scrub must leave them --
        // they are removed by name, in order, by stripControlTokens.
        EXPECT_EQ( stripControlTokens( "a<|channel>b<channel|>c" ), "abc" );
    }

    // ---- Tool declarations ---------------------------------------------------
    // The golden strings here are asserted verbatim by the Python suite in
    // Mila/Adaptors/Inference/Server/tests/test_gemma_protocol.py, which now drives the same
    // renderer through the binding. Change one and the other must change in the same commit.

    TEST( GemmaProtocolDeclaration, RendersTrainedGrammarWithSortedPropertiesAndUppercasedTypes )
    {
        const std::string rendered = serializeToolDeclarations( R"([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "units": { "type": "string", "enum": ["c", "f"] },
                        "city": { "type": "string", "description": "The city" }
                    },
                    "required": ["city"]
                }
            }
        }])" );

        EXPECT_EQ( rendered,
            "<|tool>declaration:get_weather{description:<|\"|>Get the weather<|\"|>,"
            "parameters:{properties:{"
            "city:{description:<|\"|>The city<|\"|>,type:<|\"|>STRING<|\"|>},"
            "units:{enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}},"
            "required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>" );
    }

    TEST( GemmaProtocolDeclaration, ArrayPropertyRendersItemsBlock )
    {
        const std::string rendered = serializeToolDeclarations( R"([{
            "name": "tag",
            "description": "d",
            "parameters": { "type": "object",
                "properties": { "tags": { "type": "array", "items": { "type": "string" } } } }
        }])" );

        EXPECT_NE( rendered.find( "items:{type:<|\"|>STRING<|\"|>}" ), std::string::npos );
        EXPECT_NE( rendered.find( "type:<|\"|>ARRAY<|\"|>" ), std::string::npos );
    }

    TEST( GemmaProtocolDeclaration, ParametersBlockIsAlwaysClosed )
    {
        // The canonical template closes the block from inside its `type:` branch, so a schema
        // with no type leaves it unclosed and a comma dangling. This closes it either way.
        const std::string rendered = serializeToolDeclarations( R"([{
            "name": "f", "description": "d",
            "parameters": { "properties": { "a": { "type": "string" } } }
        }])" );

        ASSERT_FALSE( rendered.empty() );
        EXPECT_TRUE( rendered.ends_with( "}}<tool|>" ) );
    }

    TEST( GemmaProtocolDeclaration, NoParametersRendersDescriptionOnly )
    {
        EXPECT_EQ( serializeToolDeclarations( R"([{ "name": "ping", "description": "d" }])" ),
            "<|tool>declaration:ping{description:<|\"|>d<|\"|>}<tool|>" );
    }

    TEST( GemmaProtocolDeclaration, UnusableInputAdvertisesNothing )
    {
        // Empty is what tells the model there are no tools, so every degenerate input has to
        // reach it rather than emitting a half-formed declaration.
        EXPECT_EQ( serializeToolDeclarations( "" ), "" );
        EXPECT_EQ( serializeToolDeclarations( "[]" ), "" );
        EXPECT_EQ( serializeToolDeclarations( "not json" ), "" );
        EXPECT_EQ( serializeToolDeclarations( R"({"name":"f"})" ), "" );
        EXPECT_EQ( serializeToolDeclarations( R"([{"description":"no name"}])" ), "" );
        EXPECT_EQ( serializeToolDeclarations( R"([{"type":"retrieval","name":"f"}])" ), "" );
    }

    // ---- Turn template -------------------------------------------------------

    TEST( GemmaProtocolPrompt, FreshTurnCarriesBosSystemAndTheThoughtPrime )
    {
        const std::vector<Conversation::Turn> history{
            { Conversation::Role::System, "You are helpful." },
            { Conversation::Role::User, "Hi" } };

        EXPECT_EQ( formatPrompt( history ),
            "<bos><|turn>system\nYou are helpful.<turn|>\n"
            "<|turn>user\nHi<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>" );
    }

    TEST( GemmaProtocolPrompt, AssistantIsSpelledModel )
    {
        const std::vector<Conversation::Turn> history{
            { Conversation::Role::User, "Hi" },
            { Conversation::Role::Assistant, "Hello" },
            { Conversation::Role::User, "Again" } };

        EXPECT_NE( formatPrompt( history ).find( "<|turn>model\nHello<turn|>" ), std::string::npos );
    }

    TEST( GemmaProtocolPrompt, DeclarationsAttachToTheSystemTurnWithNoSeparator )
    {
        // <|tool> is an atomic special token, so it needs no whitespace to separate it from
        // the system prose -- and inserting some would be a prompt the model was not tuned on.
        const std::vector<Conversation::Turn> history{
            { Conversation::Role::System, "Be brief." },
            { Conversation::Role::User, "Hi" } };

        const std::string prompt = formatPrompt( history, "<|tool>declaration:f{}<tool|>" );

        EXPECT_NE( prompt.find( "<|turn>system\nBe brief.<|tool>declaration:f{}<tool|><turn|>" ),
            std::string::npos );
    }

    TEST( GemmaProtocolPrompt, DeclarationsAloneStillOpenASystemTurn )
    {
        const std::vector<Conversation::Turn> history{ { Conversation::Role::User, "Hi" } };

        EXPECT_TRUE( formatPrompt( history, "<|tool>declaration:f{}<tool|>" )
            .starts_with( "<bos><|turn>system\n<|tool>declaration:f{}<tool|><turn|>\n" ) );
    }

    TEST( GemmaProtocolPrompt, ContinueOpenLeavesTheFinalTurnUnclosedAndUnprimed )
    {
        // The shape after a tool response: the turn already carries its thought channel from
        // before the call, and priming a second one mid-turn is off-distribution.
        const std::vector<Conversation::Turn> history{
            { Conversation::Role::User, "Hi" },
            { Conversation::Role::Assistant, "partial" } };

        const std::string prompt = formatPrompt( history, {}, true );

        EXPECT_TRUE( prompt.ends_with( "<|turn>model\npartial" ) );
        EXPECT_EQ( prompt.find( kThoughtPrime ), std::string::npos );
    }

    TEST( GemmaProtocolPrompt, ContinueOpenOnEmptyHistoryIsRefused )
    {
        const std::vector<Conversation::Turn> empty;

        EXPECT_THROW( (void)formatPrompt( empty, {}, true ), std::invalid_argument );
    }

    TEST( GemmaProtocolPrompt, AssistantToolCallsRenderInTheNativeGrammar )
    {
        std::vector<Conversation::Turn> history{
            { Conversation::Role::User, "Weather?" },
            { Conversation::Role::Assistant, "" },
            { Conversation::Role::User, "and now?" } };

        history[ 1 ].tool_calls.push_back( { "get_weather", R"({"city":"Paris"})" } );

        EXPECT_NE( formatPrompt( history ).find(
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>" ),
            std::string::npos );
    }

    // ---- Answer extraction ---------------------------------------------------

    TEST( GemmaProtocolAnswer, ReasoningChannelIsRemoved )
    {
        EXPECT_EQ( extractAnswer( "<|channel>thought\nreasoning<channel|>Four." ), "Four." );
    }

    TEST( GemmaProtocolAnswer, InteriorChannelsAreRemovedNotJustALeadingRun )
    {
        // The 12B emits mid-answer thought channels on the agentic path DESPITE the prime, and
        // removing only the markers would leave the label and body behind as literal text.
        EXPECT_EQ( extractAnswer( "A<|channel>thought\nmid<channel|>B" ), "AB" );
    }

    TEST( GemmaProtocolAnswer, UnclosedTrailingChannelTruncatesRatherThanLeaking )
    {
        EXPECT_EQ( extractAnswer( "Answer.<|channel>thought\ncut off" ), "Answer." );
    }

    TEST( GemmaProtocolAnswer, CompleteToolCallSpanIsDropped )
    {
        EXPECT_EQ( extractAnswer( "before<|tool_call>call:f{a:1}<tool_call|>after" ),
            "beforeafter" );
    }

    TEST( GemmaProtocolAnswer, DanglingToolCallKeepsItsBodyRatherThanBlankingTheTurn )
    {
        // The off-spec `call:name:key=value` form the 12B emits with no closing marker. Dropping
        // from the open would collapse a response that STARTS with one to an empty string.
        EXPECT_EQ( extractAnswer( "<|tool_call>call:run:cmd=ls" ), "call:run:cmd=ls" );
    }
}
