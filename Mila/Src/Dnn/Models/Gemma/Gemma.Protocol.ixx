/**
 * @file Gemma.Protocol.ixx
 * @brief Canonical Gemma 4 native token grammar: the turn template, the tool-declaration
 *        renderer, the tool-call grammar and answer extraction.
 *
 * The Gemma 4 grammar is a property of the model, not of any single adaptor, so this runtime
 * module is the one source of truth for it -- the same rule Qwen's module states (see
 * Qwen.Protocol.ixx). The Chat harness consumes it directly and the inference server reaches
 * it through the binding, so neither carries a copy that can drift.
 *
 * It was seeded from the union of the two prior implementations -- the Python
 * gemma_protocol.py, which carried the spec-verified behaviors (the <|"|> string delimiter,
 * tool-response output-field distillation, failed-tool error surfacing), and the C++
 * GemmaToolCallParser it replaces. The template, the <|tool> declaration renderer and answer
 * extraction came down later, from the same Python module, which then retired: until they did,
 * Gemma's grammar was written THREE times (this module, gemma_protocol.py, and a second
 * template in the server's prompt.py) and the adaptors disagreed -- Chat advertised tools as a
 * JSON array while the server rendered the trained declaration form the model was tuned on.
 * See GemmaChatProtocol.md.
 *
 * String-level parse/format only. Token-level splice into the live KV cache is
 * the decided direction but is post-release (see MilaProductFamily.md).
 */

module;
#include <string>
#include <string_view>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <cstdlib>
// nlohmann::json::create instantiates here and compares a unique_ptr against nullptr, so this
// TU needs <memory>'s operators by ordinary lookup. Importing nlohmann.json does not supply
// them: they reach the BMI only as pruned global-module-fragment declarations.
#include <memory>

export module Dnn.Models.GemmaProtocol;

export import Dnn.Models.Conversation;

import nlohmann.json;

namespace Mila::Dnn::Gemma
{
    // --- Registered control tokens (Mila Gemma 4 checkpoint) -----------------
    // The single source for these strings. Confirmed empirically against the live
    // 12B FP4 checkpoint; the checkpoint vocabulary registers <|turn> / <turn|>
    // (NOT the Gemma 3-style <start_of_turn> / <end_of_turn>). See GemmaChatProtocol.md.

    export inline constexpr std::string_view kBos = "<bos>";
    export inline constexpr std::string_view kEos = "<eos>";
    export inline constexpr std::string_view kPad = "<pad>";
    export inline constexpr std::string_view kTurnOpen = "<|turn>";
    export inline constexpr std::string_view kTurnClose = "<turn|>";
    export inline constexpr std::string_view kChannelOpen = "<|channel>";
    export inline constexpr std::string_view kChannelClose = "<channel|>";
    export inline constexpr std::string_view kThink = "<|think|>";
    export inline constexpr std::string_view kToolOpen = "<|tool>";
    export inline constexpr std::string_view kToolClose = "<tool|>";
    export inline constexpr std::string_view kToolCallOpen = "<|tool_call>";
    export inline constexpr std::string_view kToolCallClose = "<tool_call|>";
    export inline constexpr std::string_view kToolResponseOpen = "<|tool_response>";
    export inline constexpr std::string_view kToolResponseClose = "<tool_response|>";

    // Gemma inconsistently wraps tool-call string values in this registered
    // delimiter instead of plain double quotes (e.g. cmd: <|"|>ls -F<|"|>).
    // Confirmed empirically 2026-07-06; the earlier "not observed" note in the
    // spec doc was wrong. Parsed as an alternate string quote on the way in and
    // emitted on the way out.
    export inline constexpr std::string_view kStringDelimiter = "<|\"|>";

    /**
     * @brief The empty reasoning channel that opens a fresh model turn.
     *
     * Load-bearing rather than cosmetic: without it the 12B emits ghost thought channels and
     * generation degenerates. It belongs at the START of a fresh turn and nowhere else --
     * priming a second one mid-turn is off-distribution and the model parrots it back
     * (measured: empty-channel echoes and one 682-character runaway).
     */
    export inline constexpr std::string_view kThoughtPrime = "<|channel>thought\n<channel|>";

    /// The labels a reasoning channel is opened with, for answer extraction.
    export inline constexpr std::string_view kChannelLabels[] = {
        "thought", "thinking", "analysis", "reasoning" };

    /**
     * @brief A tool call parsed out of the model's native <|tool_call> emission.
     *
     * The family-neutral type, so a host describes a conversation once rather than once per
     * family: arguments is a JSON object string, which is what the model wrote and what a
     * dispatcher takes without re-parsing the grammar.
     */
    export using GemmaToolCall = Conversation::ToolCall;

    // --- Shared helpers ------------------------------------------------------

    namespace detail
    {
        inline std::string_view trim( std::string_view text )
        {
            const auto first = text.find_first_not_of( " \t\r\n" );

            if ( first == std::string_view::npos )
                return {};

            const auto last = text.find_last_not_of( " \t\r\n" );

            return text.substr( first, last - first + 1 );
        }

        /**
         * @brief Remove pipe-bracketed registered tokens the enumerated set does not name.
         *
         * Two forms: the delimiter family `<|...|>` (a sibling of `<|"|>`) and the bare `<|>`
         * the checkpoint emits. Left in place they ride verbatim into answers and into tool
         * arguments -- the observed case is a file path arriving as `foo.cpp<|>`.
         *
         * Hand-written rather than std::regex: MSVC's implementation is the one that already
         * costs the BPE pre-tokenizer its Unicode classes, and this grammar is two literal
         * shapes. The angle-form markers (`<|channel>`, `<|tool_call>`) do NOT match, because
         * the run between the brackets stops at '>' and the close must be "|>".
         */
        inline std::string stripPipeTokens( std::string_view text )
        {
            std::string result;
            result.reserve( text.size() );

            size_t cursor = 0;

            while ( cursor < text.size() )
            {
                if ( text[ cursor ] != '<' || cursor + 1 >= text.size() || text[ cursor + 1 ] != '|' )
                {
                    result += text[ cursor++ ];

                    continue;
                }

                size_t scan = cursor + 2;

                while ( scan < text.size() && text[ scan ] != '|' && text[ scan ] != '>' )
                    ++scan;

                if ( scan + 1 < text.size() && text[ scan ] == '|' && text[ scan + 1 ] == '>' )
                {
                    cursor = scan + 2;

                    continue;
                }

                // The bare `<|>`: an empty run closed by '>' alone.
                if ( scan == cursor + 2 && scan < text.size() && text[ scan ] == '>' )
                {
                    cursor = scan + 1;

                    continue;
                }

                result += text[ cursor++ ];
            }

            return result;
        }

        /// stripPipeTokens over every string in a parsed argument tree. Recursive, because the
        /// parser returns containers and a string nested in one needs the same scrub.
        inline nlohmann::json scrubPipeTokensDeep( const nlohmann::json& value )
        {
            if ( value.is_string() )
                return stripPipeTokens( value.get<std::string>() );

            if ( value.is_object() )
            {
                nlohmann::json scrubbed = nlohmann::json::object();

                for ( auto it = value.begin(); it != value.end(); ++it )
                    scrubbed[ it.key() ] = scrubPipeTokensDeep( it.value() );

                return scrubbed;
            }

            if ( value.is_array() )
            {
                nlohmann::json scrubbed = nlohmann::json::array();

                for ( const auto& item : value )
                    scrubbed.push_back( scrubPipeTokensDeep( item ) );

                return scrubbed;
            }

            return value;
        }

        /**
         * @brief Reduce a possibly-namespaced call target to the bare handler name.
         *
         * Gemma inconsistently namespaces tools under a module (observed: bare
         * "get_weather" on some turns, "default_api:get_weather" or the Python-style
         * "default_api.get_weather" on others). Handlers key on the bare name, so
         * drop everything up to and including the last ':' or '.' separator.
         */
        inline std::string stripNamespace( std::string_view name )
        {
            const auto separator = name.find_last_of( ":." );

            return std::string( separator == std::string_view::npos
                ? name
                : name.substr( separator + 1 ) );
        }

        /**
         * @brief Coerce a bare (unquoted) argument literal to bool / integer /
         *        double, falling back to a string when it is none of those.
         *
         * Integer is tried before double so 42 stays an integer in the emitted JSON
         * rather than becoming 42.0 (the spec-verified Python behavior; the retired
         * C++ parser coerced everything through strtod and lost the distinction).
         */
        inline nlohmann::json coerceBare( std::string_view bare )
        {
            if ( bare == "null" )
                return nullptr;

            if ( bare == "true" )
                return true;

            if ( bare == "false" )
                return false;

            const std::string text( bare );
            char* end = nullptr;

            const long long integer = std::strtoll( text.c_str(), &end, 10 );

            if ( end != nullptr && *end == '\0' && end != text.c_str() )
                return integer;

            end = nullptr;
            const double real = std::strtod( text.c_str(), &end );

            if ( end != nullptr && *end == '\0' && end != text.c_str() )
                return real;

            return text;
        }

        /**
         * @brief A read cursor over an argument body, for the recursive-descent parse.
         *
         * The grammar nests (objects and arrays hold values that are themselves objects
         * and arrays), so a single left-to-right scan with a shared position is the only
         * thing that composes. The predecessor searched for the next ',' from the start
         * of each value and so could not see container boundaries at all.
         */
        struct Cursor
        {
            std::string_view text;
            size_t position = 0;

            bool atEnd() const { return position >= text.size(); }
            char peek() const { return text[ position ]; }
            bool startsWith( std::string_view token ) const { return text.compare( position, token.size(), token ) == 0; }

            void skipWhitespace()
            {
                while ( position < text.size() && ( text[ position ] == ' ' || text[ position ] == '\t'
                    || text[ position ] == '\r' || text[ position ] == '\n' ) )
                {
                    ++position;
                }
            }
        };

        /// Consume a <|"|>-delimited span. The span is literal: the trained format has no
        /// backslash escaping, so it ends at the next delimiter token and nothing else.
        inline std::string parseDelimitedString( Cursor& cursor )
        {
            cursor.position += kStringDelimiter.size();
            const auto close = cursor.text.find( kStringDelimiter, cursor.position );

            if ( close == std::string_view::npos )
            {
                // Unterminated: take the rest rather than discarding the argument.
                std::string value( cursor.text.substr( cursor.position ) );
                cursor.position = cursor.text.size();

                return value;
            }

            std::string value( cursor.text.substr( cursor.position, close - cursor.position ) );
            cursor.position = close + kStringDelimiter.size();

            return value;
        }

        /// Consume a plain "..." span -- the alternate quoting the model also emits.
        inline std::string parseQuotedString( Cursor& cursor )
        {
            ++cursor.position;
            const auto close = cursor.text.find( '"', cursor.position );

            if ( close == std::string_view::npos )
            {
                std::string value( cursor.text.substr( cursor.position ) );
                cursor.position = cursor.text.size();

                return value;
            }

            std::string value( cursor.text.substr( cursor.position, close - cursor.position ) );
            cursor.position = close + 1;

            return value;
        }

        /// Consume a bare literal, stopping at whatever closes the enclosing construct.
        inline nlohmann::json parseBare( Cursor& cursor )
        {
            const size_t start = cursor.position;

            while ( !cursor.atEnd() && cursor.peek() != ',' && cursor.peek() != '}' && cursor.peek() != ']' )
                ++cursor.position;

            return coerceBare( trim( cursor.text.substr( start, cursor.position - start ) ) );
        }

        inline nlohmann::json parseValue( Cursor& cursor );

        /// Consume a key: bare, or delimiter-/quote-wrapped (declarations wrap keys, and
        /// the model sometimes mirrors that inside call arguments).
        inline std::string parseKey( Cursor& cursor )
        {
            cursor.skipWhitespace();

            if ( cursor.startsWith( kStringDelimiter ) )
                return parseDelimitedString( cursor );

            if ( !cursor.atEnd() && cursor.peek() == '"' )
                return parseQuotedString( cursor );

            const size_t start = cursor.position;

            while ( !cursor.atEnd() && cursor.peek() != ':' && cursor.peek() != ',' && cursor.peek() != '}' )
                ++cursor.position;

            return std::string( trim( cursor.text.substr( start, cursor.position - start ) ) );
        }

        inline nlohmann::json parseObject( Cursor& cursor )
        {
            nlohmann::json object = nlohmann::json::object();
            ++cursor.position;
            cursor.skipWhitespace();

            if ( !cursor.atEnd() && cursor.peek() == '}' )
            {
                ++cursor.position;

                return object;
            }

            while ( !cursor.atEnd() )
            {
                const std::string key = parseKey( cursor );
                cursor.skipWhitespace();

                if ( cursor.atEnd() || cursor.peek() != ':' )
                    break;

                ++cursor.position;
                object[ key ] = parseValue( cursor );
                cursor.skipWhitespace();

                if ( !cursor.atEnd() && cursor.peek() == ',' )
                {
                    ++cursor.position;
                    cursor.skipWhitespace();

                    continue;
                }

                if ( !cursor.atEnd() && cursor.peek() == '}' )
                    ++cursor.position;

                break;
            }

            return object;
        }

        inline nlohmann::json parseArray( Cursor& cursor )
        {
            nlohmann::json array = nlohmann::json::array();
            ++cursor.position;
            cursor.skipWhitespace();

            if ( !cursor.atEnd() && cursor.peek() == ']' )
            {
                ++cursor.position;

                return array;
            }

            while ( !cursor.atEnd() )
            {
                array.push_back( parseValue( cursor ) );
                cursor.skipWhitespace();

                if ( !cursor.atEnd() && cursor.peek() == ',' )
                {
                    ++cursor.position;

                    continue;
                }

                if ( !cursor.atEnd() && cursor.peek() == ']' )
                    ++cursor.position;

                break;
            }

            return array;
        }

        /**
         * @brief Parse one value of the trained grammar -- the inverse of renderValue.
         */
        inline nlohmann::json parseValue( Cursor& cursor )
        {
            cursor.skipWhitespace();

            if ( cursor.atEnd() )
                return nullptr;

            if ( cursor.startsWith( kStringDelimiter ) )
                return parseDelimitedString( cursor );

            if ( cursor.peek() == '"' )
                return parseQuotedString( cursor );

            if ( cursor.peek() == '{' )
                return parseObject( cursor );

            if ( cursor.peek() == '[' )
                return parseArray( cursor );

            return parseBare( cursor );
        }

        /**
         * @brief Parse a `key:value,key2:42` argument body into a JSON object.
         *
         * The inverse of renderArguments, and recursive: container values parse back to
         * containers. Whitespace around ':' and ',' is tolerated on the way in even though
         * the trained format emits none.
         *
         * Malformed input degrades rather than throwing -- a truncated body keeps the
         * arguments parsed so far, so a cut-off tool call surfaces partially instead of
         * blanking. It is NOT strict: this reads what the model emits, and the model is
         * inconsistent (see the plain-quote and namespacing notes above).
         */
        inline nlohmann::json parseArguments( std::string_view body )
        {
            nlohmann::json arguments = nlohmann::json::object();
            Cursor cursor{ body, 0 };

            while ( true )
            {
                cursor.skipWhitespace();

                if ( cursor.atEnd() )
                    break;

                const std::string key = parseKey( cursor );
                cursor.skipWhitespace();

                if ( key.empty() || cursor.atEnd() || cursor.peek() != ':' )
                    break;

                ++cursor.position;
                arguments[ key ] = parseValue( cursor );
                cursor.skipWhitespace();

                if ( !cursor.atEnd() && cursor.peek() == ',' )
                    ++cursor.position;
            }

            return arguments;
        }

        /**
         * @brief Render a string value in Gemma's trained delimiter form: <|"|>value<|"|>.
         *
         * The span between the delimiter tokens is literal -- the trained format has
         * no backslash escaping -- so an embedded delimiter (which would otherwise
         * close the span early) is replaced with a plain double quote.
         */
        inline std::string renderStringValue( const std::string& text )
        {
            std::string escaped = text;

            for ( auto pos = escaped.find( kStringDelimiter );
                pos != std::string::npos;
                pos = escaped.find( kStringDelimiter, pos + 1 ) )
            {
                escaped.replace( pos, kStringDelimiter.size(), "\"" );
            }

            return std::string( kStringDelimiter ) + escaped + std::string( kStringDelimiter );
        }

        /**
         * @brief Render one argument value in Gemma's trained grammar.
         *
         * Mirrors format_argument in Google's canonical chat_template.jinja: strings
         * take the <|"|> delimiter, null and bool are bare literals, and containers
         * RECURSE into the same grammar. Recursion is the point -- emitting a nested
         * object or array as raw JSON (["a","b"]) instead of the trained form
         * ([<|"|>a<|"|>,<|"|>b<|"|>]) puts untrained tokens in front of the model on
         * every call carrying a non-scalar parameter.
         *
         * @param escape_keys Wraps object KEYS in the delimiter. Argument bodies pass false
         *        (bare keys); tool declarations pass true on their free-form branch, which is
         *        the template's own default.
         */
        inline std::string renderValue( const nlohmann::json& value, bool escape_keys = false )
        {
            if ( value.is_null() )
                return "null";

            if ( value.is_string() )
                return renderStringValue( value.get<std::string>() );

            if ( value.is_boolean() )
                return value.get<bool>() ? "true" : "false";

            if ( value.is_object() )
            {
                std::string body = "{";
                bool first = true;

                for ( auto it = value.begin(); it != value.end(); ++it )
                {
                    if ( !first )
                        body += ",";
                    first = false;

                    body += ( escape_keys ? renderStringValue( it.key() ) : it.key() )
                        + ":" + renderValue( it.value(), escape_keys );
                }

                return body + "}";
            }

            if ( value.is_array() )
            {
                std::string body = "[";
                bool first = true;

                for ( const auto& item : value )
                {
                    if ( !first )
                        body += ",";
                    first = false;

                    body += renderValue( item, escape_keys );
                }

                return body + "]";
            }

            return value.dump();
        }

        /**
         * @brief Render a JSON object as Gemma's trained `key:value,...` argument body.
         *
         * No whitespace around ':' or ',' -- the trained format has none, and in a
         * tokenized grammar "key: value" and "key:value" are different tokens. Key
         * order is nlohmann's std::map ordering (sorted), matching the template's
         * `| dictsort` and the Python twin; see Gemma.Protocol.cpp GemmaProtocolParity.
         */
        inline std::string renderArguments( const nlohmann::json& values )
        {
            std::string body;
            bool first = true;

            for ( auto it = values.begin(); it != values.end(); ++it )
            {
                if ( !first )
                    body += ",";
                first = false;

                body += it.key() + ":" + renderValue( it.value() );
            }

            return body;
        }

        // --- Tool declarations ----------------------------------------------
        // Mirrors format_function_declaration in Google's canonical chat_template.jinja. The
        // parameters schema is a full DSL, not a JSON blob: a declaration reads
        //   <|tool>declaration:name{description:<|"|>..<|"|>,parameters:{properties:{
        //   city:{description:<|"|>..<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>city<|"|>],
        //   type:<|"|>OBJECT<|"|>}}<tool|>

        inline std::string toUpper( std::string_view text )
        {
            std::string upper( text );

            for ( char& character : upper )
            {
                if ( character >= 'a' && character <= 'z' )
                    character = static_cast<char>( character - 'a' + 'A' );
            }

            return upper;
        }

        /// A required-list body: the names are delimiter-wrapped, comma separated.
        inline std::string renderRequired( const nlohmann::json& required )
        {
            std::string body;
            bool first = true;

            for ( const auto& item : required )
            {
                if ( !first )
                    body += ",";
                first = false;

                body += renderStringValue(
                    item.is_string() ? item.get<std::string>() : item.dump() );
            }

            return body;
        }

        inline std::string renderParameters( const nlohmann::json& properties );

        /**
         * @brief The items:{...} body of an ARRAY property.
         *
         * Mirrors the template's items branch INCLUDING its quirks: `type` is uppercased and
         * delimiter-wrapped, and any other free-form key renders with escape_keys true -- so
         * nested mapping keys here ARE delimiter-wrapped, unlike in argument bodies.
         */
        inline std::string renderDeclarationItems( const nlohmann::json& items )
        {
            std::string body;
            bool first = true;

            const auto append = [&]( const std::string& part )
                {
                    if ( !first )
                        body += ",";
                    first = false;
                    body += part;
                };

            for ( auto it = items.begin(); it != items.end(); ++it )
            {
                if ( it.value().is_null() )
                    continue;

                if ( it.key() == "properties" )
                {
                    append( "properties:{"
                        + ( it.value().is_object() ? renderParameters( it.value() ) : std::string{} )
                        + "}" );
                }
                else if ( it.key() == "required" )
                {
                    append( "required:[" + renderRequired( it.value() ) + "]" );
                }
                else if ( it.key() == "type" )
                {
                    if ( it.value().is_string() )
                    {
                        append( "type:" + renderStringValue( toUpper( it.value().get<std::string>() ) ) );
                    }
                    else
                    {
                        nlohmann::json uppercased = nlohmann::json::array();

                        for ( const auto& entry : it.value() )
                        {
                            uppercased.push_back( toUpper(
                                entry.is_string() ? entry.get<std::string>() : entry.dump() ) );
                        }

                        append( "type:" + renderValue( uppercased ) );
                    }
                }
                else
                {
                    append( it.key() + ":" + renderValue( it.value(), true ) );
                }
            }

            return body;
        }

        /**
         * @brief One JSON-schema property in the trained declaration grammar.
         *
         * Field order is POSITIONAL, not alphabetical -- description, enum/items, nullable,
         * properties/required, type. That is the order the template emits and the order the
         * model saw in training, so it is not ours to tidy. Types are UPPERCASED.
         */
        inline std::string renderProperty( const nlohmann::json& schema )
        {
            std::vector<std::string> parts;

            const auto description = schema.find( "description" );

            if ( description != schema.end() && description->is_string()
                && !description->get<std::string>().empty() )
            {
                parts.push_back( "description:" + renderStringValue( description->get<std::string>() ) );
            }

            const auto type = schema.find( "type" );
            const std::string schema_type = type != schema.end() && type->is_string()
                ? toUpper( type->get<std::string>() ) : std::string{};

            const auto enumeration = schema.find( "enum" );
            const auto items = schema.find( "items" );

            if ( schema_type == "STRING" && enumeration != schema.end() && !enumeration->empty() )
            {
                parts.push_back( "enum:" + renderValue( *enumeration, true ) );
            }
            else if ( schema_type == "ARRAY" && items != schema.end() && items->is_object()
                && !items->empty() )
            {
                parts.push_back( "items:{" + renderDeclarationItems( *items ) + "}" );
            }

            const auto nullable = schema.find( "nullable" );

            if ( nullable != schema.end() && nullable->is_boolean() && nullable->get<bool>() )
                parts.push_back( "nullable:true" );

            if ( schema_type == "OBJECT" )
            {
                const auto properties = schema.find( "properties" );

                if ( properties != schema.end() && properties->is_object() )
                    parts.push_back( "properties:{" + renderParameters( *properties ) + "}" );

                const auto required = schema.find( "required" );

                if ( required != schema.end() && !required->empty() )
                    parts.push_back( "required:[" + renderRequired( *required ) + "]" );
            }

            parts.push_back( "type:" + renderStringValue( schema_type ) );

            std::string body = "{";

            for ( size_t index = 0; index < parts.size(); ++index )
            {
                if ( index != 0 )
                    body += ",";

                body += parts[ index ];
            }

            return body + "}";
        }

        /// The properties:{...} body: name:{...},name2:{...}, names in sorted order.
        inline std::string renderParameters( const nlohmann::json& properties )
        {
            std::string body;
            bool first = true;

            for ( auto it = properties.begin(); it != properties.end(); ++it )
            {
                if ( !it.value().is_object() )
                    continue;

                if ( !first )
                    body += ",";
                first = false;

                body += it.key() + ":" + renderProperty( it.value() );
            }

            return body;
        }
    }

    // --- Tool-call parsing (model output -> GemmaToolCall) -------------------

    /**
     * @brief Parse the most recent native Gemma tool call out of accumulated model
     *        output, once generation has stopped right after <tool_call|>.
     *
     * Returns nullopt when no <|tool_call> ... call:name{...} block is present or
     * the block is malformed (so the caller surfaces the text as-is rather than
     * looping on a call it cannot dispatch).
     */
    export std::optional<GemmaToolCall> parseToolCall( std::string_view text )
    {
        constexpr std::string_view kCallPrefix = "call:";

        const auto open = text.rfind( kToolCallOpen );

        if ( open == std::string_view::npos )
            return std::nullopt;

        const std::string_view body = text.substr( open + kToolCallOpen.size() );

        const auto call_pos = body.find( kCallPrefix );

        if ( call_pos == std::string_view::npos )
            return std::nullopt;

        const auto name_start = call_pos + kCallPrefix.size();
        const auto brace_open = body.find( '{', name_start );

        if ( brace_open == std::string_view::npos )
            return std::nullopt;

        const auto brace_close = body.rfind( '}' );

        if ( brace_close == std::string_view::npos || brace_close <= brace_open )
            return std::nullopt;

        GemmaToolCall call;
        call.name = detail::stripPipeTokens(
            detail::stripNamespace( detail::trim( body.substr( name_start, brace_open - name_start ) ) ) );

        if ( call.name.empty() )
            return std::nullopt;

        // A stray registered token the checkpoint slipped into a value rides into the client's
        // tool arguments otherwise -- the observed case is a path arriving as `foo.cpp<|>`.
        call.arguments = detail::scrubPipeTokensDeep( detail::parseArguments(
            body.substr( brace_open + 1, brace_close - brace_open - 1 ) ) ).dump();

        return call;
    }

    // --- Tool-call / tool-response formatting (replay -> prompt) -------------

    /**
     * @brief Render an assistant tool call back into Gemma's native call grammar.
     *
     * Non-object / unparseable arguments render as an empty body rather than
     * throwing, so a malformed history entry degrades to a bare call.
     */
    export std::string formatToolCall( std::string_view name, std::string_view arguments_json )
    {
        nlohmann::json values;

        try
        {
            values = nlohmann::json::parse( arguments_json );
        }
        catch ( const nlohmann::json::parse_error& )
        {
            values = nlohmann::json::object();
        }

        if ( !values.is_object() )
            values = nlohmann::json::object();

        return std::string( kToolCallOpen ) + "call:" + std::string( name )
            + "{" + detail::renderArguments( values ) + "}" + std::string( kToolCallClose );
    }

    // A JSON-envelope tool result surfaces exactly one of these fields to the
    // model; sibling fields (chunk ids, exit codes, timing) are metadata the model
    // must NOT see, or it echoes them back as content.
    inline constexpr std::string_view kOutputKeys[] = { "output", "result", "content", "stdout", "text" };

    /**
     * @brief Render a client-executed tool result into Gemma's <|tool_response> grammar.
     *
     * When the result is a JSON envelope only its primary output field is surfaced
     * (the first non-empty of output/result/content/stdout/text); metadata siblings
     * are dropped. A failed tool ({"content": "", "error": "..."}) has no usable
     * output field, so its `error` is surfaced explicitly -- without it the model
     * sees an empty result and blind-retries. A non-JSON result is passed through
     * under the `value:` key, which is what the canonical template emits for a
     * non-mapping response.
     */
    export std::string formatToolResponse( std::string_view name, std::string_view result_json )
    {
        nlohmann::json parsed;
        bool is_object = false;

        try
        {
            parsed = nlohmann::json::parse( result_json );
            is_object = parsed.is_object();
        }
        catch ( const nlohmann::json::parse_error& )
        {
            is_object = false;
        }

        std::string body;

        if ( is_object )
        {
            nlohmann::json fields = nlohmann::json::object();

            for ( const auto key : kOutputKeys )
            {
                const auto it = parsed.find( std::string( key ) );

                if ( it != parsed.end() && it->is_string() && !detail::trim( it->get<std::string>() ).empty() )
                {
                    fields[ "result" ] = it->get<std::string>();
                    break;
                }
            }

            const auto error = parsed.find( "error" );

            if ( error != parsed.end() && error->is_string() && !detail::trim( error->get<std::string>() ).empty() )
                fields[ "error" ] = error->get<std::string>();

            if ( fields.empty() )
                fields[ "result" ] = parsed.dump();

            body = detail::renderArguments( fields );
        }
        else
        {
            body = "value:" + detail::renderStringValue( std::string( result_json ) );
        }

        return std::string( kToolResponseOpen ) + "response:" + std::string( name )
            + "{" + body + "}" + std::string( kToolResponseClose );
    }

    // --- Tool declarations (schemas -> system-turn suffix) -------------------

    /**
     * @brief Render tool schemas into Gemma's trained <|tool>declaration:...<tool|> grammar.
     *
     * There is ONE declaration form because Gemma has one. A plain-text JSON list is not a
     * Gemma format at all -- it is prose plus a JSON dump -- and when it was measured against
     * this the 12B invented tools that did not exist.
     *
     * Deliberately NO call-syntax instructions: the model emits calls through its trained
     * <|tool_call> protocol, so teaching a foreign call format only confuses it.
     *
     * Declarations concatenate with no separator and no leading blank line, which is how the
     * template appends them to the system turn -- <|tool> is an atomic special token, so it
     * needs no whitespace to separate it from the system prose.
     *
     * ONE deliberate deviation from the template: it closes the parameters block from inside
     * its `type:` branch, so a schema with no `type` leaves the block unclosed and a comma
     * dangling. This always closes it. Malformed either way; this stays parseable.
     *
     * @param tools_json A JSON array of tool schemas, each either an OpenAI function envelope
     *        ({type, function:{name, description, parameters}}) or a bare declaration.
     *        Filtering which tools to advertise is the host's business, not the model's -- a
     *        harness with UI-only tools excludes them before calling.
     * @return Empty when the array is empty, does not parse, or names nothing, which omits the
     *         declarations entirely -- and that absence is what tells the model there are none.
     */
    export inline std::string serializeToolDeclarations( const std::string& tools_json )
    {
        nlohmann::json parsed;

        try
        {
            parsed = nlohmann::json::parse( tools_json.empty() ? "[]" : tools_json );
        }
        catch ( const nlohmann::json::exception& )
        {
            return {};
        }

        if ( !parsed.is_array() )
            return {};

        std::string declarations;

        for ( const auto& entry : parsed )
        {
            if ( !entry.is_object() )
                continue;

            const auto type = entry.find( "type" );

            if ( type != entry.end() && type->is_string() && type->get<std::string>() != "function" )
                continue;

            const auto function = entry.find( "function" );
            const nlohmann::json& declaration = function != entry.end() && function->is_object()
                ? *function : entry;

            const auto name = declaration.find( "name" );

            if ( name == declaration.end() || !name->is_string() || name->get<std::string>().empty() )
                continue;

            const auto description = declaration.find( "description" );

            std::string body = "description:" + detail::renderStringValue(
                description != declaration.end() && description->is_string()
                    ? description->get<std::string>() : std::string{} );

            const auto parameters = declaration.find( "parameters" );

            if ( parameters != declaration.end() && parameters->is_object() && !parameters->empty() )
            {
                std::vector<std::string> parameter_parts;

                const auto properties = parameters->find( "properties" );

                if ( properties != parameters->end() && properties->is_object() && !properties->empty() )
                    parameter_parts.push_back( "properties:{" + detail::renderParameters( *properties ) + "}" );

                const auto required = parameters->find( "required" );

                if ( required != parameters->end() && !required->empty() )
                    parameter_parts.push_back( "required:[" + detail::renderRequired( *required ) + "]" );

                const auto parameters_type = parameters->find( "type" );

                if ( parameters_type != parameters->end() && parameters_type->is_string()
                    && !parameters_type->get<std::string>().empty() )
                {
                    parameter_parts.push_back( "type:" + detail::renderStringValue(
                        detail::toUpper( parameters_type->get<std::string>() ) ) );
                }

                body += ",parameters:{";

                for ( size_t index = 0; index < parameter_parts.size(); ++index )
                {
                    if ( index != 0 )
                        body += ",";

                    body += parameter_parts[ index ];
                }

                body += "}";
            }

            declarations += std::string( kToolOpen ) + "declaration:" + name->get<std::string>()
                + "{" + body + "}" + std::string( kToolClose );
        }

        return declarations;
    }

    // --- Turn template (conversation -> prompt) ------------------------------

    /**
     * @brief How the template spells a role.
     *
     * The assistant is "model" in Gemma's vocabulary, and a tool result is a USER turn --
     * rendering Tool by its own name would open a role the model was never trained to read.
     */
    export inline std::string_view roleSpelling( Conversation::Role role )
    {
        switch ( role )
        {
            case Conversation::Role::System:
                return "system";

            case Conversation::Role::Assistant:
                return "model";

            case Conversation::Role::User:
            case Conversation::Role::Tool:
                return "user";
        }

        return "user";
    }

    /**
     * @brief One closed turn: <|turn>{role}\n{content}<turn|>\n.
     *
     * An Assistant turn appends each of its tool calls in the native call grammar. A Tool turn
     * carries its result ALREADY rendered -- formatToolResponse is the renderer, and it needs
     * the tool's name, which a Conversation::Turn does not carry.
     */
    export inline std::string formatTurn( const Conversation::Turn& message )
    {
        std::string turn;
        turn.reserve( 64 + message.content.size() );

        turn += kTurnOpen;
        turn += roleSpelling( message.role );
        turn += "\n";
        turn += message.content;

        for ( const auto& call : message.tool_calls )
            turn += formatToolCall( call.name, call.arguments );

        turn += kTurnClose;
        turn += "\n";

        return turn;
    }

    /**
     * @brief Render a conversation into Gemma's native prompt, primed for generation.
     *
     * The system turn is assembled here rather than taken from history, because the tool
     * declarations are a SUFFIX of it and a caller that had already concatenated them could not
     * put them in that order.
     *
     * @param tool_declarations What serializeToolDeclarations returns. Empty advertises none.
     * @param continue_open Emit the final turn OPEN, with no closing marker and no thought
     *        prime, so the next token continues it. That is the shape after a tool response:
     *        the turn already carries its thought channel from before the call, and priming a
     *        SECOND empty one mid-turn is off-distribution -- the model parrots it back.
     *        Otherwise a fresh model turn is opened and primed, which is where the empty
     *        thought channel belongs and the only place it does.
     *
     * @throws std::invalid_argument if continue_open is set on an empty history, since there
     *         would be no turn to resume.
     */
    export inline std::string formatPrompt(
        std::span<const Conversation::Turn> history,
        const std::string& tool_declarations = {},
        bool continue_open = false )
    {
        if ( continue_open && history.empty() )
        {
            throw std::invalid_argument(
                "Gemma::formatPrompt: continue_open needs a final turn to resume" );
        }

        std::string system_content;

        for ( const auto& message : history )
        {
            if ( message.role != Conversation::Role::System )
                continue;

            system_content += system_content.empty() ? "" : "\n\n";
            system_content += message.content;
        }

        // No separator: <|tool> is atomic, and the template appends declarations directly.
        system_content += tool_declarations;

        std::string prompt( kBos );

        if ( !system_content.empty() )
            prompt += formatTurn( { Conversation::Role::System, system_content } );

        const size_t body_count = continue_open ? history.size() - 1 : history.size();

        for ( size_t index = 0; index < body_count; ++index )
        {
            if ( history[ index ].role == Conversation::Role::System )
                continue;

            prompt += formatTurn( history[ index ] );
        }

        if ( continue_open )
        {
            const auto& last = history.back();

            prompt += kTurnOpen;
            prompt += roleSpelling( last.role );
            prompt += "\n";
            prompt += last.content;

            return prompt;
        }

        prompt += kTurnOpen;
        prompt += "model\n";
        prompt += kThoughtPrime;

        return prompt;
    }

    // --- Answer extraction (model output -> user-facing text) ---------------

    namespace detail
    {
        /// Drop every open..close span. An unterminated open truncates from there, which is
        /// what a response cut off mid-reasoning should yield: the prefix, not a leaked tail.
        inline std::string removeSpans( std::string_view text,
            std::string_view open_token, std::string_view close_token )
        {
            std::string result( text );

            while ( true )
            {
                const auto start = result.find( open_token );

                if ( start == std::string::npos )
                    return result;

                const auto end = result.find( close_token, start + open_token.size() );

                if ( end == std::string::npos )
                    return result.substr( 0, start );

                result = result.substr( 0, start ) + result.substr( end + close_token.size() );
            }
        }

        /**
         * @brief Drop complete tool spans; keep the BODY of a dangling one.
         *
         * The safety net for what parseToolCall could not classify. removeSpans would truncate
         * from the first open, so a response STARTING with an unclosed <|tool_call> would
         * collapse to nothing -- and the 12B does emit an off-spec `call:name:key=value` form
         * with no closing marker. Keeping the body degrades it to readable text instead of
         * blanking the turn; the residual open marker is cleared by stripControlTokens.
         */
        inline std::string stripToolSpans( std::string_view text,
            std::string_view open_token, std::string_view close_token )
        {
            std::string result;
            size_t cursor = 0;

            while ( true )
            {
                const auto start = text.find( open_token, cursor );

                if ( start == std::string_view::npos )
                {
                    result += text.substr( cursor );

                    return result;
                }

                const auto end = text.find( close_token, start + open_token.size() );
                result += text.substr( cursor, start - cursor );

                cursor = end == std::string_view::npos
                    ? start + open_token.size()
                    : end + close_token.size();
            }
        }
    }

    /// Remove every enumerated control token, then scrub whatever the set missed.
    export inline std::string stripControlTokens( std::string_view text )
    {
        constexpr std::string_view kControlTokens[] = {
            kBos, kEos, kPad, kTurnOpen, kTurnClose, kChannelOpen, kChannelClose, kThink,
            kToolOpen, kToolClose, kToolCallOpen, kToolCallClose,
            kToolResponseOpen, kToolResponseClose, kStringDelimiter,
            "<end_of_turn>", "<start_of_turn>" };

        std::string result( text );

        for ( const auto token : kControlTokens )
        {
            for ( auto position = result.find( token );
                position != std::string::npos;
                position = result.find( token, position ) )
            {
                result.erase( position, token.size() );
            }
        }

        return detail::stripPipeTokens( result );
    }

    /**
     * @brief Reduce a channel-structured response to just the user-facing answer.
     *
     * Reasoning channels appear more than once and INTERLEAVED with answer text -- the 12B
     * emits mid-answer thought channels on the agentic path despite the empty-thought prime --
     * so every channel span is removed, not just a leading run. Removing only the markers would
     * leave the label and its reasoning body behind as literal text.
     */
    export inline std::string extractAnswer( std::string_view text )
    {
        std::string result = detail::stripToolSpans( text, kToolCallOpen, kToolCallClose );
        result = detail::stripToolSpans( result, kToolResponseOpen, kToolResponseClose );
        result = detail::removeSpans( result, kChannelOpen, kChannelClose );

        const std::string cleaned = stripControlTokens( result );

        return std::string( detail::trim( cleaned ) );
    }
}
