/**
 * @file Gemma.Protocol.ixx
 * @brief Canonical Gemma 4 native token grammar: parse/format for the model's
 *        registered turn / channel / tool-call / tool-response vocabulary.
 *
 * The Gemma 4 grammar is a property of the model, not of any single adaptor.
 * This runtime module is the one source of truth for it; the Chat harness and
 * (via a future pybind or a parity test) the Python inference server consume it
 * rather than each carrying a private copy that drifts. It was seeded from the
 * union of the two prior implementations -- the Python gemma_protocol.py, which
 * carried the spec-verified behaviors (the <|"|> string delimiter, tool-response
 * output-field distillation, failed-tool error surfacing), and the C++
 * GemmaToolCallParser it replaces. See GemmaChatProtocol.md.
 *
 * String-level parse/format only. Token-level splice into the live KV cache is
 * the decided direction but is post-release (see MilaProductFamily.md).
 */

module;
#include <string>
#include <string_view>
#include <optional>
#include <vector>
#include <cstdint>
#include <cstdlib>
// nlohmann::json::create instantiates here and compares a unique_ptr against nullptr, so this
// TU needs <memory>'s operators by ordinary lookup. Importing nlohmann.json does not supply
// them: they reach the BMI only as pruned global-module-fragment declarations.
#include <memory>

export module Dnn.Components.GemmaProtocol;

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
     * @brief A tool call parsed out of the model's native <|tool_call> emission.
     *
     * arguments is a JSON object string (nlohmann dump), so the adaptor can hand
     * it straight to a tool dispatcher without re-parsing the grammar.
     */
    export struct GemmaToolCall
    {
        std::string name;
        std::string arguments;
    };

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
         * Object keys stay bare: the template renders argument bodies with
         * escape_keys=False, reserving <|"|>-wrapped keys for tool declarations.
         */
        inline std::string renderValue( const nlohmann::json& value )
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

                    body += it.key() + ":" + renderValue( it.value() );
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

                    body += renderValue( item );
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
        call.name = detail::stripNamespace( detail::trim( body.substr( name_start, brace_open - name_start ) ) );

        if ( call.name.empty() )
            return std::nullopt;

        call.arguments = detail::parseArguments(
            body.substr( brace_open + 1, brace_close - brace_open - 1 ) ).dump();

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
}
