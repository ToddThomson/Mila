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
         * @brief Parse a `key: "value", key2: 42` argument body into a JSON object.
         *
         * String values are wrapped either in the registered <|"|> delimiter or in
         * plain double quotes; bare literals coerce via coerceBare. A quoted value
         * may contain commas (they do not terminate the value).
         */
        inline nlohmann::json parseArguments( std::string_view body )
        {
            nlohmann::json arguments = nlohmann::json::object();
            std::string_view remaining = body;

            while ( !remaining.empty() )
            {
                const auto colon = remaining.find( ':' );

                if ( colon == std::string_view::npos )
                    break;

                const std::string key( trim( remaining.substr( 0, colon ) ) );
                remaining = remaining.substr( colon + 1 );

                const auto value_start = remaining.find_first_not_of( " \t" );
                remaining = ( value_start == std::string_view::npos )
                    ? std::string_view{}
                    : remaining.substr( value_start );

                if ( remaining.empty() )
                    break;

                if ( remaining.substr( 0, kStringDelimiter.size() ) == kStringDelimiter )
                {
                    const auto close = remaining.find( kStringDelimiter, kStringDelimiter.size() );

                    if ( close == std::string_view::npos )
                        break;

                    arguments[ key ] = std::string(
                        remaining.substr( kStringDelimiter.size(), close - kStringDelimiter.size() ) );

                    const auto after = remaining.find_first_not_of( ", \t", close + kStringDelimiter.size() );
                    remaining = ( after == std::string_view::npos ) ? std::string_view{} : remaining.substr( after );
                }
                else if ( remaining.front() == '"' )
                {
                    const auto close = remaining.find( '"', 1 );

                    if ( close == std::string_view::npos )
                        break;

                    arguments[ key ] = std::string( remaining.substr( 1, close - 1 ) );

                    const auto after = remaining.find_first_not_of( ", \t", close + 1 );
                    remaining = ( after == std::string_view::npos ) ? std::string_view{} : remaining.substr( after );
                }
                else
                {
                    const auto comma = remaining.find( ',' );
                    const std::string_view bare = trim( comma == std::string_view::npos
                        ? remaining
                        : remaining.substr( 0, comma ) );
                    remaining = ( comma == std::string_view::npos )
                        ? std::string_view{}
                        : remaining.substr( comma + 1 );

                    arguments[ key ] = coerceBare( bare );
                }
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

        /// Render a JSON object as Gemma's `key: value, ...` body (string values in
        /// the trained delimiter; everything else as its compact JSON form).
        inline std::string renderArguments( const nlohmann::json& values )
        {
            std::string body;
            bool first = true;

            for ( auto it = values.begin(); it != values.end(); ++it )
            {
                if ( !first )
                    body += ", ";
                first = false;

                body += it.key() + ": ";
                body += it.value().is_string()
                    ? renderStringValue( it.value().get<std::string>() )
                    : it.value().dump();
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
     * as a single `result:` string.
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
            body = "result: " + detail::renderStringValue( std::string( result_json ) );
        }

        return std::string( kToolResponseOpen ) + "response:" + std::string( name )
            + "{" + body + "}" + std::string( kToolResponseClose );
    }
}
