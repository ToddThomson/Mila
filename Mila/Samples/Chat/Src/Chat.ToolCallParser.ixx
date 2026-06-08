/**
 * @file Chat.ToolCallParser.ixx
 * @brief Detects and parses Llama 3.x tool call output into a ToolCall value.
 *
 * Scans a generated response string for a <|python_tag|>-prefixed block and
 * extracts a pythonic call or JSON tool call. Falls back to bare bracket or
 * JSON detection for models that omit the prefix token.
 */

module;
#include <string>
#include <vector>
#include <optional>
#include <stdexcept>
#include <format>
#include <atomic>

export module Chat.ToolCallParser;

import Chat.Message;
import Chat.Json;

namespace Mila::ChatApp
{
    /**
     * @brief Parses Llama 3.x tool call output from a raw generated response string.
     *
     * The generation loop strips stop tokens before they reach the callback, so the
     * response string never contains a literal <|eot_id|> or <|eom_id|> terminator.
     * parseTagged therefore treats end-of-string as an implicit terminator when the
     * <|python_tag|> prefix is present.
     *
     * Detection order:
     *   1. <|python_tag|> prefix  -> parseTagged  (pythonic or JSON content)
     *   2. Leading '['            -> parsePythonicCall
     *   3. Fallback               -> parseBareJson
     *
     * Thread safety: parse() is stateless except for the call_id counter,
     * which uses a relaxed atomic increment. Concurrent calls produce unique
     * ids but do not impose ordering guarantees on each other.
     */
    export class ToolCallParser
    {
    public:

        /**
         * @brief Attempt to parse a tool call from a raw generated response.
         *
         * Returns std::nullopt when no tool call pattern is detected.
         *
         * @param response Raw text produced by the model's generation step.
         * @return         Populated ToolCall on success, std::nullopt otherwise.
         * @throws std::runtime_error if a <|python_tag|> prefix is found but the
         *         content is malformed or contains no recognisable call structure.
         */
        static std::optional<ToolCall> parse( const std::string& response )
        {
            const auto tag_pos = response.find( kPythonTag );

            if ( tag_pos != std::string::npos )
                return parseTagged( response, tag_pos );

            const auto bracket_pos = response.find( '[' );

            if ( bracket_pos != std::string::npos )
                return parsePythonicCall( response, bracket_pos );

            return parseBareJson( response );
        }

    private:

        static constexpr std::string_view kPythonTag = "<|python_tag|>";
        static constexpr std::string_view kEot = "<|eot_id|>";
        static constexpr std::string_view kEom = "<|eom_id|>";

        static inline std::atomic<uint32_t> next_id_{ 1 };

        /**
         * @brief Parse a <|python_tag|>-prefixed tool call block.
         *
         * The generation loop consumes stop tokens without forwarding them to the
         * on_token callback, so <|eot_id|> and <|eom_id|> are never present in the
         * response string. End-of-string is therefore a valid implicit terminator
         * and is tried last after the explicit token search.
         */
        static std::optional<ToolCall> parseTagged( const std::string& response, size_t tag_pos )
        {
            const auto content_start = tag_pos + kPythonTag.size();

            // Prefer an explicit terminator when present; accept end-of-string as an
            // implicit terminator because the generation loop strips stop tokens.
            auto end_pos = response.find( kEot, content_start );

            if ( end_pos == std::string::npos )
                end_pos = response.find( kEom, content_start );

            const std::string content = (end_pos != std::string::npos)
                ? response.substr( content_start, end_pos - content_start )
                : response.substr( content_start );

            const auto trim_start = content.find_first_not_of( " \t\n\r" );

            if ( trim_start == std::string::npos )
                return std::nullopt;

            const std::string trimmed = content.substr( trim_start );

            if ( trimmed.front() == '[' )
                return parsePythonicCall( trimmed, 0 );

            return parseJson( trimmed, /*throws_on_error=*/true );
        }

        static std::optional<ToolCall> parseBareJson( const std::string& response )
        {
            const auto brace_start = response.find( '{' );

            if ( brace_start == std::string::npos )
                return std::nullopt;

            const auto brace_end = response.rfind( '}' );

            if ( brace_end == std::string::npos || brace_end <= brace_start )
                return std::nullopt;

            const std::string json_str = response.substr( brace_start, brace_end - brace_start + 1 );

            return parseJson( json_str, /*throws_on_error=*/false );
        }

        static std::optional<ToolCall> parseJson( const std::string& json_str, bool throws_on_error )
        {
            nlohmann::json j;

            try
            {
                j = nlohmann::json::parse( json_str );
            }
            catch ( const nlohmann::json::parse_error& e )
            {
                if ( throws_on_error )
                {
                    throw std::runtime_error(
                        std::format( "ToolCallParser: malformed tool call JSON: {}", e.what() ) );
                }
                return std::nullopt;
            }

            if ( !j.contains( "name" ) || !j[ "name" ].is_string() )
                return std::nullopt;

            if ( !j.contains( "arguments" ) || !j[ "arguments" ].is_object() )
                return std::nullopt;

            ToolCall call;
            call.id = std::format( "call_{}", next_id_.fetch_add( 1, std::memory_order_relaxed ) );
            call.name = j[ "name" ].get<std::string>();
            call.arguments = j[ "arguments" ].dump();

            return call;
        }

        static std::optional<ToolCall> parsePythonicCall( const std::string& response, size_t bracket_pos )
        {
            const auto bracket_end = response.rfind( ']' );

            if ( bracket_end == std::string::npos || bracket_end <= bracket_pos )
                return std::nullopt;

            const std::string inner = response.substr( bracket_pos + 1, bracket_end - bracket_pos - 1 );

            const auto paren_pos = inner.find( '(' );

            if ( paren_pos == std::string::npos )
                return std::nullopt;

            const auto paren_end = inner.rfind( ')' );

            if ( paren_end == std::string::npos || paren_end <= paren_pos )
                return std::nullopt;

            const std::string tool_name = inner.substr( 0, paren_pos );
            const std::string params_str = inner.substr( paren_pos + 1, paren_end - paren_pos - 1 );

            nlohmann::json arguments = nlohmann::json::object();

            std::string param = params_str;

            while ( !param.empty() )
            {
                const auto eq_pos = param.find( '=' );

                if ( eq_pos == std::string::npos )
                    break;

                std::string key = param.substr( 0, eq_pos );
                param = param.substr( eq_pos + 1 );

                key.erase( 0, key.find_first_not_of( " \t" ) );
                key.erase( key.find_last_not_of( " \t" ) + 1 );

                std::string value;

                if ( !param.empty() && param.front() == '"' )
                {
                    const auto close_quote = param.find( '"', 1 );

                    if ( close_quote == std::string::npos )
                        return std::nullopt;

                    value = param.substr( 1, close_quote - 1 );
                    param = param.substr( close_quote + 1 );

                    const auto next = param.find_first_not_of( ", \t" );
                    param = (next == std::string::npos) ? "" : param.substr( next );
                }
                else
                {
                    const auto comma_pos = param.find( ',' );

                    if ( comma_pos == std::string::npos )
                    {
                        value = param;
                        param.clear();
                    }
                    else
                    {
                        value = param.substr( 0, comma_pos );
                        param = param.substr( comma_pos + 1 );

                        const auto next = param.find_first_not_of( " \t" );
                        param = (next == std::string::npos) ? "" : param.substr( next );
                    }

                    value.erase( value.find_last_not_of( " \t" ) + 1 );
                }

                arguments[ key ] = value;
            }

            ToolCall call;
            call.id = std::format( "call_{}", next_id_.fetch_add( 1, std::memory_order_relaxed ) );
            call.name = tool_name;
            call.arguments = arguments.dump();

            return call;
        }
    };
}