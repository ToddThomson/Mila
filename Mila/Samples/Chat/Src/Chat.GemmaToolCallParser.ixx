/**
 * @file Chat.GemmaToolCallParser.ixx
 * @brief Detects and parses Gemma 4's native <|tool_call> protocol into a ToolCall value,
 *        and mirrors a dispatched result back into the matching <|tool_response> body.
 *
 * Unlike Llama's text-convention tool calls, Gemma 4's <|tool_call> / <tool_call|> and
 * <|tool_response> / <tool_response|> are registered vocabulary tokens (GemmaChatProtocol.md)
 * -- the call boundary is unambiguous, not pattern-matched. The body grammar itself
 * (call:function_name{key: "value", key2: 42}) was confirmed empirically against a live
 * checkpoint, not taken from the vendor doc (which claimed a since-disproven <|"|> string
 * delimiter). This is a best-effort probe grammar, not a hardened parser -- it exists to
 * validate the protocol, not to ship it.
 */

module;
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <format>
#include <atomic>
#include <cstdlib>

export module Chat.GemmaToolCallParser;

import Chat.Message;
import Chat.Json;

namespace Mila::ChatApp
{
    export class GemmaToolCallParser
    {
    public:

        /**
         * @brief Parse the most recently generated Gemma tool call out of the accumulated
         *        response text, once generation has stopped right after <tool_call|>.
         *
         * Expects body grammar call:function_name{key: "string value", key2: 42}. Values are
         * either double-quoted strings or bare literals (number/bool); bare literals that
         * parse as neither fall back to a raw string.
         *
         * Thread safety: parse() is stateless except for the call-id counter, which uses a
         * relaxed atomic increment (mirrors ToolCallParser::next_id_).
         */
        static std::optional<ToolCall> parse( const std::string& response )
        {
            constexpr std::string_view kOpen = "<|tool_call>";
            constexpr std::string_view kCallPrefix = "call:";

            const auto open_pos = response.rfind( kOpen );

            if ( open_pos == std::string::npos )
                return std::nullopt;

            const std::string_view body( response.data() + open_pos + kOpen.size(),
                response.size() - open_pos - kOpen.size() );

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

            ToolCall call;
            call.id = std::format( "gemma_call_{}", next_id_.fetch_add( 1, std::memory_order_relaxed ) );
            call.name = stripNamespace( trimCopy( body.substr( name_start, brace_open - name_start ) ) );

            const std::string_view args_body = body.substr( brace_open + 1, brace_close - brace_open - 1 );
            call.arguments = parseArguments( args_body ).dump();

            return call;
        }

        /**
         * @brief Mirror a dispatched tool's JSON result back into Gemma's call-side key:value
         *        grammar for the spliced <|tool_response> block (no confirmed vendor spec for
         *        the response body -- this applies the one grammar element measured on the
         *        call side symmetrically, rather than trusting the unverified doc format).
         */
        static std::string formatToolResponse( const std::string& name, const std::string& result_json )
        {
            nlohmann::json result;

            try
            {
                result = nlohmann::json::parse( result_json );
            }
            catch ( const nlohmann::json::parse_error& )
            {
                return std::format( "<|tool_response>response:{}{{result: \"{}\"}}<tool_response|>",
                    name, result_json );
            }

            if ( !result.is_object() )
                return std::format( "<|tool_response>response:{}{{result: \"{}\"}}<tool_response|>",
                    name, result_json );

            std::string body;
            bool first = true;

            for ( auto it = result.begin(); it != result.end(); ++it )
            {
                if ( !first )
                    body += ", ";
                first = false;

                body += it.key() + ": ";
                body += it.value().is_string() ? std::format( "\"{}\"", it.value().get<std::string>() ) : it.value().dump();
            }

            return std::format( "<|tool_response>response:{}{{{}}}<tool_response|>", name, body );
        }

    private:

        static inline std::atomic<uint32_t> next_id_{ 1 };

        static nlohmann::json parseArguments( std::string_view args_body )
        {
            nlohmann::json arguments = nlohmann::json::object();
            std::string_view remaining = args_body;

            while ( !remaining.empty() )
            {
                const auto colon_pos = remaining.find( ':' );

                if ( colon_pos == std::string_view::npos )
                    break;

                const std::string key = trimCopy( remaining.substr( 0, colon_pos ) );
                remaining = remaining.substr( colon_pos + 1 );

                const auto value_start = remaining.find_first_not_of( " \t" );
                remaining = ( value_start == std::string_view::npos ) ? std::string_view{} : remaining.substr( value_start );

                if ( remaining.empty() )
                    break;

                if ( remaining.front() == '"' )
                {
                    const auto close_quote = remaining.find( '"', 1 );

                    if ( close_quote == std::string_view::npos )
                        break;

                    arguments[ key ] = std::string( remaining.substr( 1, close_quote - 1 ) );

                    const auto next = remaining.find_first_not_of( ", \t", close_quote + 1 );
                    remaining = ( next == std::string_view::npos ) ? std::string_view{} : remaining.substr( next );
                }
                else
                {
                    const auto comma_pos = remaining.find( ',' );
                    const std::string bare = trimCopy( comma_pos == std::string_view::npos
                        ? remaining
                        : remaining.substr( 0, comma_pos ) );
                    remaining = ( comma_pos == std::string_view::npos )
                        ? std::string_view{}
                        : remaining.substr( comma_pos + 1 );

                    if ( bare == "true" )
                        arguments[ key ] = true;
                    else if ( bare == "false" )
                        arguments[ key ] = false;
                    else
                    {
                        char* end = nullptr;
                        const double d = std::strtod( bare.c_str(), &end );

                        if ( end != nullptr && *end == '\0' && end != bare.c_str() )
                            arguments[ key ] = d;
                        else
                            arguments[ key ] = bare;
                    }
                }
            }

            return arguments;
        }

        static std::string trimCopy( std::string_view text )
        {
            const auto first = text.find_first_not_of( " \t\r\n" );

            if ( first == std::string_view::npos )
                return {};

            const auto last = text.find_last_not_of( " \t\r\n" );

            return std::string( text.substr( first, last - first + 1 ) );
        }

        /**
         * @brief Reduce a possibly-namespaced call target to the bare function name.
         *
         * Gemma 4 inconsistently namespaces tools under a module (observed empirically:
         * bare "get_weather" on some turns, "default_api:get_weather" on others; the
         * Python-style "default_api.get_weather" is also plausible). Registered handlers
         * key on the bare name, so drop everything up to and including the last ':' or '.'
         * separator. A name with no separator is returned unchanged.
         */
        static std::string stripNamespace( const std::string& name )
        {
            const auto sep = name.find_last_of( ":." );

            if ( sep == std::string::npos )
                return name;

            return name.substr( sep + 1 );
        }
    };
}
