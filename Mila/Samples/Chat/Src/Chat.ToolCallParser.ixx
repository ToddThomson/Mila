/**
 * @file Chat.ToolCallParser.ixx
 * @brief Detects and parses Llama 3.x tool call output into a ToolCall value.
 *
 * Scans a generated response string for a <|python_tag|>-bounded JSON block
 * and extracts the tool name and arguments. Returns std::nullopt for plain
 * text responses that contain no tool call boundary.
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
     * @brief Parses Llama 3.x <|python_tag|>-bounded tool call output.
     *
     * The Llama 3.x tool calling format wraps a JSON object between
     * <|python_tag|> and <|eom_id|>. The JSON object must contain a string
     * "name" field and an object "arguments" field:
     * @code
     *   <|python_tag|>{"name": "get_weather", "arguments": {"location": "Vancouver, CA"}}<|eom_id|>
     * @endcode
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
         * Returns std::nullopt when the response contains no <|python_tag|>
         * boundary (i.e., the model produced a plain text answer). The assigned
         * ToolCall::id is unique within the process lifetime.
         *
         * @param response Raw text produced by the model's generation step.
         * @return         Populated ToolCall on success, std::nullopt if no tool
         *                 call boundary is present.
         * @throws std::runtime_error if a <|python_tag|> boundary is found but
         *         the JSON is malformed, missing required fields, or unterminated.
         */
        static std::optional<ToolCall> parse( const std::string& response )
        {
            const auto tag_pos = response.find( kPythonTag );

            if ( tag_pos == std::string::npos )
                return std::nullopt;

            const auto json_start = tag_pos + kPythonTag.size();
            const auto eom_pos = response.find( kEom, json_start );

            if ( eom_pos == std::string::npos )
            {
                throw std::runtime_error(
                    "ToolCallParser: <|python_tag|> boundary found but <|eom_id|> terminator is missing" );
            }

            const std::string json_str = response.substr( json_start, eom_pos - json_start );

            nlohmann::json j;

            try
            {
                j = nlohmann::json::parse( json_str );
            }
            catch ( const nlohmann::json::parse_error& e )
            {
                throw std::runtime_error(
                    std::format( "ToolCallParser: malformed tool call JSON: {}", e.what() ) );
            }

            if ( !j.contains( "name" ) || !j[ "name" ].is_string() )
            {
                throw std::runtime_error(
                    "ToolCallParser: tool call JSON missing required string field 'name'" );
            }

            if ( !j.contains( "arguments" ) || !j[ "arguments" ].is_object() )
            {
                throw std::runtime_error(
                    "ToolCallParser: tool call JSON missing required object field 'arguments'" );
            }

            ToolCall call;
            call.id        = std::format( "call_{}", next_id_.fetch_add( 1, std::memory_order_relaxed ) );
            call.name      = j[ "name" ].get<std::string>();
            call.arguments = j[ "arguments" ].dump();

            return call;
        }

    private:

        static constexpr std::string_view kPythonTag = "<|python_tag|>";
        static constexpr std::string_view kEom       = "<|eom_id|>";

        static inline std::atomic<uint32_t> next_id_{ 1 };
    };
}