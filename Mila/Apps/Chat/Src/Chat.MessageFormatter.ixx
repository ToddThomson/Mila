/**
 * @file Chat.MessageFormatter.ixx
 * @brief Llama 3.x instruct prompt formatter for the Mila chat application.
 *
 * Renders a sequence of ChatMessage turns into the token string expected
 * by Llama 3.x instruct models. The formatted string is passed to the
 * tokenizer before being fed to the model.
 */

module;
#include <string>
#include <vector>
#include <stdexcept>
#include <format>

export module Chat.MessageFormatter;

import Chat.Message;
import Chat.Json;

namespace Mila::ChatApp
{
    /**
     * @brief Renders a ChatMessage sequence into the Llama 3.x instruct prompt format.
     *
     * The Llama 3.x instruct template wraps each turn with role header tokens and
     * terminates completed turns with <|eot_id|>. The assistant header is appended
     * at the end without a closing <|eot_id|> to prime the model for generation.
     *
     * Complete turn format:
     * @code
     *   <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
     * @endcode
     *
     * Full conversation format:
     * @code
     *   <|begin_of_text|>
     *   <|start_header_id|>system<|end_header_id|>\n\n{system_content}<|eot_id|>
     *   <|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>
     *   <|start_header_id|>assistant<|end_header_id|>\n\n
     * @endcode
     *
     * Tool call turns (assistant role with tool_calls) format the call as a
     * <|python_tag|>-bounded JSON block in place of the content field:
     * @code
     *   <|start_header_id|>assistant<|end_header_id|>\n\n
     *   <|python_tag|>{"name": "...", "arguments": {...}}<|eom_id|>
     * @endcode
     *
     * Tool result turns (Tool role) format the result as:
     * @code
     *   <|start_header_id|>tool<|end_header_id|>\n\n{content}<|eot_id|>
     * @endcode
     *
     * Thread safety: all methods are stateless and safe to call concurrently.
     */
    export class MessageFormatter
    {
    public:

        /**
         * @brief Render a full conversation history into a model-ready prompt string.
         *
         * Appends the assistant primer at the end to prime generation. The last
         * message in history must be a User or Tool turn — passing an Assistant
         * turn as the final message throws std::invalid_argument.
         *
         * @param history Ordered sequence of conversation turns.
         * @return        Formatted prompt string ready for tokenization.
         * @throws std::invalid_argument if history is empty or ends with an Assistant turn.
         */
        static std::string format( const std::vector<ChatMessage>& history )
        {
            if ( history.empty() )
            {
                throw std::invalid_argument( "MessageFormatter: conversation history must not be empty" );
            }

            if ( history.back().role == MessageRole::Assistant )
            {
                throw std::invalid_argument(
                    "MessageFormatter: final message must be User or Tool, not Assistant" );
            }

            std::string prompt;
            prompt.reserve( estimateSize( history ) );

            prompt += kBos;

            for ( const auto& message : history )
            {
                prompt += formatTurn( message );
            }

            prompt += kAssistantPrimer;

            return prompt;
        }

        /**
         * @brief Render a single completed turn including its closing <|eot_id|>.
         *
         * Useful for appending a new turn to an already-formatted prefix without
         * re-rendering the full history.
         *
         * @param message The turn to render.
         * @return        Formatted turn string.
         */
        static std::string formatTurn( const ChatMessage& message )
        {
            std::string turn;
            turn.reserve( 128 );

            turn += kHeaderOpen;
            turn += messageRoleToString( message.role );
            turn += kHeaderClose;

            if ( message.role == MessageRole::Assistant && !message.tool_calls.empty() )
            {
                for ( const auto& call : message.tool_calls )
                {
                    turn += formatToolCall( call );
                }
            }
            else
            {
                turn += message.content;
                turn += kEot;
            }

            return turn;
        }

        /**
         * @brief Render a ToolCall in the Llama 3.2 3B pythonic format.
         *
         * Produces the zero-shot tool calling format expected by Llama 3.2 1B/3B:
         * @code
         *   [tool_name(param_name="value", param_name2="value2")]<|eot_id|>
         * @endcode
         *
         * Note: Llama 3.2 1B/3B uses pythonic format terminated with <|eot_id|>,
         * not the <|python_tag|>/<|eom_id|> format used by built-in tools and
         * larger Llama models.
         *
         * @param call The tool call to render.
         * @return     Formatted tool call string.
         */
        static std::string formatToolCall( const ToolCall& call )
        {
            nlohmann::json args = nlohmann::json::parse( call.arguments );

            std::string params;

            for ( auto it = args.begin(); it != args.end(); ++it )
            {
                if ( !params.empty() )
                    params += ", ";

                params += it.key();
                params += "=\"";
                params += it.value().get<std::string>();
                params += "\"";
            }

            return std::format( "{}[{}({})]{}", kPythonTag, call.name, params, kEot );
        }

    private:

        static constexpr std::string_view kBos = "<|begin_of_text|>";
        static constexpr std::string_view kHeaderOpen = "<|start_header_id|>";
        static constexpr std::string_view kHeaderClose = "<|end_header_id|>\n\n";
        static constexpr std::string_view kEot = "<|eot_id|>";
        //static constexpr std::string_view kEom = "<|eom_id|>";
        static constexpr std::string_view kPythonTag = "<|python_tag|>";
        static constexpr std::string_view kAssistantPrimer = "<|start_header_id|>assistant<|end_header_id|>\n\n";

        /**
         * @brief Estimate the formatted string size to reserve capacity upfront.
         *
         * Avoids repeated reallocation for long conversation histories.
         */
        static size_t estimateSize( const std::vector<ChatMessage>& history )
        {
            size_t total = kBos.size() + kAssistantPrimer.size();

            for ( const auto& m : history )
            {
                total += kHeaderOpen.size() + kHeaderClose.size() + kEot.size();
                total += messageRoleToString( m.role ).size();
                total += m.content.size();

                for ( const auto& call : m.tool_calls )
                {
                    total += kPythonTag.size() + kEot.size();
                    total += call.name.size() + call.arguments.size() + 32;
                }
            }

            return total;
        }
    };
}