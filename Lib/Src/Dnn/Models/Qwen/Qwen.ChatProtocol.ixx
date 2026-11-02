/**
 * @file Qwen.ChatProtocol.ixx
 * @brief Qwen 3.8's chat template behind the Protocol interface, rendered exactly as the checkpoint ships it.
 *
 * Ported from Qwen/Qwen3.8-27B's chat_template.jinja and held to it byte for byte by goldens that
 * Tools/ChatTemplates/record_template_goldens.py records through transformers.
 */

export module Dnn.Models.QwenChatProtocol;

import std;
import Dnn.Models.Conversation;
import Dnn.Models.Conversation.TemplateJson;
import Dnn.Models.QwenProtocol;

import nlohmann.json;

namespace Mila::Dnn::Qwen
{
    namespace detail
    {
        inline constexpr std::string_view kWhitespace = " \t\n\r\v\f";

        /// Jinja's `trim`, which is Python's str.strip().
        inline std::string_view trim( std::string_view text )
        {
            const auto first = text.find_first_not_of( kWhitespace );

            if ( first == std::string_view::npos )
            {
                return {};
            }

            return text.substr( first, text.find_last_not_of( kWhitespace ) - first + 1 );
        }

        inline constexpr std::string_view kCallInstructions =
            "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
            "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
            "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
            "Reminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> "
            "block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be "
            "specified\n- You may provide optional reasoning for your function call in natural language "
            "BEFORE the function call, but NOT after\n- If there is no function call available, answer the "
            "question like normal with your current knowledge and do not tell the user about function "
            "calls\n</IMPORTANT>";

        inline std::optional<Conversation::ToolCall> readToolCall( std::string_view span )
        {
            try
            {
                return parseToolCall( std::string( span ) );
            }
            catch ( const std::exception& )
            {
                return std::nullopt;
            }
        }

        /// A result the model writes is the engine's, so it is not part of the reply.
        inline Conversation::ReplyGrammar replyGrammar()
        {
            Conversation::ReplyGrammar grammar;

            grammar.reasoning_open = kThinkOpen;
            grammar.reasoning_close = kThinkClose;
            grammar.tool_call_open = kToolCallOpen;
            grammar.tool_call_close = kToolCallClose;
            grammar.tool_response_open = kToolResponseOpen;
            grammar.tool_response_close = kToolResponseClose;
            grammar.read_tool_call = &readToolCall;

            return grammar;
        }
    }

    /**
     * @brief The grammar of Qwen 3.8's chat template.
     *
     * Its variables are the template's own: enable_thinking (true), reasoning_effort (xhigh, medium,
     * low; read only while thinking is on) and preserve_thinking (true).
     */
    export class QwenChatProtocol final : public Conversation::Protocol
    {
    public:

        QwenChatProtocol()
            : variables_{
                { "enable_thinking", { "true", "false" }, "true" },
                { "reasoning_effort", { "xhigh", "medium", "low" }, "xhigh" },
                { "preserve_thinking", { "true", "false" }, "true" } },
            markers_{
                std::string( kThinkOpen ), std::string( kThinkClose ),
                std::string( kToolCallOpen ), std::string( kToolCallClose ), std::string( kToolResponseOpen ),
                { std::string( kToolResponseOpen ) } },
            grammar_( detail::replyGrammar() )
        {
        }

        /**
         * @throws std::invalid_argument for a template variable the template does not accept, a system
         *         turn that is not the first, no user turn to answer, or call arguments that are not a
         *         JSON object -- each a history the template itself refuses.
         */
        Conversation::Prompt formatPrompt(
            std::span<const Conversation::Turn> history, const Conversation::PromptOptions& options ) const override
        {
            Conversation::validateTemplateVariables( variables_, options.template_variables );

            if ( history.empty() )
            {
                throw std::invalid_argument( "Qwen chat template: no messages provided" );
            }

            const bool thinking = variable( options, "enable_thinking" ) == "true";
            const bool preserve_thinking = variable( options, "preserve_thinking" ) == "true";

            std::string_view instructions;

            if ( thinking )
            {
                const auto effort = variable( options, "reasoning_effort" );

                if ( effort == "xhigh" )
                {
                    instructions = reasoningInstruction( ReasoningEffort::Xhigh );
                } else if ( effort == "low" ) {
                    instructions = reasoningInstruction( ReasoningEffort::Low );
                }
            }

            const nlohmann::ordered_json tools = parseTools( options.tools_json );
            const bool first_is_system = history.front().role == Conversation::Role::System;

            std::string prompt;

            if ( !tools.empty() )
            {
                prompt += "<|im_start|>system\n";

                if ( !instructions.empty() )
                {
                    prompt += instructions;
                    prompt += "\n\n";
                }

                prompt += "# Tools\n\nYou have access to the following functions:\n\n<tools>";

                for ( const auto& tool : tools )
                {
                    prompt += "\n";
                    prompt += Conversation::templateJson( tool );
                }

                prompt += "\n</tools>";
                prompt += detail::kCallInstructions;

                if ( first_is_system )
                {
                    const auto content = detail::trim( history.front().content );

                    if ( !content.empty() )
                    {
                        prompt += "\n\n";
                        prompt += content;
                    }
                }

                prompt += "<|im_end|>\n";
            } else if ( first_is_system && !detail::trim( history.front().content ).empty() ) {
                prompt += "<|im_start|>system\n";

                if ( !instructions.empty() )
                {
                    prompt += instructions;
                    prompt += "\n\n";
                }

                prompt += detail::trim( history.front().content );
                prompt += "<|im_end|>\n";
            } else if ( !instructions.empty() ) {
                prompt += "<|im_start|>system\n";
                prompt += instructions;
                prompt += "<|im_end|>\n";
            }

            const std::size_t last_query = lastQueryIndex( history );

            for ( std::size_t index = 0; index < history.size(); ++index )
            {
                const auto& turn = history[ index ];
                const auto content = detail::trim( turn.content );

                switch ( turn.role )
                {
                    case Conversation::Role::System:
                        if ( index != 0 )
                        {
                            throw std::invalid_argument( "Qwen chat template: a system turn must be the first" );
                        }

                        break;

                    case Conversation::Role::User:
                        prompt += "<|im_start|>user\n";
                        prompt += content;
                        prompt += "<|im_end|>\n";
                        break;

                    case Conversation::Role::Assistant:
                        appendAssistant( prompt, turn, content, preserve_thinking || index > last_query );
                        break;

                    case Conversation::Role::Tool:
                        if ( index > 0 && history[ index - 1 ].role != Conversation::Role::Tool )
                        {
                            prompt += "<|im_start|>user";
                        }

                        prompt += "\n<tool_response>\n";
                        prompt += content;
                        prompt += "\n</tool_response>";

                        if ( index + 1 == history.size() || history[ index + 1 ].role != Conversation::Role::Tool )
                        {
                            prompt += "<|im_end|>\n";
                        }

                        break;
                }
            }

            prompt += "<|im_start|>assistant\n";
            prompt += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";

            return { std::move( prompt ), thinking };
        }

        const Conversation::ReplyGrammar& replyGrammar() const noexcept override
        {
            return grammar_;
        }

        std::string formatToolResult( std::string_view, std::string_view result ) const override
        {
            return std::format( "{}\n{}\n{}", kToolResponseOpen, detail::trim( result ), kToolResponseClose );
        }

        /**
         * The end of the assistant turn, the results in the user turn that carries them, and the primer of the next
         * assistant turn.
         *
         * @throws std::invalid_argument for no results, or one that is not a Tool turn.
         */
        Conversation::Prompt continueWithToolResults(
            std::span<const Conversation::Turn> results, const Conversation::PromptOptions& options ) const override
        {
            Conversation::validateTemplateVariables( variables_, options.template_variables );

            if ( results.empty() )
            {
                throw std::invalid_argument( "Qwen chat template: a reply that made calls continues with their results" );
            }

            std::string text = "<|im_end|>\n<|im_start|>user";

            for ( const auto& result : results )
            {
                if ( result.role != Conversation::Role::Tool )
                {
                    throw std::invalid_argument( "Qwen chat template: only tool results continue a reply that made calls" );
                }

                text += "\n";
                text += formatToolResult( result.tool_name, result.content );
            }

            const bool thinking = variable( options, "enable_thinking" ) == "true";

            text += "<|im_end|>\n<|im_start|>assistant\n";
            text += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";

            return { std::move( text ), thinking };
        }

        const Conversation::ProtocolMarkers& markers() const noexcept override
        {
            return markers_;
        }

        std::span<const Conversation::TemplateVariable> templateVariables() const noexcept override
        {
            return variables_;
        }

        // The template raises "System message must be at the beginning." for any other.
        bool rendersLaterSystemTurns() const noexcept override
        {
            return false;
        }

    private:

        std::vector<Conversation::TemplateVariable> variables_;
        Conversation::ProtocolMarkers markers_;
        Conversation::ReplyGrammar grammar_;

        std::string_view variable( const Conversation::PromptOptions& options, std::string_view name ) const
        {
            return Conversation::resolveTemplateVariable( variables_, options.template_variables, name );
        }

        /// Tools in the order given, with each one's keys in the order given: the template writes them through tojson.
        static nlohmann::ordered_json parseTools( const std::string& tools_json )
        {
            if ( detail::trim( tools_json ).empty() )
            {
                return nlohmann::ordered_json::array();
            }

            nlohmann::ordered_json tools = nlohmann::ordered_json::parse( tools_json, nullptr, false );

            if ( tools.is_discarded() || !tools.is_array() )
            {
                throw std::invalid_argument( "Qwen chat template: tools must be a JSON array" );
            }

            return tools;
        }

        /**
         * @brief The last user turn that is a question rather than a relayed tool result.
         *
         * The template's own scan: reasoning in history is rendered for every assistant turn after
         * it whether or not preserve_thinking is set.
         */
        static std::size_t lastQueryIndex( std::span<const Conversation::Turn> history )
        {
            for ( std::size_t index = history.size(); index-- > 0; )
            {
                if ( history[ index ].role != Conversation::Role::User )
                {
                    continue;
                }

                const auto content = detail::trim( history[ index ].content );

                if ( !( content.starts_with( kToolResponseOpen ) && content.ends_with( kToolResponseClose ) ) )
                {
                    return index;
                }
            }

            throw std::invalid_argument( "Qwen chat template: no user query found in messages" );
        }

        static void appendAssistant(
            std::string& prompt, const Conversation::Turn& turn, std::string_view content, bool render_reasoning )
        {
            prompt += "<|im_start|>assistant\n";

            if ( render_reasoning )
            {
                prompt += std::format( "{}\n{}\n{}\n\n", kThinkOpen, detail::trim( turn.reasoning ), kThinkClose );
            }

            prompt += content;

            for ( std::size_t index = 0; index < turn.tool_calls.size(); ++index )
            {
                const auto& call = turn.tool_calls[ index ];

                if ( index == 0 )
                {
                    prompt += content.empty() ? "" : "\n\n";
                } else {
                    prompt += "\n";
                }

                prompt += std::format( "{}\n<function={}>\n", kToolCallOpen, call.name );

                if ( !call.arguments.empty() )
                {
                    const nlohmann::ordered_json arguments = nlohmann::ordered_json::parse( call.arguments, nullptr, false );

                    if ( arguments.is_discarded() || !arguments.is_object() )
                    {
                        throw std::invalid_argument( std::format(
                            "Qwen chat template: the arguments of call '{}' are not a JSON object", call.name ) );
                    }

                    for ( auto entry = arguments.begin(); entry != arguments.end(); ++entry )
                    {
                        prompt += std::format( "<parameter={}>\n", entry.key() );
                        prompt += entry.value().is_string()
                            ? entry.value().get<std::string>() : Conversation::templateJson( entry.value() );
                        prompt += "\n</parameter>\n";
                    }
                }

                prompt += "</function>\n";
                prompt += kToolCallClose;
            }

            prompt += "<|im_end|>\n";
        }
    };
}
