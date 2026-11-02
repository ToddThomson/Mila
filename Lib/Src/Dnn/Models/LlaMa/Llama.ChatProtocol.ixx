/**
 * @file Llama.ChatProtocol.ixx
 * @brief Llama 3.1's and 3.2's chat templates behind the Protocol interface, rendered exactly as Meta ships them.
 *
 * Ported from the chat_template in meta-llama/Llama-3.1-8B-Instruct's and Llama-3.2-3B-Instruct's
 * tokenizer_config.json and held to both byte for byte by goldens that
 * Tools/ChatTemplates/record_template_goldens.py records through transformers.
 */

export module Dnn.Models.LlamaChatProtocol;

import std;
import Dnn.Models.Conversation;
import Dnn.Models.Conversation.TemplateJson;

import nlohmann.json;

namespace Mila::Dnn::Llama
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

        inline constexpr std::string_view kBeginOfText = "<|begin_of_text|>";
        inline constexpr std::string_view kEndOfTurn = "<|eot_id|>";
        inline constexpr std::string_view kEndOfMessage = "<|eom_id|>";
        inline constexpr std::string_view kPythonTag = "<|python_tag|>";

        inline std::string header( std::string_view role )
        {
            return std::format( "<|start_header_id|>{}<|end_header_id|>\n\n", role );
        }

        inline constexpr std::string_view kCallFormat =
            "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}."
            "Do not use variables.\n\n";
    }

    /**
     * @brief The grammar of Meta's Llama 3.1 and 3.2 chat templates.
     *
     * One renderer for two templates: they differ in the default of `date_string` -- a fixed
     * 26 Jul 2024 in 3.1, today's date in 3.2 -- and in 3.1's built-in tools, which this renderer
     * does not take, so a request for them is refused as an unknown variable. A turn makes at most
     * one call, and a result is a turn of its own written through `tojson`, so a plain-text result
     * reaches the model JSON-quoted -- both as the templates write them.
     */
    export class LlamaChatProtocol final : public Conversation::Protocol
    {
    public:

        /// The meta-llama/Llama-3.1-8B-Instruct template.
        static LlamaChatProtocol llama31()
        {
            return LlamaChatProtocol( "26 Jul 2024" );
        }

        /// The meta-llama/Llama-3.2-3B-Instruct template, whose date defaults to today's.
        static LlamaChatProtocol llama32()
        {
            return LlamaChatProtocol( std::string{} );
        }

        /**
         * @throws std::invalid_argument for a template variable the template does not accept, a turn
         *         with more than one call, call arguments that are not a JSON object, or tools offered
         *         with no message to carry them -- each a history the template itself refuses.
         */
        Conversation::Prompt formatPrompt(
            std::span<const Conversation::Turn> history, const Conversation::PromptOptions& options ) const override
        {
            Conversation::validateTemplateVariables( variables_, options.template_variables );

            const bool tools_in_user_message = variable( options, "tools_in_user_message" ) == "true";
            std::string date( variable( options, "date_string" ) );

            if ( date.empty() )
            {
                date = today();
            }

            const std::optional<nlohmann::ordered_json> tools = parseTools( options.tools_json );

            auto turns = history;
            std::string_view system_message;

            if ( !turns.empty() && turns.front().role == Conversation::Role::System )
            {
                system_message = detail::trim( turns.front().content );
                turns = turns.subspan( 1 );
            }

            std::string prompt( detail::kBeginOfText );
            prompt += detail::header( "system" );

            if ( tools.has_value() )
            {
                prompt += "Environment: ipython\n";
            }

            prompt += "Cutting Knowledge Date: December 2023\n";
            prompt += std::format( "Today Date: {}\n\n", date );

            if ( tools.has_value() && !tools_in_user_message )
            {
                prompt += "You have access to the following functions. To call a function, please respond with JSON for a function call.";
                prompt += detail::kCallFormat;
                appendTools( prompt, *tools );
            }

            prompt += system_message;
            prompt += detail::kEndOfTurn;

            if ( tools.has_value() && tools_in_user_message )
            {
                if ( turns.empty() )
                {
                    throw std::invalid_argument(
                        "Llama chat template: cannot put tools in the first user message when there is no first user message" );
                }

                prompt += detail::header( "user" );
                prompt += "Given the following functions, please respond with a JSON for a function call "
                    "with its proper arguments that best answers the given prompt.\n\n";
                prompt += detail::kCallFormat;
                appendTools( prompt, *tools );
                prompt += detail::trim( turns.front().content );
                prompt += detail::kEndOfTurn;
                turns = turns.subspan( 1 );
            }

            for ( const auto& turn : turns )
            {
                if ( turn.role == Conversation::Role::Tool )
                {
                    prompt += detail::header( "ipython" );
                    prompt += formatToolResult( turn.tool_name, turn.content );
                    prompt += detail::kEndOfTurn;
                } else if ( !turn.tool_calls.empty() ) {
                    prompt += detail::header( "assistant" );
                    prompt += formatCall( turn.tool_calls );
                    prompt += detail::kEndOfTurn;
                } else {
                    prompt += detail::header( roleSpelling( turn.role ) );
                    prompt += detail::trim( turn.content );
                    prompt += detail::kEndOfTurn;
                }
            }

            prompt += detail::header( "assistant" );

            return { std::move( prompt ), false };
        }

        const Conversation::ReplyGrammar& replyGrammar() const noexcept override
        {
            return grammar_;
        }

        /// A result as the template writes the body of its ipython turn: through tojson, so text is JSON-quoted.
        std::string formatToolResult( std::string_view, std::string_view result ) const override
        {
            return Conversation::templateJson( nlohmann::ordered_json( std::string( result ) ) );
        }

        /**
         * The end of the assistant turn, the ipython turn that carries the result, and the next assistant header.
         *
         * @throws std::invalid_argument for anything but one result: a turn makes at most one call.
         */
        Conversation::Prompt continueWithToolResults(
            std::span<const Conversation::Turn> results, const Conversation::PromptOptions& options ) const override
        {
            Conversation::validateTemplateVariables( variables_, options.template_variables );

            if ( results.size() != 1 || results.front().role != Conversation::Role::Tool )
            {
                throw std::invalid_argument( "Llama chat template: a reply that made its one call continues with one result" );
            }

            std::string text( detail::kEndOfTurn );
            text += detail::header( "ipython" );
            text += formatToolResult( results.front().tool_name, results.front().content );
            text += detail::kEndOfTurn;
            text += detail::header( "assistant" );

            return { std::move( text ), false };
        }

        const Conversation::ProtocolMarkers& markers() const noexcept override
        {
            return markers_;
        }

        std::span<const Conversation::TemplateVariable> templateVariables() const noexcept override
        {
            return variables_;
        }

        bool rendersLaterSystemTurns() const noexcept override
        {
            return true;
        }

    private:

        std::vector<Conversation::TemplateVariable> variables_;
        Conversation::ProtocolMarkers markers_;
        Conversation::ReplyGrammar grammar_;

        explicit LlamaChatProtocol( std::string default_date )
            : variables_{
                { "tools_in_user_message", { "true", "false" }, "true" },
                { "date_string", {}, std::move( default_date ) } },
            markers_{ {}, {}, std::string( detail::kPythonTag ), {}, {}, {} },
            grammar_( templateReplyGrammar() )
        {
        }

        /**
         * A reply that is a JSON object, or what follows `<|python_tag|>`, is a call, and it runs to the end of
         * the reply; anything else is content. A turn makes at most one call.
         */
        static Conversation::ReplyGrammar templateReplyGrammar()
        {
            Conversation::ReplyGrammar grammar;

            grammar.tool_call_open = detail::kPythonTag;
            grammar.bare_tool_call_open = "{";
            grammar.reply_end = { std::string( detail::kEndOfTurn ), std::string( detail::kEndOfMessage ) };
            grammar.read_tool_call = &readToolCall;

            return grammar;
        }

        /**
         * Arguments are read from "parameters", the key the template asks for, or from "arguments", which the
         * model also writes.
         */
        static std::optional<Conversation::ToolCall> readToolCall( std::string_view span )
        {
            if ( span.starts_with( detail::kPythonTag ) )
            {
                span.remove_prefix( detail::kPythonTag.size() );
            }

            try
            {
                return parseCall( detail::trim( span ) );
            }
            catch ( const std::exception& )
            {
                return std::nullopt;
            }
        }

        std::string_view variable( const Conversation::PromptOptions& options, std::string_view name ) const
        {
            return Conversation::resolveTemplateVariable( variables_, options.template_variables, name );
        }

        /// 3.2's default, as strftime_now("%d %b %Y") writes it.
        static std::string today()
        {
            const auto now = std::chrono::current_zone()->to_local( std::chrono::system_clock::now() );

            return std::format( "{:%d %b %Y}", std::chrono::floor<std::chrono::days>( now ) );
        }

        static std::string_view roleSpelling( Conversation::Role role )
        {
            switch ( role )
            {
                case Conversation::Role::System:
                    return "system";

                case Conversation::Role::Assistant:
                    return "assistant";

                case Conversation::Role::User:
                case Conversation::Role::Tool:
                    break;
            }

            return "user";
        }

        /// Absent when none are offered. An empty array is offered and renders the ipython environment, as the template does.
        static std::optional<nlohmann::ordered_json> parseTools( const std::string& tools_json )
        {
            if ( detail::trim( tools_json ).empty() )
            {
                return std::nullopt;
            }

            nlohmann::ordered_json tools = nlohmann::ordered_json::parse( tools_json, nullptr, false );

            if ( tools.is_discarded() || !tools.is_array() )
            {
                throw std::invalid_argument( "Llama chat template: tools must be a JSON array" );
            }

            return tools;
        }

        static void appendTools( std::string& prompt, const nlohmann::ordered_json& tools )
        {
            for ( const auto& tool : tools )
            {
                prompt += Conversation::templateJson( tool, 4 );
                prompt += "\n\n";
            }
        }

        static std::string formatCall( std::span<const Conversation::ToolCall> calls )
        {
            if ( calls.size() != 1 )
            {
                throw std::invalid_argument( "Llama chat template: this model only supports single tool-calls at once" );
            }

            const auto& call = calls.front();
            const nlohmann::ordered_json arguments = call.arguments.empty()
                ? nlohmann::ordered_json::object()
                : nlohmann::ordered_json::parse( call.arguments, nullptr, false );

            if ( arguments.is_discarded() || !arguments.is_object() )
            {
                throw std::invalid_argument( std::format(
                    "Llama chat template: the arguments of call '{}' are not a JSON object", call.name ) );
            }

            return std::format( "{{\"name\": \"{}\", \"parameters\": {}}}", call.name, Conversation::templateJson( arguments ) );
        }

        static std::optional<Conversation::ToolCall> parseCall( std::string_view text )
        {
            const nlohmann::ordered_json call = nlohmann::ordered_json::parse( text, nullptr, false );

            if ( call.is_discarded() || !call.is_object() )
            {
                return std::nullopt;
            }

            const auto name = call.find( "name" );

            if ( name == call.end() || !name->is_string() || name->get<std::string>().empty() )
            {
                return std::nullopt;
            }

            auto arguments = call.find( "parameters" );

            if ( arguments == call.end() )
            {
                arguments = call.find( "arguments" );
            }

            if ( arguments == call.end() || !arguments->is_object() )
            {
                return std::nullopt;
            }

            return Conversation::ToolCall{ name->get<std::string>(), arguments->dump() };
        }
    };
}
