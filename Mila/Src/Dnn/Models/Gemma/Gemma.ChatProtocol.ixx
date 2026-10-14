/**
 * @file Gemma.ChatProtocol.ixx
 * @brief Gemma 4's chat template behind the Protocol interface, rendered exactly as the checkpoint ships it.
 *
 * Ported from google/gemma-4-12b-it's chat_template.jinja -- the 12B, its QAT build and the 26B-A4B
 * ship the same file -- and held to it byte for byte by goldens that
 * Tools/ChatTemplates/record_template_goldens.py records through transformers.
 */

export module Dnn.Models.GemmaChatProtocol;

import std;
import Dnn.Models.Conversation;
import Dnn.Models.GemmaProtocol;

import nlohmann.json;

namespace Mila::Dnn::Gemma
{
    namespace detail
    {
        inline constexpr std::string_view kTemplateWhitespace = " \t\n\r\v\f";

        /// Jinja's `trim`, which is Python's str.strip().
        inline std::string_view templateTrim( std::string_view text )
        {
            const auto first = text.find_first_not_of( kTemplateWhitespace );

            if ( first == std::string_view::npos )
            {
                return {};
            }

            return text.substr( first, text.find_last_not_of( kTemplateWhitespace ) - first + 1 );
        }

        /// The template's strip_thinking: what is outside every channel span, trimmed.
        inline std::string stripThinking( std::string_view text )
        {
            std::string result;
            std::size_t cursor = 0;

            while ( true )
            {
                const auto close = text.find( kChannelClose, cursor );
                const auto part = text.substr( cursor, close == std::string_view::npos ? std::string_view::npos : close - cursor );
                const auto open = part.find( kChannelOpen );

                result += open == std::string_view::npos ? part : part.substr( 0, open );

                if ( close == std::string_view::npos )
                {
                    break;
                }

                cursor = close + kChannelClose.size();
            }

            return std::string( templateTrim( result ) );
        }

        /// A channel body without the label line the model opens it with ("thought\n").
        inline std::string_view channelBody( std::string_view body )
        {
            const auto newline = body.find( '\n' );

            if ( newline != std::string_view::npos )
            {
                const auto label = body.substr( 0, newline );

                if ( std::ranges::find( kChannelLabels, label ) != std::end( kChannelLabels ) )
                {
                    return body.substr( newline + 1 );
                }
            }

            return body;
        }
    }

    /**
     * @brief The grammar of Gemma 4's chat template.
     *
     * Its variables are the template's own: enable_thinking and preserve_thinking, both false by
     * default. A call and its results share one model turn, so consecutive Tool turns after an
     * Assistant turn with calls render inside it, and a history ending on them resumes that turn.
     */
    export class GemmaChatProtocol final : public Conversation::Protocol
    {
    public:

        GemmaChatProtocol()
            : variables_{
                { "enable_thinking", { "false", "true" }, "false" },
                { "preserve_thinking", { "false", "true" }, "false" } },
            markers_{
                std::string( kChannelOpen ), std::string( kChannelClose ),
                std::string( kToolCallOpen ), std::string( kToolCallClose ), std::string( kToolResponseOpen ),
                { std::string( kToolCallClose ), std::string( kToolResponseOpen ) } }
        {
        }

        /**
         * @throws std::invalid_argument for a template variable the template does not accept, a Tool
         *         turn that does not follow an Assistant turn with calls or names no tool, or call
         *         arguments that are not a JSON object. The template renders an unpaired result
         *         nowhere and names an unnamed one `unknown`; both are refused rather than rendered
         *         as a prompt that looks valid.
         */
        Conversation::Prompt formatPrompt(
            std::span<const Conversation::Turn> history, const Conversation::PromptOptions& options ) const override
        {
            Conversation::validateTemplateVariables( variables_, options.template_variables );
            validateToolTurns( history );

            const bool thinking = variable( options, "enable_thinking" ) == "true";
            const bool preserve_thinking = variable( options, "preserve_thinking" ) == "true";
            const std::string declarations = serializeToolDeclarations( options.tools_json );
            const bool first_is_system = !history.empty() && history.front().role == Conversation::Role::System;

            enum class Previous { None, Think, Tool, ToolCall, ToolResponse };

            Previous previous = Previous::None;
            std::string prompt( kBos );

            if ( thinking || !declarations.empty() || first_is_system )
            {
                prompt += kTurnOpen;
                prompt += "system\n";

                if ( thinking )
                {
                    prompt += kThink;
                    prompt += "\n";
                    previous = Previous::Think;
                }

                if ( first_is_system )
                {
                    prompt += detail::templateTrim( history.front().content );
                }

                if ( !declarations.empty() )
                {
                    prompt += declarations;
                    previous = Previous::Tool;
                }

                prompt += kTurnClose;
                prompt += "\n";
            }

            const auto turns = first_is_system ? history.subspan( 1 ) : history;

            std::optional<std::size_t> last_user;

            for ( std::size_t index = 0; index < turns.size(); ++index )
            {
                if ( turns[ index ].role == Conversation::Role::User )
                {
                    last_user = index;
                }
            }

            std::optional<Conversation::Role> previous_role;

            for ( std::size_t index = 0; index < turns.size(); ++index )
            {
                const auto& turn = turns[ index ];

                if ( turn.role == Conversation::Role::Tool )
                {
                    continue;
                }

                previous = Previous::None;

                const bool model = turn.role == Conversation::Role::Assistant;

                if ( !( model && previous_role == Conversation::Role::Assistant ) )
                {
                    prompt += kTurnOpen;
                    prompt += roleSpelling( turn.role );
                    prompt += "\n";
                }

                const bool after_last_user = !last_user.has_value() || index > *last_user;

                if ( !turn.reasoning.empty() && ( after_last_user || ( preserve_thinking && !turn.tool_calls.empty() ) ) )
                {
                    prompt += std::format( "{}thought\n{}\n{}", kChannelOpen, turn.reasoning, kChannelClose );
                }

                for ( const auto& call : turn.tool_calls )
                {
                    prompt += formatCall( call );
                    previous = Previous::ToolCall;
                }

                bool results = false;

                if ( !turn.tool_calls.empty() )
                {
                    for ( std::size_t next = index + 1; next < turns.size() && turns[ next ].role == Conversation::Role::Tool; ++next )
                    {
                        prompt += formatToolResponseText( turns[ next ].tool_name, turns[ next ].content );
                        results = true;
                        previous = Previous::ToolResponse;
                    }
                }

                const std::string content = model ? detail::stripThinking( turn.content ) : std::string( detail::templateTrim( turn.content ) );
                prompt += content;

                std::optional<Conversation::Role> next_role;

                for ( std::size_t next = index + 1; next < turns.size(); ++next )
                {
                    if ( turns[ next ].role != Conversation::Role::Tool )
                    {
                        next_role = turns[ next ].role;
                        break;
                    }
                }

                const bool continues_into_next = model && next_role == Conversation::Role::Assistant
                    && ( turn.tool_calls.empty() || results );

                // A turn that ends on its results with nothing after them stays open: that is the resume.
                const bool resumes = results && detail::templateTrim( content ).empty() && !next_role.has_value();

                if ( previous == Previous::ToolCall && !results )
                {
                    prompt += kToolResponseOpen;
                } else if ( !continues_into_next && !resumes ) {
                    prompt += kTurnClose;
                    prompt += "\n";
                }

                previous_role = turn.role;
            }

            bool begins_in_reasoning = false;

            if ( previous != Previous::ToolResponse && previous != Previous::ToolCall )
            {
                prompt += kTurnOpen;
                prompt += "model\n";

                if ( !thinking )
                {
                    prompt += kThoughtPrime;
                }
            } else if ( previous == Previous::ToolResponse && thinking ) {
                prompt += kChannelOpen;
                prompt += "thought\n";
                begins_in_reasoning = true;
            }

            return { std::move( prompt ), begins_in_reasoning };
        }

        Conversation::Reply parseReply( std::string_view text, const Conversation::Prompt& prompt ) const override
        {
            Conversation::Reply reply;
            std::string_view rest = text;

            if ( prompt.reply_begins_in_reasoning )
            {
                const auto close = rest.find( kChannelClose );

                appendReasoning( reply, rest.substr( 0, close ) );
                rest = close == std::string_view::npos ? std::string_view{} : rest.substr( close + kChannelClose.size() );
            }

            std::string content;

            while ( !rest.empty() )
            {
                const auto channel = rest.find( kChannelOpen );
                const auto call = rest.find( kToolCallOpen );
                const auto response = rest.find( kToolResponseOpen );
                const auto next = std::min( { channel, call, response } );

                if ( next == std::string_view::npos )
                {
                    content += rest;
                    break;
                }

                content += rest.substr( 0, next );
                rest.remove_prefix( next );

                if ( next == channel )
                {
                    const auto close = rest.find( kChannelClose );

                    appendReasoning( reply, detail::channelBody(
                        rest.substr( kChannelOpen.size(), close == std::string_view::npos ? std::string_view::npos : close - kChannelOpen.size() ) ) );
                    rest = close == std::string_view::npos ? std::string_view{} : rest.substr( close + kChannelClose.size() );
                } else if ( next == call ) {
                    const auto close = rest.find( kToolCallClose );

                    if ( close == std::string_view::npos )
                    {
                        reply.malformed_tool_calls.emplace_back( rest );
                        break;
                    }

                    const auto span = rest.substr( 0, close + kToolCallClose.size() );
                    const auto parsed = parseToolCall( span );

                    if ( parsed.has_value() )
                    {
                        reply.tool_calls.push_back( *parsed );
                    } else {
                        reply.malformed_tool_calls.emplace_back( span );
                    }

                    rest.remove_prefix( span.size() );
                } else {
                    // A result is the engine's to write: one the model wrote is not part of its answer.
                    const auto close = rest.find( kToolResponseClose );
                    rest = close == std::string_view::npos ? std::string_view{} : rest.substr( close + kToolResponseClose.size() );
                }
            }

            reply.content = detail::templateTrim( stripControlTokens( content ) );

            return reply;
        }

        std::string formatToolResult( std::string_view name, std::string_view result ) const override
        {
            return formatToolResponseText( name, result );
        }

        const Conversation::ProtocolMarkers& markers() const noexcept override
        {
            return markers_;
        }

        std::span<const Conversation::TemplateVariable> templateVariables() const noexcept override
        {
            return variables_;
        }

    private:

        std::vector<Conversation::TemplateVariable> variables_;
        Conversation::ProtocolMarkers markers_;

        std::string_view variable( const Conversation::PromptOptions& options, std::string_view name ) const
        {
            return Conversation::resolveTemplateVariable( variables_, options.template_variables, name );
        }

        static void appendReasoning( Conversation::Reply& reply, std::string_view body )
        {
            const auto text = detail::templateTrim( body );

            if ( text.empty() )
            {
                return;
            }

            reply.reasoning += reply.reasoning.empty() ? "" : "\n";
            reply.reasoning += text;
        }

        /// The template requires arguments to be a mapping; it renders keys sorted (dictsort).
        static std::string formatCall( const Conversation::ToolCall& call )
        {
            if ( call.arguments.empty() )
            {
                return std::format( "{}call:{}{{}}{}", kToolCallOpen, call.name, kToolCallClose );
            }

            const nlohmann::json arguments = nlohmann::json::parse( call.arguments, nullptr, false );

            if ( arguments.is_discarded() || !arguments.is_object() )
            {
                throw std::invalid_argument( std::format(
                    "Gemma chat template: the arguments of call '{}' are not a JSON object", call.name ) );
            }

            return formatToolCall( call.name, call.arguments );
        }

        static void validateToolTurns( std::span<const Conversation::Turn> history )
        {
            for ( std::size_t index = 0; index < history.size(); ++index )
            {
                if ( history[ index ].role != Conversation::Role::Tool )
                {
                    continue;
                }

                std::size_t owner = index;

                while ( owner > 0 && history[ owner - 1 ].role == Conversation::Role::Tool )
                {
                    --owner;
                }

                if ( owner == 0 || history[ owner - 1 ].role != Conversation::Role::Assistant
                    || history[ owner - 1 ].tool_calls.empty() )
                {
                    throw std::invalid_argument(
                        "Gemma chat template: a tool result must follow an assistant turn that made calls" );
                }

                if ( history[ index ].tool_name.empty() )
                {
                    throw std::invalid_argument( "Gemma chat template: a tool result must name its tool" );
                }
            }
        }
    };
}
