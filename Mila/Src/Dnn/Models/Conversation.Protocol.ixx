/**
 * @file Conversation.Protocol.ixx
 * @brief A chat template's grammar behind one interface: render a conversation, read a reply.
 */

export module Dnn.Models.Conversation.Protocol;

import std;
export import Dnn.Models.Conversation.Turn;
export import Dnn.Models.Conversation.TemplateVariable;
export import Dnn.Models.Conversation.PromptOptions;
export import Dnn.Models.Conversation.Prompt;
export import Dnn.Models.Conversation.Reply;
export import Dnn.Models.Conversation.ReplyGrammar;
export import Dnn.Models.Conversation.ReplyReader;
export import Dnn.Models.Conversation.ProtocolMarkers;

namespace Mila::Dnn::Conversation
{
    /**
     * @brief The grammar of one checkpoint's chat template.
     *
     * Implemented once per template, not per family: the template ships with the weights and its
     * tool grammar is part of it, so it is not stable within a family. Each implementation is
     * proven byte-identical to the checkpoint's own chat_template.jinja (ModelHandle.md P1).
     *
     * Virtual because a consumer holds the one chosen for the model it loaded at run time; the call
     * is once per turn, never per token.
     */
    export class Protocol
    {
    public:

        virtual ~Protocol() = default;

        /**
         * @brief Render a conversation and the tools offered as the template's prompt, primed for generation.
         *
         * @throws std::invalid_argument for a template variable the template does not accept, or a
         *         history the template cannot render -- each implementation names its own.
         */
        virtual Prompt formatPrompt( std::span<const Turn> history, const PromptOptions& options ) const = 0;

        /// How the template's replies are written, which every reading of a reply follows.
        virtual const ReplyGrammar& replyGrammar() const noexcept = 0;

        /**
         * @brief A reader for one reply as it streams: its reasoning, its answer and its calls as they arrive.
         *
         * @param prompt The prompt the reply continues, which says whether it begins inside the reasoning
         *        channel. The reader refers to this protocol's grammar, so the protocol must outlive it.
         */
        ReplyReader readReply( const Prompt& prompt ) const
        {
            return ReplyReader( replyGrammar(), prompt );
        }

        /**
         * @brief Parse a finished reply into its reasoning, its content and every call it made.
         *
         * The reply read whole by the reader that reads it streamed, so the two never differ.
         *
         * @param text The reply as generated, its grammar's markers included.
         * @param prompt The prompt the reply continues. Never throws for a malformed call
         *        (Reply::malformed_tool_calls).
         */
        Reply parseReply( std::string_view text, const Prompt& prompt ) const
        {
            ReplyReader reader = readReply( prompt );
            reader.feed( text );
            reader.finish();

            return reader.reply();
        }

        /**
         * @brief One tool result as the template's result span, without the turn around it.
         *
         * What continues a turn after a call, for a consumer that appends to a live cache rather
         * than re-rendering the conversation.
         */
        virtual std::string formatToolResult( std::string_view name, std::string_view result ) const = 0;

        /**
         * @brief What the template writes after a reply that ended in calls, through where the model resumes.
         *
         * Begins with the marker the reply stopped at and ends where formatPrompt leaves a history that ends on
         * these results, so a reply as the template writes it followed by this is that rendering. A consumer that
         * holds the conversation's tokens appends it to the reply as generated instead of rendering again.
         *
         * @param results The Tool turns that answer the reply's calls, in order.
         * @param options As for formatPrompt; they decide how the model's next reply opens.
         * @throws std::invalid_argument for results the template cannot render after a reply -- each implementation
         *         names its own -- and as formatPrompt for the options.
         */
        virtual Prompt continueWithToolResults( std::span<const Turn> results, const PromptOptions& options ) const = 0;

        virtual const ProtocolMarkers& markers() const noexcept = 0;

        /// The template's own variables, each with the values it accepts and its default. Empty when it takes none.
        virtual std::span<const TemplateVariable> templateVariables() const noexcept = 0;

        /**
         * @brief Whether a system turn after the first renders where it stands.
         *
         * False for a template that accepts one system turn, first, and refuses any other: formatPrompt
         * throws for it, and a consumer whose API places system text later carries that text another way.
         */
        virtual bool rendersLaterSystemTurns() const noexcept = 0;
    };
}
