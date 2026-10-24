/**
 * @file Chat.StreamingChannelRouter.ixx
 * @brief Routes each token of a streamed Gemma reply to its channel, as the session's events.
 */

export module Chat.StreamingChannelRouter;

import std;
import Chat.ChannelParser;
import Chat.GemmaStreamTokens;
import Chat.SessionEvents;
import Chat.StreamingDetokenizer;
import Chat.TextChannel;

namespace Mila::ChatApp
{
    /**
     * @brief A small state machine over the Gemma 4 channel protocol.
     *
     * Answer text and reasoning go out as text on their channels, and tool-call spans not at all: the session
     * dispatches them and reports each call. Every reasoning word is reported whatever the detail level, which is a
     * display's choice. The full response string keeps accumulating in the session regardless -- its post-hoc parse
     * is both the conversation history and the oracle a display checks what it streamed against.
     */
    export class StreamingChannelRouter
    {
    public:

        StreamingChannelRouter( SessionEvents& events, const GemmaStreamTokens& tokens )
            : events_( events ), tokens_( tokens )
        {
        }

        void onToken( std::int32_t token, std::string_view piece )
        {
            if ( token == tokens_.tool_call_open && state_ != State::ToolCall )
            {
                deliverText( detokenizer_.flush() );
                state_before_tool_call_ = state_;
                state_ = State::ToolCall;
                events_.toolCallOpened();

                return;
            }

            if ( token == tokens_.tool_call_close )
            {
                if ( state_ == State::ToolCall )
                    state_ = state_before_tool_call_;

                return;
            }

            if ( state_ == State::ToolCall )
                return;

            if ( token == tokens_.channel_open && state_ == State::Answer )
            {
                deliverText( detokenizer_.flush() );
                state_ = State::ChannelHeader;
                header_.clear();

                return;
            }

            if ( token == tokens_.channel_close && state_ != State::Answer )
            {
                deliverText( detokenizer_.flush() );
                closeChannel();

                return;
            }

            // Note the two guards above: an open marker inside a reasoning region
            // and a close marker with no open cycle are literal text to the
            // post-hoc ChannelParser, so they stream as literal text here too.

            if ( isSuppressed( token ) )
            {
                deliverText( detokenizer_.flush() );

                return;
            }

            deliverText( detokenizer_.feed( piece ) );
        }

        /// A round whose prompt left the reasoning channel open, its label already written: what the model writes
        /// next is reasoning, until it closes the channel.
        void beginRoundInReasoning()
        {
            deliverText( detokenizer_.flush() );
            header_.clear();
            beginThoughtBody();
        }

        /// Round boundary: surrender any incomplete UTF-8 tail (no next token in this round will complete it).
        void onRoundEnd()
        {
            deliverText( detokenizer_.flush() );
        }

        /// Turn end: deliver what is still held.
        void finish()
        {
            deliverText( detokenizer_.flush() );

            if ( state_ == State::ChannelHeader )
                beginThoughtBody();  // generation stopped inside the header
        }

    private:

        enum class State
        {
            Answer,         ///< Default: displayable answer text.
            ChannelHeader,  ///< After <|channel>: withholding the label line.
            Thought,        ///< Reasoning region until <channel|>.
            ToolCall,       ///< Suppressed span until <tool_call|>.
        };

        bool isSuppressed( std::int32_t token ) const
        {
            for ( const std::int32_t suppressed : tokens_.suppressed )
            {
                if ( token == suppressed )
                    return true;
            }

            return false;
        }

        void deliverText( std::string_view text )
        {
            if ( text.empty() )
                return;

            switch ( state_ )
            {
                case State::Answer:
                    events_.text( TextChannel::Answer, text );
                    break;

                case State::Thought:
                    events_.text( TextChannel::Thought, text );
                    break;

                case State::ChannelHeader:
                {
                    header_ += text;

                    // Withhold the header until its first content line is complete,
                    // mirroring ChannelParser::stripChannelLabel.
                    const auto content = header_.find_first_not_of( " \t\r\n" );

                    if ( content == std::string::npos )
                        break;

                    if ( header_.find( '\n', content ) == std::string::npos )
                        break;

                    beginThoughtBody();
                    break;
                }

                case State::ToolCall:
                    break;
            }
        }

        /**
         * @brief Leave ChannelHeader for Thought, delivering any non-label header text.
         *
         * A recognized label line (thought/thinking/...) is dropped with its
         * newline; anything else is reasoning content -- the same classification
         * the post-hoc ChannelParser applies to the region.
         */
        void beginThoughtBody()
        {
            std::string body;
            const auto content = header_.find_first_not_of( " \t\r\n" );

            if ( content != std::string::npos )
            {
                const auto newline = header_.find( '\n', content );
                const std::string_view first_line = (newline == std::string::npos)
                    ? std::string_view( header_ ).substr( content )
                    : std::string_view( header_ ).substr( content, newline - content );

                if ( ChannelParser::isChannelLabel( ChannelParser::trimView( first_line ) ) )
                    body = (newline == std::string::npos) ? std::string{} : header_.substr( newline + 1 );
                else
                    body = header_;
            }

            header_.clear();
            state_ = State::Thought;

            events_.thoughtBreak();

            if ( !body.empty() )
                events_.text( TextChannel::Thought, body );
        }

        /// <channel|>: finish the reasoning region (classifying a still-open
        /// header first) and return to the answer channel.
        void closeChannel()
        {
            if ( state_ == State::ChannelHeader )
                beginThoughtBody();

            if ( state_ == State::Thought )
            {
                events_.thoughtClosed();
                state_ = State::Answer;
            }
        }

        SessionEvents& events_;
        GemmaStreamTokens tokens_;

        StreamingDetokenizer detokenizer_;

        State state_ = State::Answer;
        State state_before_tool_call_ = State::Answer;
        std::string header_;
    };
}
