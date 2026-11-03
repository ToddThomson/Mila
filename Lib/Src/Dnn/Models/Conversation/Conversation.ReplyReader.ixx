/**
 * @file Conversation.ReplyReader.ixx
 * @brief Reads a reply as it streams, by its template's grammar: reasoning, answer and every call.
 */

export module Dnn.Models.Conversation.ReplyReader;

import std;
export import Dnn.Models.Conversation.Prompt;
export import Dnn.Models.Conversation.Reply;
export import Dnn.Models.Conversation.ReplyEvent;
export import Dnn.Models.Conversation.ReplyGrammar;

namespace Mila::Dnn::Conversation
{
    /**
     * @brief Reads one reply, fed in pieces as it is generated, into events and a Reply.
     *
     * The same reader reads a finished reply (Protocol::parseReply feeds it whole), so a reply streamed and a
     * reply parsed cannot differ: the events' text is the Reply's text, and how the reply was cut into pieces
     * changes neither. Text that may still become a marker, and an incomplete UTF-8 character, are held until
     * the next piece settles them. Reasoning and content are trimmed as a whole: leading whitespace is never
     * sent and trailing whitespace is held until text follows it.
     *
     * Holds a reference to its grammar, which must outlive it.
     */
    export class ReplyReader
    {
    public:

        ReplyReader( const ReplyGrammar& grammar, const Prompt& prompt )
            : grammar_( &grammar )
        {
            if ( prompt.reply_begins_in_reasoning )
            {
                state_ = State::Reasoning;
                at_reply_start_ = false;
                pushEvent( ReplyEvent::Kind::ReasoningOpened );

                // Delivered with what the first piece settles.
                events_delivered_ = false;
            }
        }

        /**
         * @brief Read the next piece of the reply.
         *
         * @return What the piece settled, valid until the reader is next fed.
         * @throws std::logic_error after finish().
         */
        std::span<const ReplyEvent> feed( std::string_view piece )
        {
            if ( finished_ )
            {
                throw std::logic_error( "ReplyReader: fed after the reply finished" );
            }

            beginEvents();
            pending_ += piece;
            process( false );

            return endEvents();
        }

        /**
         * @brief The reply ended: settle what is held. An unclosed call is read as the model left it.
         *
         * @return What ending settled, valid until the reader is destroyed.
         */
        std::span<const ReplyEvent> finish()
        {
            if ( finished_ )
            {
                return {};
            }

            beginEvents();
            process( true );

            if ( state_ == State::ReasoningHeader )
            {
                emitReasoning( header_ );
                header_.clear();
            }

            if ( state_ == State::Call )
            {
                finishCall();
            }

            finished_ = true;

            return endEvents();
        }

        /// The reply read so far: complete once finish() has been called, and holding every call closed before it.
        const Reply& reply() const noexcept
        {
            return reply_;
        }

    private:

        enum class State
        {
            Answer,
            ReasoningHeader,
            Reasoning,
            Call,
            Response,
            Ended,
        };

        enum class Rule
        {
            ReasoningOpen,
            ReasoningClose,
            ToolCallOpen,
            ToolCallClose,
            ToolResponseOpen,
            ToolResponseClose,
            ReplyEnd,
            Hidden,
            PipeToken,
        };

        struct Match
        {
            std::size_t position{ std::string::npos };
            std::size_t length{ 0 };
            Rule rule{ Rule::Hidden };

            /// The text from position on may still become this rule, or another, once more arrives.
            bool partial{ false };
        };

        static constexpr std::string_view kWhitespace = " \t\n\r\v\f";

        const ReplyGrammar* grammar_;
        State state_{ State::Answer };
        std::string pending_;
        Reply reply_;

        bool at_reply_start_{ true };
        bool answer_started_{ false };
        std::string answer_whitespace_;

        bool span_started_{ false };
        std::string reasoning_whitespace_;
        std::string header_;

        std::string call_span_;
        bool finished_{ false };

        struct PendingEvent
        {
            ReplyEvent::Kind kind;
            std::size_t offset;
            std::size_t length;
        };

        std::vector<PendingEvent> pending_events_;
        std::string event_text_;
        std::vector<ReplyEvent> events_;
        bool events_delivered_{ true };

        void process( bool final )
        {
            while ( !pending_.empty() )
            {
                const bool progressed = [&]
                    {
                        switch ( state_ )
                        {
                            case State::Answer:
                                return stepAnswer( final );

                            case State::ReasoningHeader:
                                return stepReasoningHeader( final );

                            case State::Reasoning:
                                return stepReasoning( final );

                            case State::Call:
                                return stepCall( final );

                            case State::Response:
                                return stepResponse( final );

                            case State::Ended:
                                pending_.clear();
                                return true;
                        }

                        return false;
                    }();

                if ( !progressed )
                {
                    return;
                }
            }
        }

        bool stepAnswer( bool final )
        {
            const ReplyGrammar& grammar = *grammar_;

            if ( at_reply_start_ )
            {
                const auto first = pending_.find_first_not_of( kWhitespace );

                if ( first == std::string::npos )
                {
                    // Whitespace opening the answer is never sent.
                    pending_.clear();

                    return true;
                }

                pending_.erase( 0, first );

                const std::string_view bare = grammar.bare_tool_call_open;

                if ( !bare.empty() )
                {
                    if ( pending_.size() < bare.size() && bare.starts_with( pending_ ) && !final )
                    {
                        return false;
                    }

                    if ( pending_.starts_with( bare ) )
                    {
                        at_reply_start_ = false;
                        openCall();

                        return true;
                    }
                }

                at_reply_start_ = false;
            }

            const Match match = findFirst( final, {
                { grammar.reasoning_open, Rule::ReasoningOpen },
                { grammar.tool_call_open, Rule::ToolCallOpen },
                { grammar.tool_response_open, Rule::ToolResponseOpen } },
                true, true );

            if ( !commitBefore( match, final, [this]( std::string_view text ) { emitAnswer( text ); } ) )
            {
                return false;
            }

            consume( match.length );

            switch ( match.rule )
            {
                case Rule::ReasoningOpen:
                    span_started_ = false;
                    state_ = grammar.reasoning_labels.empty() ? State::Reasoning : State::ReasoningHeader;
                    pushEvent( ReplyEvent::Kind::ReasoningOpened );
                    break;

                case Rule::ToolCallOpen:
                    openCall();
                    call_span_ = grammar.tool_call_open;
                    break;

                case Rule::ToolResponseOpen:
                    state_ = State::Response;
                    break;

                case Rule::ReplyEnd:
                    state_ = State::Ended;
                    break;

                default:
                    // Control text: not part of the answer.
                    break;
            }

            return true;
        }

        bool stepReasoningHeader( bool final )
        {
            const Match close = findFirst( final, { { grammar_->reasoning_close, Rule::ReasoningClose } }, true, false );
            const std::size_t limit = close.position == std::string::npos ? pending_.size() : close.position;
            const auto newline = pending_.find( '\n' );

            if ( newline != std::string::npos && newline < limit )
            {
                header_.append( pending_, 0, newline );
                consume( newline + 1 );

                if ( std::ranges::find( grammar_->reasoning_labels, header_ ) == grammar_->reasoning_labels.end() )
                {
                    header_ += '\n';
                    emitReasoning( header_ );
                }

                header_.clear();
                state_ = State::Reasoning;

                return true;
            }

            if ( close.position != std::string::npos && !close.partial )
            {
                // The span closed on its first line: all of it is reasoning.
                header_.append( pending_, 0, close.position );
                consume( close.position );
                emitReasoning( header_ );
                header_.clear();
                state_ = State::Reasoning;

                return true;
            }

            header_.append( pending_, 0, limit );
            consume( limit );

            return false;
        }

        bool stepReasoning( bool final )
        {
            const Match match = findFirst( final, { { grammar_->reasoning_close, Rule::ReasoningClose } }, true, false );

            if ( !commitBefore( match, final, [this]( std::string_view text ) { emitReasoning( text ); } ) )
            {
                return false;
            }

            consume( match.length );

            if ( match.rule == Rule::ReplyEnd )
            {
                state_ = State::Ended;

                return true;
            }

            reasoning_whitespace_.clear();
            state_ = State::Answer;
            pushEvent( ReplyEvent::Kind::ReasoningClosed );

            return true;
        }

        bool stepCall( bool final )
        {
            const Match match = findFirst( final, { { grammar_->tool_call_close, Rule::ToolCallClose } }, true, false );

            if ( match.position == std::string::npos )
            {
                call_span_ += pending_;
                pending_.clear();

                return true;
            }

            call_span_.append( pending_, 0, match.position );
            consume( match.position );

            if ( match.partial )
            {
                return false;
            }

            if ( match.rule == Rule::ToolCallClose )
            {
                call_span_.append( pending_, 0, match.length );
                consume( match.length );
                finishCall();
                state_ = State::Answer;

                return true;
            }

            consume( match.length );
            finishCall();
            state_ = State::Ended;

            return true;
        }

        bool stepResponse( bool final )
        {
            const Match match = findFirst( final, { { grammar_->tool_response_close, Rule::ToolResponseClose } }, true, false );

            if ( match.position == std::string::npos )
            {
                pending_.clear();

                return true;
            }

            consume( match.position );

            if ( match.partial )
            {
                return false;
            }

            consume( match.length );
            state_ = match.rule == Rule::ReplyEnd ? State::Ended : State::Answer;

            return true;
        }

        void openCall()
        {
            call_span_.clear();
            state_ = State::Call;
            pushEvent( ReplyEvent::Kind::ToolCallOpened );
        }

        void finishCall()
        {
            const auto last = call_span_.find_last_not_of( kWhitespace );
            const std::string_view span = last == std::string::npos
                ? std::string_view{} : std::string_view( call_span_ ).substr( 0, last + 1 );

            std::optional<ToolCall> call;

            if ( grammar_->read_tool_call != nullptr )
            {
                call = grammar_->read_tool_call( span );
            }

            if ( call.has_value() )
            {
                reply_.tool_calls.push_back( std::move( *call ) );
            } else {
                reply_.malformed_tool_calls.emplace_back( span );
            }

            call_span_.clear();
            pushEvent( ReplyEvent::Kind::ToolCallClosed );
        }

        /**
         * @brief Send the text before a match, or hold what may still become one.
         *
         * @return False when the reader needs more of the reply to go on.
         */
        template<typename TEmit>
        bool commitBefore( const Match& match, bool final, TEmit&& emit )
        {
            if ( match.position == std::string::npos )
            {
                const std::size_t complete = final ? pending_.size() : completeUtf8Length( pending_ );

                emit( std::string_view( pending_ ).substr( 0, complete ) );
                consume( complete );

                return false;
            }

            emit( std::string_view( pending_ ).substr( 0, match.position ) );
            consume( match.position );

            return !match.partial;
        }

        /**
         * @brief The first place in what is pending where a rule matches, or may once more arrives.
         *
         * Rules are tried in the order given, after them the reply's end and, where asked, the control text.
         * At one position a rule that may still match outranks a later one that already does, so that what
         * is read does not depend on where the reply was cut.
         */
        Match findFirst( bool final, std::initializer_list<std::pair<std::string_view, Rule>> markers,
            bool include_reply_end, bool include_hidden ) const
        {
            const ReplyGrammar& grammar = *grammar_;

            for ( std::size_t position = 0; position < pending_.size(); ++position )
            {
                const std::string_view rest = std::string_view( pending_ ).substr( position );
                std::optional<Match> found;

                const auto consider = [&]( std::string_view marker, Rule rule ) -> bool
                    {
                        if ( marker.empty() )
                        {
                            return false;
                        }

                        if ( rest.starts_with( marker ) )
                        {
                            found = Match{ position, marker.size(), rule, false };

                            return true;
                        }

                        if ( !final && rest.size() < marker.size() && marker.starts_with( rest ) )
                        {
                            found = Match{ position, 0, rule, true };

                            return true;
                        }

                        return false;
                    };

                bool settled = false;

                for ( const auto& [marker, rule] : markers )
                {
                    if ( consider( marker, rule ) )
                    {
                        settled = true;
                        break;
                    }
                }

                if ( !settled && include_reply_end )
                {
                    for ( const auto& marker : grammar.reply_end )
                    {
                        if ( consider( marker, Rule::ReplyEnd ) )
                        {
                            settled = true;
                            break;
                        }
                    }
                }

                if ( !settled && include_hidden )
                {
                    for ( const auto& marker : grammar.hidden )
                    {
                        if ( consider( marker, Rule::Hidden ) )
                        {
                            settled = true;
                            break;
                        }
                    }

                    if ( !settled && grammar.hides_pipe_tokens )
                    {
                        found = pipeToken( rest, position, final );
                        settled = found.has_value();
                    }
                }

                if ( settled )
                {
                    return *found;
                }
            }

            return {};
        }

        /// `<|name|>` with no `|` or `>` inside the name, or the bare `<|>`.
        static std::optional<Match> pipeToken( std::string_view rest, std::size_t position, bool final )
        {
            if ( rest.front() != '<' )
            {
                return std::nullopt;
            }

            if ( rest.size() == 1 )
            {
                return final ? std::nullopt : std::optional<Match>( Match{ position, 0, Rule::PipeToken, true } );
            }

            if ( rest[ 1 ] != '|' )
            {
                return std::nullopt;
            }

            std::size_t scan = 2;

            while ( scan < rest.size() && rest[ scan ] != '|' && rest[ scan ] != '>' )
            {
                ++scan;
            }

            const auto undecided = [&]
                {
                    return final ? std::nullopt : std::optional<Match>( Match{ position, 0, Rule::PipeToken, true } );
                };

            if ( scan == rest.size() )
            {
                return undecided();
            }

            if ( rest[ scan ] == '>' )
            {
                return scan == 2 ? std::optional<Match>( Match{ position, 3, Rule::PipeToken, false } ) : std::nullopt;
            }

            if ( scan + 1 == rest.size() )
            {
                return undecided();
            }

            return rest[ scan + 1 ] == '>'
                ? std::optional<Match>( Match{ position, scan + 2, Rule::PipeToken, false } ) : std::nullopt;
        }

        void emitAnswer( std::string_view text )
        {
            if ( !answer_started_ )
            {
                const auto first = text.find_first_not_of( kWhitespace );

                if ( first == std::string_view::npos )
                {
                    return;
                }

                text.remove_prefix( first );
                answer_started_ = true;
            }

            emitTrimmed( text, answer_whitespace_, reply_.content, ReplyEvent::Kind::Answer, {} );
        }

        void emitReasoning( std::string_view text )
        {
            std::string_view separator;

            if ( !span_started_ )
            {
                const auto first = text.find_first_not_of( kWhitespace );

                if ( first == std::string_view::npos )
                {
                    return;
                }

                text.remove_prefix( first );
                span_started_ = true;

                if ( !reply_.reasoning.empty() )
                {
                    separator = grammar_->reasoning_separator;
                }
            }

            emitTrimmed( text, reasoning_whitespace_, reply_.reasoning, ReplyEvent::Kind::Reasoning, separator );
        }

        /// Send text with the whitespace held before it, holding its own trailing whitespace back.
        void emitTrimmed( std::string_view text, std::string& held, std::string& into, ReplyEvent::Kind kind,
            std::string_view separator )
        {
            const auto last = text.find_last_not_of( kWhitespace );

            if ( last == std::string_view::npos )
            {
                held += text;

                return;
            }

            std::string sent( separator );
            sent += held;
            sent += text.substr( 0, last + 1 );
            held.assign( text.substr( last + 1 ) );

            into += sent;
            pushEvent( kind, sent );
        }

        void consume( std::size_t count )
        {
            pending_.erase( 0, count );
        }

        void beginEvents()
        {
            if ( events_delivered_ )
            {
                pending_events_.clear();
                event_text_.clear();
                events_delivered_ = false;
            }
        }

        void pushEvent( ReplyEvent::Kind kind, std::string_view text = {} )
        {
            const bool carries_text = kind == ReplyEvent::Kind::Reasoning || kind == ReplyEvent::Kind::Answer;

            if ( carries_text && !pending_events_.empty() && pending_events_.back().kind == kind )
            {
                event_text_ += text;
                pending_events_.back().length += text.size();

                return;
            }

            pending_events_.push_back( { kind, event_text_.size(), text.size() } );
            event_text_ += text;
        }

        std::span<const ReplyEvent> endEvents()
        {
            events_.clear();

            for ( const auto& event : pending_events_ )
            {
                events_.push_back( { event.kind, std::string_view( event_text_ ).substr( event.offset, event.length ) } );
            }

            events_delivered_ = true;

            return events_;
        }

        /// The length of the longest prefix that ends on a whole UTF-8 character.
        static std::size_t completeUtf8Length( std::string_view text )
        {
            std::size_t index = text.size();
            int continuation = 0;

            while ( index > 0 && continuation < 3 )
            {
                const auto byte = static_cast<unsigned char>( text[ index - 1 ] );

                if ( ( byte & 0x80 ) == 0 )
                {
                    return text.size();
                }

                if ( ( byte & 0xC0 ) == 0xC0 )
                {
                    const std::size_t need = ( byte & 0xE0 ) == 0xC0 ? 2 : ( byte & 0xF0 ) == 0xE0 ? 3 : ( byte & 0xF8 ) == 0xF0 ? 4 : 1;

                    return text.size() - ( index - 1 ) >= need ? text.size() : index - 1;
                }

                --index;
                ++continuation;
            }

            return text.size();
        }
    };
}
