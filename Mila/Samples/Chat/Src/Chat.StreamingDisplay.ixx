/**
 * @file Chat.StreamingDisplay.ixx
 * @brief Live token-streaming display pipeline for Gemma chat turns.
 *
 * Chains an incomplete-UTF-8 tail-holding detokenizer, a token-keyed channel
 * router over the Gemma 4 control tokens, and an incremental rich-text
 * formatter feeding ConsoleRenderer's streaming block writer. The buffered
 * response string still accumulates in the chat loop; its post-hoc render is
 * the validation oracle for what this pipeline displayed.
 */

module;
#include <cctype>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

export module Chat.StreamingDisplay;

import Chat.ChannelParser;
import Chat.Config;
import Chat.Renderer;
import Chat.RichText;

namespace Mila::ChatApp
{
    /**
     * @brief Re-assembles per-token decode bytes into complete UTF-8 characters.
     *
     * Per-token decode() can split a multi-byte UTF-8 sequence across tokens;
     * printing the raw bytes would show replacement garbage. feed() returns only
     * the prefix ending on a complete character boundary and holds the
     * incomplete tail for the next token. flush() surrenders whatever is held
     * (round end), complete or not.
     */
    export class StreamingDetokenizer
    {
    public:

        std::string feed( std::string_view bytes )
        {
            held_.append( bytes );

            const size_t complete = completeUtf8PrefixLength( held_ );
            std::string out = held_.substr( 0, complete );
            held_.erase( 0, complete );

            return out;
        }

        std::string flush()
        {
            return std::exchange( held_, std::string{} );
        }

    private:

        static size_t completeUtf8PrefixLength( std::string_view text )
        {
            size_t i = text.size();
            int continuation_bytes = 0;

            while ( i > 0 && continuation_bytes < 3 )
            {
                const unsigned char c = static_cast<unsigned char>( text[ i - 1 ] );

                if ( (c & 0x80) == 0 )
                    return text.size();  // ASCII tail: everything is complete

                if ( (c & 0xC0) == 0xC0 )
                {
                    const size_t need =
                        (c & 0xE0) == 0xC0 ? 2 :
                        (c & 0xF0) == 0xE0 ? 3 :
                        (c & 0xF8) == 0xF0 ? 4 : 1;  // invalid lead byte: emit as-is
                    const size_t have = text.size() - (i - 1);

                    return have >= need ? text.size() : i - 1;
                }

                --i;  // continuation byte: keep scanning back for its lead
                ++continuation_bytes;
            }

            // Start of buffer or 3+ trailing continuation bytes without a lead:
            // not a valid in-progress sequence, emit as-is.
            return text.size();
        }

        std::string held_;
    };

    /**
     * @brief Streams text through RichText::formatRich in chunk-equivalent pieces.
     *
     * Plain prose flows through immediately; a construct whose rendering needs
     * lookahead (a \command name or its brace groups, a trailing _/^/~, a
     * possible list marker at line start) is held until it resolves. Holds are
     * line-bounded: a newline (or an oversized buffer) forces the pending text
     * through so malformed markup cannot stall the display. Within those bounds,
     * formatting each emitted chunk independently equals formatting the whole
     * text at once — the property the stream validator checks.
     */
    export class IncrementalRichFormatter
    {
    public:

        std::string feed( std::string_view text )
        {
            pending_.append( text );

            return drain( false );
        }

        std::string flush()
        {
            return drain( true );
        }

    private:

        static constexpr size_t kMaximumHoldLength = 512;

        std::string drain( bool force )
        {
            std::string out;

            while ( !pending_.empty() )
            {
                const size_t hold = force ? std::string::npos : holdPoint( pending_, at_line_start_ );

                if ( hold == std::string::npos )
                {
                    out += formatChunk( pending_ );
                    pending_.clear();
                    break;
                }

                // A construct is never held across a display line: force the
                // whole line through rather than stalling on malformed markup.
                const size_t newline = pending_.find( '\n', hold );

                if ( newline != std::string::npos )
                {
                    out += formatChunk( pending_.substr( 0, newline + 1 ) );
                    pending_.erase( 0, newline + 1 );
                    continue;
                }

                if ( pending_.size() - hold > kMaximumHoldLength )
                {
                    out += formatChunk( pending_ );
                    pending_.clear();
                    break;
                }

                if ( hold > 0 )
                {
                    out += formatChunk( pending_.substr( 0, hold ) );
                    pending_.erase( 0, hold );
                }

                break;
            }

            return out;
        }

        std::string formatChunk( const std::string& chunk )
        {
            const std::string formatted = RichText::formatRich( chunk, at_line_start_ );

            // Track line-start context on the emitted text: transforms can create
            // newlines (\\ -> line break) that the raw input does not show.
            if ( !formatted.empty() )
                at_line_start_ = formatted.back() == '\n';
            else if ( !chunk.empty() )
                at_line_start_ = chunk.back() == '\n';

            return formatted;
        }

        /**
         * @brief Earliest position in s whose construct may continue past the end
         *        of the buffer; npos when the whole buffer is safe to format now.
         */
        static size_t holdPoint( std::string_view s, bool at_line_start )
        {
            bool line_start_run = at_line_start;  // only indent seen so far on this line
            size_t i = 0;

            while ( i < s.size() )
            {
                const char c = s[ i ];

                if ( c == '\n' )
                {
                    line_start_run = true;
                    ++i;
                    continue;
                }

                if ( line_start_run && (c == ' ' || c == '\t') )
                {
                    ++i;
                    continue;
                }

                // A leading *, -, + may be a list marker: the decision needs the
                // next character (marker iff followed by a space).
                if ( line_start_run && (c == '*' || c == '-' || c == '+') && i + 1 == s.size() )
                    return i;

                line_start_run = false;

                if ( c == '\\' )
                {
                    size_t j = i + 1;

                    while ( j < s.size() && std::isalpha( static_cast<unsigned char>( s[ j ] ) ) )
                        ++j;

                    if ( j == s.size() )
                        return i;  // the command name may continue

                    if ( j == i + 1 )
                    {
                        // Symbol escape (\\, \, ...): two characters, complete.
                        i = j + 1;
                        continue;
                    }

                    const std::string_view name = s.substr( i + 1, j - i - 1 );

                    if ( isKnownBraceCommand( name ) && j < s.size() && s[ j ] == '{' )
                    {
                        const size_t close = s.find( '}', j + 1 );

                        if ( close == std::string_view::npos )
                            return i;

                        if ( name == "frac" )
                        {
                            if ( close + 1 >= s.size() )
                                return i;  // need to see whether a second group follows

                            if ( s[ close + 1 ] == '{' )
                            {
                                const size_t second_close = s.find( '}', close + 2 );

                                if ( second_close == std::string_view::npos )
                                    return i;

                                i = second_close + 1;
                                continue;
                            }
                        }

                        i = close + 1;
                        continue;
                    }

                    i = j;
                    continue;
                }

                if ( c == '_' || c == '^' )
                {
                    if ( i + 1 == s.size() )
                        return i;

                    if ( s[ i + 1 ] == '{' )
                    {
                        const size_t close = s.find( '}', i + 2 );

                        if ( close == std::string_view::npos )
                            return i;

                        i = close + 1;
                        continue;
                    }

                    // The bare form consumes its following character only when it
                    // converts (digit/+/-) or pairs into a removed __ marker;
                    // otherwise the next character is scanned in its own right.
                    if ( isScriptableCharacter( s[ i + 1 ] ) || s[ i + 1 ] == c )
                        i += 2;
                    else
                        ++i;

                    continue;
                }

                if ( c == '~' && i + 1 == s.size() )
                    return i;  // a following ~ would make a removed ~~ pair

                ++i;
            }

            return std::string_view::npos;
        }

        static bool isKnownBraceCommand( std::string_view name )
        {
            return name == "text" || name == "mathrm" || name == "mathbf"
                || name == "mathit" || name == "operatorname" || name == "frac";
        }

        static bool isScriptableCharacter( char c )
        {
            return (c >= '0' && c <= '9') || c == '+' || c == '-';
        }

        std::string pending_;
        bool at_line_start_ = true;
    };

    /**
     * @brief Resolved Gemma control-token ids driving the channel router.
     *
     * All four routing ids must resolve (single-token vocab entries) for
     * streaming to be enabled; suppressed holds control tokens with no display
     * form (the live counterpart of stripSpecialTokens).
     */
    export struct GemmaStreamTokens
    {
        int32_t channel_open = -1;
        int32_t channel_close = -1;
        int32_t tool_call_open = -1;
        int32_t tool_call_close = -1;
        std::vector<int32_t> suppressed;
    };

    /**
     * @brief Per-turn orchestrator: routes each generated token to the live display.
     *
     * A small state machine over the Gemma channel protocol: answer text streams
     * into the solid block, reasoning dim-streams at detail Thoughts and above
     * (hidden at Off), and tool-call spans are suppressed entirely (the chat
     * loop dispatches them and prints the agentic trace). The full response
     * string keeps accumulating in the chat loop regardless — its post-hoc parse
     * is both the conversation history and this display's validation oracle.
     */
    export class StreamingResponseDisplay
    {
    public:

        enum class TokenAction
        {
            Continue,
            ToolCallOpened,  ///< Display suspended; the caller should re-arm the spinner.
        };

        StreamingResponseDisplay( ConsoleRenderer& renderer, const GemmaStreamTokens& tokens, DetailLevel detail )
            : renderer_( renderer ), tokens_( tokens ), detail_( detail )
        {
            renderer_.streamBegin();
        }

        TokenAction onToken( int32_t token, std::string_view piece )
        {
            if ( token == tokens_.tool_call_open && state_ != State::ToolCall )
            {
                deliverText( detokenizer_.flush() );
                state_before_tool_call_ = state_;
                state_ = State::ToolCall;
                renderer_.streamSuspend();

                return TokenAction::ToolCallOpened;
            }

            if ( token == tokens_.tool_call_close )
            {
                if ( state_ == State::ToolCall )
                    state_ = state_before_tool_call_;

                return TokenAction::Continue;
            }

            if ( state_ == State::ToolCall )
                return TokenAction::Continue;

            if ( token == tokens_.channel_open && state_ == State::Answer )
            {
                deliverText( detokenizer_.flush() );
                state_ = State::ChannelHeader;
                header_.clear();

                return TokenAction::Continue;
            }

            if ( token == tokens_.channel_close && state_ != State::Answer )
            {
                deliverText( detokenizer_.flush() );
                closeChannel();

                return TokenAction::Continue;
            }

            // Note the two guards above: an open marker inside a reasoning region
            // and a close marker with no open cycle are literal text to the
            // post-hoc ChannelParser, so they stream as literal text here too.

            if ( isSuppressed( token ) )
            {
                deliverText( detokenizer_.flush() );

                return TokenAction::Continue;
            }

            deliverText( detokenizer_.feed( piece ) );

            return TokenAction::Continue;
        }

        /// Round boundary: surrender any incomplete UTF-8 tail (no next token in
        /// this round will complete it).
        void onRoundEnd()
        {
            deliverText( detokenizer_.flush() );
        }

        /// Close the visual line before out-of-band lines (tool trace, info) print.
        void suspendForTrace()
        {
            renderer_.streamSuspend();
        }

        /// Turn end: flush every stage and close the block.
        void finish()
        {
            deliverText( detokenizer_.flush() );

            if ( state_ == State::ChannelHeader )
                beginThoughtBody();  // generation stopped inside the header

            if ( showThoughts() )
            {
                const std::string thought_tail = thought_formatter_.flush();

                if ( !thought_tail.empty() )
                    renderer_.streamText( true, thought_tail );
            }

            const std::string answer_tail = answer_formatter_.flush();

            if ( !answer_tail.empty() )
                renderer_.streamText( false, answer_tail );

            renderer_.streamEnd();
        }

    private:

        enum class State
        {
            Answer,         ///< Default: displayable answer text.
            ChannelHeader,  ///< After <|channel>: withholding the label line.
            Thought,        ///< Reasoning region until <channel|>.
            ToolCall,       ///< Suppressed span until <tool_call|>.
        };

        bool showThoughts() const
        {
            return detail_ >= DetailLevel::Thoughts;
        }

        bool isSuppressed( int32_t token ) const
        {
            for ( const int32_t suppressed : tokens_.suppressed )
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
                    renderer_.streamText( false, answer_formatter_.feed( text ) );
                    break;

                case State::Thought:
                    if ( showThoughts() )
                        renderer_.streamText( true, thought_formatter_.feed( text ) );
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
         * newline; anything else is reasoning content — the same classification
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

            if ( showThoughts() )
            {
                renderer_.streamThinkingBreak();

                if ( !body.empty() )
                    renderer_.streamText( true, thought_formatter_.feed( body ) );
            }
        }

        /// <channel|>: finish the reasoning region (classifying a still-open
        /// header first) and return to the answer channel.
        void closeChannel()
        {
            if ( state_ == State::ChannelHeader )
                beginThoughtBody();

            if ( state_ == State::Thought )
            {
                if ( showThoughts() )
                {
                    const std::string tail = thought_formatter_.flush();

                    if ( !tail.empty() )
                        renderer_.streamText( true, tail );
                }

                state_ = State::Answer;
            }
        }

        ConsoleRenderer& renderer_;
        GemmaStreamTokens tokens_;
        DetailLevel detail_;

        StreamingDetokenizer detokenizer_;
        IncrementalRichFormatter answer_formatter_;
        IncrementalRichFormatter thought_formatter_;

        State state_ = State::Answer;
        State state_before_tool_call_ = State::Answer;
        std::string header_;
    };
}
