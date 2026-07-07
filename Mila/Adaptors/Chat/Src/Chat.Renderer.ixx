module;
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <format>
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

export module Chat.Renderer;

import Chat.RichText;

namespace Mila::ChatApp
{
    export class ConsoleRenderer
    {
    public:

        // ── User turn ─────────────────────────────────────────────────────────

        void printUserPrompt() const
        {
            std::cout << '\n' << fg( 80, 100, 140 ) << " > " << reset();
        }

        // ── Progress spinner (turns and model loading) ────────────────────────

        /**
         * @brief Start the braille progress spinner, optionally with a trailing label.
         *
         * Used both for the per-turn "thinking" indicator (no label) and for model
         * loading ("Loading <model>"). The caller must not write to stdout between
         * startSpinner() and stopSpinner() or the animation will be corrupted.
         * A spinner already running is stopped and replaced.
         *
         * @param progress_counter Optional live counter appended to the spinner line
         * (" N tok"). Read with relaxed ordering on the spinner thread; the pointee must
         * outlive the startSpinner()/stopSpinner() window. A ticking count shows
         * generation is alive; a frozen count distinguishes a hang from a long response.
         */
        void startSpinner( std::string_view label = {}, const std::atomic<int>* progress_counter = nullptr )
        {
            if ( spinner_thread_.joinable() )
                stopSpinner();

            spinner_label_ = std::string( label );
            progress_counter_ = progress_counter;
            std::cout << "\x1b[?25l" << std::flush;  // hide cursor — eliminates blink flicker
            spinning_.store( true );
            spinner_thread_ = std::thread( [this]()
            {
                // Braille dot frames: smooth weight transitions, all same display width
                static constexpr const char* kFrames[] = {
                    "\xe2\xa0\x8b",  // ⠋
                    "\xe2\xa0\x99",  // ⠙
                    "\xe2\xa0\xb9",  // ⠹
                    "\xe2\xa0\xb8",  // ⠸
                    "\xe2\xa0\xbc",  // ⠼
                    "\xe2\xa0\xb4",  // ⠴
                    "\xe2\xa0\xa6",  // ⠦
                    "\xe2\xa0\xa7",  // ⠧
                    "\xe2\xa0\x87",  // ⠇
                    "\xe2\xa0\x8f",  // ⠏
                };
                int frame = 0;
                while ( spinning_.load() )
                {
                    std::cout << '\r' << fg( 100, 115, 155 )
                              << "  " << kFrames[ frame % 10 ];

                    if ( !spinner_label_.empty() )
                        std::cout << ' ' << spinner_label_;

                    if ( progress_counter_ )
                    {
                        const int count = progress_counter_->load( std::memory_order_relaxed );

                        if ( count > 0 )
                            std::cout << ' ' << count << " tok";
                    }

                    std::cout << reset() << "\x1b[K" << std::flush;  // erase to end of line
                    ++frame;

                    // Sliced sleep keeps stopSpinner() responsive (<= ~10 ms join):
                    // the streaming display stops the spinner from inside the token
                    // callback, where an 80 ms join would register as a token stall.
                    for ( int slice = 0; slice < 8 && spinning_.load(); ++slice )
                        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
                }
                std::cout << "\r\x1b[K" << std::flush;  // clear the spinner line
            } );
        }

        /// Stop and join the spinner. Safe to call when no spinner is running.
        void stopSpinner()
        {
            if ( !spinner_thread_.joinable() )
                return;

            spinning_.store( false );
            spinner_thread_.join();
            progress_counter_ = nullptr;
            std::cout << "\x1b[?25h" << std::flush;  // restore cursor
        }

        // ── Welcome box ───────────────────────────────────────────────────────

        void printWelcomeBox( std::string_view title ) const
        {
            const int inner  = static_cast<int>( title.size() ) + 4;  // 2-space pad each side
            const std::string dashes = repeatDash( inner );

            std::cout << "\n  " << fg( 90, 110, 170 )
                      << "\xe2\x95\xad" << dashes << "\xe2\x95\xae"   // ╭─╮
                      << reset() << '\n';

            std::cout << "  " << fg( 90, 110, 170 ) << "\xe2\x94\x82"  // │
                      << fg( 200, 220, 255 ) << "  " << title << "  "
                      << fg( 90, 110, 170 ) << "\xe2\x94\x82"           // │
                      << reset() << '\n';

            std::cout << "  " << fg( 90, 110, 170 )
                      << "\xe2\x95\xb0" << dashes << "\xe2\x95\xaf"   // ╰─╯
                      << reset() << "\n\n";
        }

        // ── Mila turn (buffered, solid color block, dynamic width) ────────────

        void printMilaResponse( std::string_view text )
        {
            // Total displayed width per line = max_width + 5 (1 margin + 2 left pad + text + 2 right pad).
            const int max_width = std::max( 20, std::min( consoleWidth() - 6, 80 ) );
            const auto lines    = RichText::wordWrap( RichText::formatRich( text ), max_width );

            int text_width = 4;
            for ( const auto& line : lines )
                text_width = std::max( text_width, static_cast<int>( line.size() ) );

            std::cout << '\n';
            for ( const auto& line : lines )
            {
                const int right_pad = text_width - static_cast<int>( line.size() ) + 2;
                std::cout << ' '
                          << bg( 40, 44, 60 ) << fg( 200, 215, 240 )
                          << "  " << line << std::string( right_pad, ' ' )
                          << reset() << '\n';
            }
            std::cout << '\n';
        }

        // ── Thinking block (dim, label, no solid background) ──────────────────

        void printThinking( std::string_view text )
        {
            const int max_width = std::max( 20, std::min( consoleWidth() - 6, 80 ) );
            const auto lines    = RichText::wordWrap( RichText::formatRich( text ), max_width );

            // \x1b[2m = dim — keeps the reasoning visually subordinate to the answer.
            std::cout << '\n' << "\x1b[2m" << fg( 120, 132, 165 )
                      << "  Thinking" << reset() << '\n';

            for ( const auto& line : lines )
                std::cout << "\x1b[2m" << fg( 120, 132, 165 )
                          << "  " << line << reset() << '\n';
        }

        // ── Streaming turn (live token display) ───────────────────────────────

        /**
         * @brief Arm the streaming block writer for a new turn.
         *
         * Captures the wrap width once so every line of the turn (and the
         * post-hoc validation oracle) wraps identically even if the console is
         * resized mid-stream. Transcript accessors from the previous turn are
         * invalidated here.
         */
        void streamBegin()
        {
            stream_ = StreamState{};
            stream_.wrap_width = std::max( 20, std::min( consoleWidth() - 6, 80 ) );
            stream_.active = true;
        }

        /**
         * @brief Append formatted text to the live turn display.
         *
         * The first visible word stops the spinner and opens the block visuals;
         * words appear as they complete (whitespace-delimited), wrapped with the
         * same greedy algorithm as RichText::wordWrap so the streamed transcript
         * matches the buffered render line for line. Leading whitespace of a
         * block is dropped and trailing whitespace never materializes, mirroring
         * the buffered pipeline's trims.
         *
         * @param thinking Selects the dim reasoning style vs the answer block style.
         * @param text     Formatted text (RichText output) to display.
         */
        void streamText( bool thinking, std::string_view text )
        {
            if ( !stream_.active || text.empty() )
                return;

            if ( stream_.block_open && stream_.thinking_style != thinking )
            {
                // Whitespace-only cross-style text never displays (the leading
                // trim after a switch discards it): drop it rather than churning
                // the block visuals between adjacent channel cycles.
                if ( text.find_first_not_of( " \t\n\r" ) == std::string_view::npos )
                    return;

                switchStreamBlock( thinking );
            }
            else if ( !stream_.block_open )
            {
                stream_.thinking_style = thinking;
            }

            for ( const char c : text )
            {
                if ( c == '\t' )
                {
                    // Mirror wordWrap's tab expansion so column accounting matches.
                    for ( int k = 0; k < 4; ++k )
                        streamCharacter( ' ' );
                }
                else
                {
                    streamCharacter( c );
                }
            }

            std::cout.flush();
        }

        /**
         * @brief Reasoning piece boundary between channel cycles.
         *
         * Mirrors ChannelParser's piece join: one blank line between non-empty
         * pieces, leading whitespace of the new piece dropped.
         */
        void streamThinkingBreak()
        {
            if ( !stream_.active )
                return;

            if ( stream_.block_open && !stream_.thinking_style )
                switchStreamBlock( true );
            else if ( !stream_.block_open )
                stream_.thinking_style = true;

            flushPendingWord();
            stream_.pending_whitespace.clear();

            if ( !stream_.thinking_lines.empty() || stream_.line_open )
                stream_.pending_whitespace = "\n\n";

            stream_.drop_leading_whitespace = true;
        }

        /**
         * @brief Close the open visual line so out-of-band lines (tool trace,
         *        info messages, spinner) can print.
         *
         * Pending word/whitespace survive: the logical text stream resumes on a
         * fresh line at the next streamText. A line forced closed mid-content
         * invalidates line-exact validation (streamForcedBreak).
         */
        void streamSuspend()
        {
            if ( !stream_.active || !stream_.line_open )
                return;

            stream_.forced_break = true;
            closeStreamLine();
            std::cout.flush();
        }

        /**
         * @brief Final flush at turn end: trailing whitespace is discarded (the
         *        buffered pipeline trims it), the last line closes, and the
         *        answer block gets its trailing blank line. Transcript accessors
         *        stay valid until the next streamBegin.
         */
        void streamEnd()
        {
            if ( !stream_.active )
                return;

            flushPendingWord();
            stream_.pending_whitespace.clear();

            if ( stream_.line_open )
                closeStreamLine();

            if ( stream_.block_open && !stream_.thinking_style )
                std::cout << '\n';

            std::cout.flush();
            stream_.active = false;
        }

        bool streamHasOutput() const { return stream_.any_output; }
        bool streamForcedBreak() const { return stream_.forced_break; }
        int streamWrapWidth() const { return stream_.wrap_width; }
        const std::vector<std::string>& streamedAnswerLines() const { return stream_.answer_lines; }
        const std::vector<std::string>& streamedThinkingLines() const { return stream_.thinking_lines; }

        // ── Tool call block (agentic flow, always visible) ────────────────────

        /**
         * @brief Render a tool call as an inline agentic-trace line, before the answer.
         *
         * The intent line (an arrow-prefixed human-readable action, e.g. "Getting weather
         * for Toronto, Canada") is always shown -- a tool call is conversational content,
         * not debug verbosity. The raw result payload is shown only at the highest detail
         * level (show_result), since the model's answer already conveys it. A greenish dim
         * tint keeps the trace distinct from the bluish Thinking block and subordinate to
         * the full-color answer block.
         */
        void printToolCall( std::string_view summary, std::string_view result, bool show_result )
        {
            const int max_width = std::max( 20, std::min( consoleWidth() - 6, 80 ) );

            std::cout << '\n';

            // Intent line at full color intensity (no dim attribute) -- an action reads as
            // conversational content, brighter than the dim Thinking/result metadata.
            const std::string intent = std::string( "\xe2\x86\x92 " ) + std::string( summary );  // arrow ->
            for ( const auto& line : RichText::wordWrap( intent, max_width ) )
                std::cout << fg( 130, 205, 155 )
                          << "  " << line << reset() << '\n';

            if ( !show_result )
                return;

            // Raw result payload stays dim -- debug detail, subordinate to the intent line.
            for ( const auto& line : RichText::wordWrap( result, max_width ) )
                std::cout << "\x1b[2m" << fg( 100, 130, 110 )
                          << "    " << line << reset() << '\n';
        }

        // ── Generation stats (on demand via /stats) ───────────────────────────

        /// Print a block of pre-formatted stats lines, dim so they read as metadata.
        void printStatsDetail( const std::vector<std::string>& lines ) const
        {
            for ( const auto& line : lines )
                std::cout << "\x1b[2m" << fg( 155, 170, 205 ) << line << reset() << '\n';
        }

        // ── Raw capture (protocol discovery) ──────────────────────────────────

        void printRaw( std::string_view text ) const
        {
            // Verbatim, unwrapped, unformatted — shows the exact control tokens
            // and markup so the Gemma channel protocol can be reverse-engineered.
            std::cout << '\n' << "\x1b[2m" << fg( 110, 120, 150 )
                      << "  [raw] " << text << reset() << '\n';
        }

        // ── System messages ───────────────────────────────────────────────────

        void printInfo( std::string_view msg ) const
        {
            std::cout << fg( 110, 125, 160 ) << msg << reset() << '\n';
        }

        void printError( std::string_view msg ) const
        {
            std::cout << fg( 195, 65, 65 ) << msg << reset() << '\n';
        }

    private:

        std::atomic<bool> spinning_{ false };
        std::thread       spinner_thread_;
        std::string       spinner_label_;
        const std::atomic<int>* progress_counter_{ nullptr };

        // ── Streaming block state ─────────────────────────────────────────────

        /**
         * @brief Incremental mirror of RichText::wordWrap plus block painting.
         *
         * current_line is the logical content of the open display line, exactly
         * as wordWrap would have built it; the console holds the same characters
         * plus the style escapes. pending_word buffers until whitespace commits
         * it; pending_whitespace buffers structural whitespace until the next
         * word proves it is interior (so trailing whitespace never renders).
         */
        struct StreamState
        {
            bool active = false;
            int wrap_width = 80;

            bool thinking_style = false;  // style of the open (or next) block
            bool block_open = false;
            bool line_open = false;
            std::string current_line;
            std::string pending_word;
            std::string pending_whitespace;
            bool drop_leading_whitespace = true;
            bool any_output = false;
            bool forced_break = false;

            std::vector<std::string> answer_lines;
            std::vector<std::string> thinking_lines;
        };

        StreamState stream_;

        void streamCharacter( char c )
        {
            if ( c == '\n' || c == ' ' )
            {
                flushPendingWord();

                if ( !stream_.drop_leading_whitespace )
                    stream_.pending_whitespace += c;

                return;
            }

            stream_.drop_leading_whitespace = false;
            stream_.pending_word += c;
        }

        /// Commit the pending word: materialize buffered whitespace, then apply
        /// the same append/wrap decisions as RichText::wordWrap's flush().
        void flushPendingWord()
        {
            if ( stream_.pending_word.empty() )
                return;

            materializePendingWhitespace();

            std::string word = std::move( stream_.pending_word );
            stream_.pending_word.clear();

            // Oversized words hard-break at the wrap width, as wordWrap does.
            while ( static_cast<int>( word.size() ) > stream_.wrap_width )
            {
                if ( !stream_.current_line.empty() )
                    closeStreamLine();

                appendToStreamLine( std::string_view( word ).substr( 0, static_cast<size_t>( stream_.wrap_width ) ) );
                closeStreamLine();
                word.erase( 0, static_cast<size_t>( stream_.wrap_width ) );
            }

            if ( stream_.current_line.empty() || stream_.current_line.back() == ' ' )
            {
                appendToStreamLine( word );
            }
            else if ( static_cast<int>( stream_.current_line.size() ) + 1 + static_cast<int>( word.size() ) <= stream_.wrap_width )
            {
                appendToStreamLine( " " );
                appendToStreamLine( word );
            }
            else
            {
                closeStreamLine();
                appendToStreamLine( word );
            }
        }

        /// Replay buffered structural whitespace through the wordWrap state
        /// machine: newlines end lines (blank lines included), spaces after a
        /// newline are preserved indentation, interior spaces collapse to the
        /// single separator the word-append rule inserts.
        void materializePendingWhitespace()
        {
            if ( stream_.pending_whitespace.empty() )
                return;

            bool at_line_start = stream_.current_line.empty();

            for ( const char c : stream_.pending_whitespace )
            {
                if ( c == '\n' )
                {
                    closeStreamLine();
                    at_line_start = true;
                }
                else if ( at_line_start )
                {
                    appendToStreamLine( " " );
                }
            }

            stream_.pending_whitespace.clear();
        }

        /// Close the current block (flushing its word) and set up the other
        /// style; the new block's visuals open lazily at its first word.
        void switchStreamBlock( bool thinking )
        {
            flushPendingWord();
            stream_.pending_whitespace.clear();

            if ( stream_.line_open )
                closeStreamLine();

            // Leaving a visible answer block mid-turn splits it visually, so
            // line-exact validation no longer applies to this turn.
            if ( !stream_.thinking_style && !stream_.answer_lines.empty() )
                stream_.forced_break = true;

            stream_.block_open = false;
            stream_.thinking_style = thinking;
            stream_.drop_leading_whitespace = true;
        }

        void ensureStreamBlockOpen()
        {
            if ( stream_.block_open )
                return;

            if ( stream_.thinking_style )
                std::cout << '\n' << "\x1b[2m" << fg( 120, 132, 165 )
                          << "  Thinking" << reset() << '\n';
            else
                std::cout << '\n';

            stream_.block_open = true;
        }

        /// Open a painted line: the first visible output takes the console line
        /// over from the spinner.
        void openStreamLine()
        {
            stopSpinner();
            ensureStreamBlockOpen();

            if ( stream_.thinking_style )
                std::cout << "\x1b[2m" << fg( 120, 132, 165 ) << "  ";
            else
                std::cout << ' ' << bg( 40, 44, 60 ) << fg( 200, 215, 240 ) << "  ";

            stream_.line_open = true;
        }

        void appendToStreamLine( std::string_view text )
        {
            if ( text.empty() )
                return;

            if ( !stream_.line_open )
                openStreamLine();

            std::cout << text;
            stream_.current_line += text;
            stream_.any_output = true;
        }

        void closeStreamLine()
        {
            if ( !stream_.line_open )
                openStreamLine();  // a blank line still paints its block bar

            std::cout << ( stream_.thinking_style ? "" : "  " ) << reset() << '\n';
            stream_.line_open = false;

            currentStreamLines().push_back( std::move( stream_.current_line ) );
            stream_.current_line.clear();
        }

        std::vector<std::string>& currentStreamLines()
        {
            return stream_.thinking_style ? stream_.thinking_lines : stream_.answer_lines;
        }

        // ── ANSI helpers ──────────────────────────────────────────────────────

        static std::string bg( int r, int g, int b )
        {
            return std::format( "\x1b[48;2;{};{};{}m", r, g, b );
        }

        static std::string fg( int r, int g, int b )
        {
            return std::format( "\x1b[38;2;{};{};{}m", r, g, b );
        }

        static const char* reset() { return "\x1b[0m"; }

        // ── Console width ─────────────────────────────────────────────────────

        static int consoleWidth()
        {
#ifdef _WIN32
            CONSOLE_SCREEN_BUFFER_INFO csbi;
            if ( GetConsoleScreenBufferInfo(
                     GetStdHandle( STD_OUTPUT_HANDLE ), &csbi ) )
                return csbi.srWindow.Right - csbi.srWindow.Left + 1;
#endif
            return 80;
        }

        // ── Box drawing helpers ───────────────────────────────────────────────

        static std::string repeatDash( int count )
        {
            std::string s;
            s.reserve( count * 3 );
            for ( int i = 0; i < count; ++i )
                s += "\xe2\x94\x80";  // U+2500 ─
            return s;
        }
    };
}
