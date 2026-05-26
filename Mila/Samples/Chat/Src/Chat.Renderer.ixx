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

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

export module Chat.Renderer;

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

        // ── Thinking spinner ──────────────────────────────────────────────────

        void beginThinking()
        {
            std::cout << "\x1b[?25l" << std::flush;  // hide cursor — eliminates blink flicker
            thinking_.store( true );
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
                while ( thinking_.load() )
                {
                    std::cout << '\r' << fg( 100, 115, 155 )
                              << "  " << kFrames[ frame % 10 ]
                              << reset() << std::flush;
                    ++frame;
                    std::this_thread::sleep_for( std::chrono::milliseconds( 80 ) );
                }
                std::cout << "\r   \r" << std::flush;
            } );
        }

        void endThinking()
        {
            thinking_.store( false );
            if ( spinner_thread_.joinable() )
                spinner_thread_.join();
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
            const auto lines    = wordWrap( text, max_width );

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

        // ── Generation stats ──────────────────────────────────────────────────

        void printStats( float prefill_ms, float decode_tps, int tokens ) const
        {
            // \x1b[2m = dim — gives lighter visual weight, reads as "smaller"
            std::cout << "\x1b[2m" << fg( 155, 170, 205 );

            if ( decode_tps > 0.0f )
                std::cout << std::format(
                    "  {:.0f} ms  \xe2\x94\x82  {:.1f} tok/s  \xe2\x94\x82  {} tokens",
                    prefill_ms, decode_tps, tokens );
            else
                std::cout << std::format(
                    "  {:.0f} ms  \xe2\x94\x82  {} token",
                    prefill_ms, tokens );

            std::cout << reset() << '\n';
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

        std::atomic<bool> thinking_{ false };
        std::thread       spinner_thread_;

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

        // ── Word-wrap ─────────────────────────────────────────────────────────

        static std::vector<std::string> wordWrap( std::string_view text, int max_width )
        {
            // Expand tabs to 4 spaces so column counting stays accurate.
            std::string expanded;
            expanded.reserve( text.size() );
            for ( char c : text )
            {
                if ( c == '\t' ) expanded += "    ";
                else             expanded += c;
            }
            text = expanded;

            std::vector<std::string> lines;
            std::string current;
            std::string word;

            auto flush = [&]()
            {
                if ( word.empty() )
                    return;

                while ( static_cast<int>( word.size() ) > max_width )
                {
                    if ( !current.empty() )
                    {
                        lines.push_back( current );
                        current.clear();
                    }
                    lines.push_back( word.substr( 0, max_width ) );
                    word = word.substr( max_width );
                }

                if ( current.empty() )
                {
                    current = word;
                }
                else if ( static_cast<int>( current.size() ) + 1 + static_cast<int>( word.size() ) <= max_width )
                {
                    current += ' ';
                    current += word;
                }
                else
                {
                    lines.push_back( current );
                    current = word;
                }

                word.clear();
            };

            for ( char c : text )
            {
                if ( c == '\n' )
                {
                    flush();
                    lines.push_back( current );
                    current.clear();
                }
                else if ( c == ' ' )
                {
                    flush();
                }
                else
                {
                    word += c;
                }
            }

            flush();

            if ( !current.empty() )
                lines.push_back( current );

            return lines;
        }
    };
}
