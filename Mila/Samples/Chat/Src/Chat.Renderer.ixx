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
#include <cctype>
#include <utility>

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

        // ── Progress spinner (turns and model loading) ────────────────────────

        /**
         * @brief Start the braille progress spinner, optionally with a trailing label.
         *
         * Used both for the per-turn "thinking" indicator (no label) and for model
         * loading ("Loading <model>"). The caller must not write to stdout between
         * startSpinner() and stopSpinner() or the animation will be corrupted.
         */
        void startSpinner( std::string_view label = {} )
        {
            spinner_label_ = std::string( label );
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

                    std::cout << reset() << "\x1b[K" << std::flush;  // erase to end of line
                    ++frame;
                    std::this_thread::sleep_for( std::chrono::milliseconds( 80 ) );
                }
                std::cout << "\r\x1b[K" << std::flush;  // clear the spinner line
            } );
        }

        void stopSpinner()
        {
            spinning_.store( false );
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
            const auto lines    = wordWrap( formatRich( text ), max_width );

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
            const auto lines    = wordWrap( formatRich( text ), max_width );

            // \x1b[2m = dim — keeps the reasoning visually subordinate to the answer.
            std::cout << '\n' << "\x1b[2m" << fg( 120, 132, 165 )
                      << "  Thinking" << reset() << '\n';

            for ( const auto& line : lines )
                std::cout << "\x1b[2m" << fg( 120, 132, 165 )
                          << "  " << line << reset() << '\n';
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

        // ── Rich text (markdown + LaTeX-lite -> readable Unicode) ─────────────

        /**
         * @brief Convert a model response's markdown/LaTeX markup to plain Unicode.
         *
         * The console block has no rich styling, so emphasis markers are removed
         * (their text kept) and common LaTeX is rendered to Unicode rather than
         * shown literally: $...$ delimiters dropped, \text{x}->x, \frac{a}{b}->a/b,
         * digit sub/superscripts (H_2O -> H2O), and a small set of symbol commands.
         * Unknown \commands are stripped. This is a pragmatic subset, not a full
         * TeX engine.
         */
        static std::string formatRich( std::string_view in )
        {
            std::string s( in );

            // Brace-wrapped text commands: keep the inner content.
            for ( const char* cmd : { "\\text", "\\mathrm", "\\mathbf", "\\mathit", "\\operatorname" } )
                s = unwrapBraceCommand( s, cmd );

            s = convertFraction( s );

            // Symbol commands -> Unicode. \\ (line break) and spacing macros too.
            static const std::pair<std::string_view, std::string_view> kSymbols[] = {
                { "\\times", "\xC3\x97" },       // ×
                { "\\cdot", "\xC2\xB7" },        // ·
                { "\\div", "\xC3\xB7" },         // ÷
                { "\\pm", "\xC2\xB1" },          // ±
                { "\\approx", "\xE2\x89\x88" },  // ≈
                { "\\neq", "\xE2\x89\xA0" },     // ≠
                { "\\leq", "\xE2\x89\xA4" }, { "\\le", "\xE2\x89\xA4" },  // ≤
                { "\\geq", "\xE2\x89\xA5" }, { "\\ge", "\xE2\x89\xA5" },  // ≥
                { "\\rightarrow", "\xE2\x86\x92" }, { "\\to", "\xE2\x86\x92" }, // →
                { "\\leftarrow", "\xE2\x86\x90" },  // ←
                { "\\infty", "\xE2\x88\x9E" },   // ∞
                { "\\degree", "\xC2\xB0" }, { "\\circ", "\xC2\xB0" },  // °
                { "\\\\", "\n" },
                { "\\,", " " }, { "\\;", " " }, { "\\:", " " }, { "\\!", "" },
            };
            for ( auto [from, to] : kSymbols )
                replaceAll( s, from, to );

            s = convertScripts( s, '_' );
            s = convertScripts( s, '^' );

            s = stripBackslashCommands( s );

            // Drop math delimiters and any leftover grouping braces.
            replaceAll( s, "$$", "" );
            replaceAll( s, "$", "" );
            eraseAnyOf( s, "{}" );

            // Markdown lists: convert a leading "* "/"- "/"+ " marker to a bullet
            // glyph, preserving indentation. This must run before emphasis stripping
            // so a list '*' is not erased along with the inline emphasis markers.
            s = convertListMarkers( s );

            // Markdown emphasis: keep the text, drop the markers.
            replaceAll( s, "**", "" );
            replaceAll( s, "__", "" );
            replaceAll( s, "~~", "" );
            eraseAnyOf( s, "*`" );

            return s;
        }

        /**
         * @brief Convert leading markdown list markers to a bullet glyph.
         *
         * A line whose first non-blank content is "* ", "- ", or "+ " is a list
         * item; the marker becomes U+2022 with the indentation preserved. Other
         * uses of '*'/'-'/'+' (emphasis, hyphens, minus signs) are left untouched.
         */
        static std::string convertListMarkers( const std::string& s )
        {
            std::string out;
            out.reserve( s.size() );

            size_t i = 0;

            while ( i < s.size() )
            {
                const size_t line_start = i;

                while ( i < s.size() && (s[ i ] == ' ' || s[ i ] == '\t') )
                    ++i;

                const bool is_marker = i + 1 < s.size()
                    && (s[ i ] == '*' || s[ i ] == '-' || s[ i ] == '+')
                    && s[ i + 1 ] == ' ';

                out.append( s, line_start, i - line_start );  // preserved indentation

                if ( is_marker )
                {
                    out += "\xE2\x80\xA2";  // U+2022 bullet
                    ++i;                    // drop the marker; its trailing space stays
                }

                while ( i < s.size() && s[ i ] != '\n' )
                    out += s[ i++ ];

                if ( i < s.size() )
                    out += s[ i++ ];  // the newline
            }

            return out;
        }

        static void replaceAll( std::string& s, std::string_view from, std::string_view to )
        {
            if ( from.empty() )
                return;

            size_t pos = 0;

            while ( (pos = s.find( from, pos )) != std::string::npos )
            {
                s.replace( pos, from.size(), to );
                pos += to.size();
            }
        }

        static void eraseAnyOf( std::string& s, std::string_view chars )
        {
            std::erase_if( s, [chars]( char c )
            {
                return chars.find( c ) != std::string_view::npos;
            } );
        }

        static std::string unwrapBraceCommand( const std::string& s, std::string_view cmd )
        {
            const std::string token = std::string( cmd ) + "{";
            std::string out;
            size_t i = 0;

            while ( i < s.size() )
            {
                const size_t p = s.find( token, i );

                if ( p == std::string::npos )
                {
                    out += s.substr( i );
                    break;
                }

                out += s.substr( i, p - i );

                const size_t inner = p + token.size();
                const size_t close = s.find( '}', inner );

                if ( close == std::string::npos )
                {
                    out += s.substr( p );
                    break;
                }

                out += s.substr( inner, close - inner );
                i = close + 1;
            }

            return out;
        }

        static std::string convertFraction( const std::string& s )
        {
            const std::string token = "\\frac{";
            std::string out;
            size_t i = 0;

            while ( i < s.size() )
            {
                const size_t p = s.find( token, i );

                if ( p == std::string::npos )
                {
                    out += s.substr( i );
                    break;
                }

                out += s.substr( i, p - i );

                const size_t a0 = p + token.size();
                const size_t a1 = s.find( '}', a0 );

                if ( a1 == std::string::npos || a1 + 1 >= s.size() || s[ a1 + 1 ] != '{' )
                {
                    out += s.substr( p );
                    break;
                }

                const size_t b0 = a1 + 2;
                const size_t b1 = s.find( '}', b0 );

                if ( b1 == std::string::npos )
                {
                    out += s.substr( p );
                    break;
                }

                out += s.substr( a0, a1 - a0 ) + "/" + s.substr( b0, b1 - b0 );
                i = b1 + 1;
            }

            return out;
        }

        static std::string convertScripts( const std::string& s, char marker )
        {
            const bool subscript = (marker == '_');
            std::string out;
            size_t i = 0;

            while ( i < s.size() )
            {
                if ( s[ i ] == marker && i + 1 < s.size() )
                {
                    if ( s[ i + 1 ] == '{' )
                    {
                        const size_t close = s.find( '}', i + 2 );

                        if ( close != std::string::npos )
                        {
                            for ( size_t k = i + 2; k < close; ++k )
                                out += scriptChar( s[ k ], subscript );

                            i = close + 1;
                            continue;
                        }
                    }
                    else
                    {
                        out += scriptChar( s[ i + 1 ], subscript );
                        i += 2;
                        continue;
                    }
                }

                out += s[ i ];
                ++i;
            }

            return out;
        }

        static std::string scriptChar( char c, bool subscript )
        {
            static const char* const kSub[ 10 ] = {
                "\xE2\x82\x80", "\xE2\x82\x81", "\xE2\x82\x82", "\xE2\x82\x83", "\xE2\x82\x84",
                "\xE2\x82\x85", "\xE2\x82\x86", "\xE2\x82\x87", "\xE2\x82\x88", "\xE2\x82\x89" };
            static const char* const kSup[ 10 ] = {
                "\xE2\x81\xB0", "\xC2\xB9", "\xC2\xB2", "\xC2\xB3", "\xE2\x81\xB4",
                "\xE2\x81\xB5", "\xE2\x81\xB6", "\xE2\x81\xB7", "\xE2\x81\xB8", "\xE2\x81\xB9" };

            if ( c >= '0' && c <= '9' )
                return subscript ? kSub[ c - '0' ] : kSup[ c - '0' ];

            if ( c == '+' )
                return subscript ? "\xE2\x82\x8A" : "\xE2\x81\xBA";

            if ( c == '-' )
                return subscript ? "\xE2\x82\x8B" : "\xE2\x81\xBB";

            // No Unicode form (e.g. a letter index): keep the character, drop the marker.
            return std::string( 1, c );
        }

        static std::string stripBackslashCommands( const std::string& s )
        {
            std::string out;
            size_t i = 0;

            while ( i < s.size() )
            {
                if ( s[ i ] == '\\' )
                {
                    size_t j = i + 1;

                    while ( j < s.size() && std::isalpha( static_cast<unsigned char>( s[ j ] ) ) )
                        ++j;

                    // Drop the whole \command (or a lone backslash before a symbol).
                    i = (j > i + 1) ? j : i + 1;
                    continue;
                }

                out += s[ i ];
                ++i;
            }

            return out;
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
            bool at_line_start = true;

            // Append word to current, inserting a space separator only when the last
            // character of current is not already a space (i.e. not a leading-indent run).
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

                if ( current.empty() || current.back() == ' ' )
                {
                    current += word;  // no separator: empty line or following indent spaces
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
                    at_line_start = true;
                }
                else if ( c == ' ' )
                {
                    if ( at_line_start )
                        current += ' ';  // preserve leading indentation
                    else
                        flush();
                }
                else
                {
                    at_line_start = false;
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
