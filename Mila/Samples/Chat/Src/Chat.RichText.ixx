/**
 * @file Chat.RichText.ixx
 * @brief Shared display-text pipeline: markdown/LaTeX-lite to readable Unicode,
 *        plus the greedy word-wrap used by every console block.
 *
 * Extracted from ConsoleRenderer so the buffered render path and the streaming
 * display path share one implementation: the streamed transcript is validated
 * against the buffered render of the same response, which only holds if both
 * run exactly the same transforms.
 */

module;
#include <cctype>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

export module Chat.RichText;

namespace Mila::ChatApp
{
    export class RichText
    {
    public:

        /**
         * @brief Convert a model response's markdown/LaTeX markup to plain Unicode.
         *
         * The console block has no rich styling, so emphasis markers are removed
         * (their text kept) and common LaTeX is rendered to Unicode rather than
         * shown literally: $...$ delimiters dropped, \text{x}->x, \frac{a}{b}->a/b,
         * digit sub/superscripts (H_2O -> H2O), and a small set of symbol commands.
         * Unknown \commands are stripped. This is a pragmatic subset, not a full
         * TeX engine.
         *
         * at_line_start says whether the first character of in sits at a display
         * line start; the streaming formatter passes false when a chunk begins
         * mid-line so a mid-line "* " is not mistaken for a list marker.
         */
        static std::string formatRich( std::string_view in, bool at_line_start = true )
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
            s = convertListMarkers( s, at_line_start );

            // Markdown emphasis: keep the text, drop the markers.
            replaceAll( s, "**", "" );
            replaceAll( s, "__", "" );
            replaceAll( s, "~~", "" );
            eraseAnyOf( s, "*`" );

            return s;
        }

        /**
         * @brief Greedy word-wrap with leading-indent preservation and hard breaks
         *        for words longer than the wrap width. Tabs expand to 4 spaces.
         */
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

    private:

        /**
         * @brief Convert leading markdown list markers to a bullet glyph.
         *
         * A line whose first non-blank content is "* ", "- ", or "+ " is a list
         * item; the marker becomes U+2022 with the indentation preserved. Other
         * uses of '*'/'-'/'+' (emphasis, hyphens, minus signs) are left untouched.
         * at_line_start gates the first line: a chunk that begins mid-line must
         * not treat its first characters as a line start.
         */
        static std::string convertListMarkers( const std::string& s, bool at_line_start )
        {
            std::string out;
            out.reserve( s.size() );

            size_t i = 0;

            while ( i < s.size() )
            {
                if ( at_line_start )
                {
                    const size_t indent_start = i;

                    while ( i < s.size() && (s[ i ] == ' ' || s[ i ] == '\t') )
                        ++i;

                    const bool is_marker = i + 1 < s.size()
                        && (s[ i ] == '*' || s[ i ] == '-' || s[ i ] == '+')
                        && s[ i + 1 ] == ' ';

                    out.append( s, indent_start, i - indent_start );  // preserved indentation

                    if ( is_marker )
                    {
                        out += "\xE2\x80\xA2";  // U+2022 bullet
                        ++i;                    // drop the marker; its trailing space stays
                    }
                }

                while ( i < s.size() && s[ i ] != '\n' )
                    out += s[ i++ ];

                if ( i < s.size() )
                    out += s[ i++ ];  // the newline

                at_line_start = true;
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
                    else if ( isScriptable( s[ i + 1 ] ) )
                    {
                        // Bare (non-brace) form: only consume the marker when it maps to a
                        // real subscript/superscript glyph (digit/+/-). Anything else -- a
                        // letter -- is overwhelmingly more likely a snake_case identifier
                        // (e.g. get_weather) than LaTeX shorthand, so leave marker and
                        // character both untouched rather than silently deleting the marker.
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

        static bool isScriptable( char c )
        {
            return (c >= '0' && c <= '9') || c == '+' || c == '-';
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

            // No Unicode form (e.g. a letter index inside explicit braces, x_{max}):
            // keep the character, drop the marker. Only reachable from the brace-form
            // call site -- the bare-marker call site in convertScripts() is gated by
            // isScriptable() and never invokes this with a non-digit/+/- character.
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
    };
}
