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
    /**
     * @brief What a run of characters is, as distinct from how it is painted.
     *
     * The formatter decides meaning; ConsoleRenderer decides escapes. Keeping those
     * apart is what lets the stream validator keep comparing plain text: weight and
     * colour never enter the comparison, so styling cannot make a correct stream
     * look divergent.
     */
    export enum StyleFlag : unsigned char
    {
        StyleNone = 0,
        StyleBold = 1 << 0,
        StyleHeading = 1 << 1,
    };

    /**
     * @brief Text with one attribute byte per text byte.
     *
     * A parallel string rather than a span list because every consumer either wraps
     * this text or paints it, and both walk it linearly: slicing a line slices the
     * attributes with the identical arithmetic, so the two cannot drift. Per byte
     * rather than per code point for the same reason the wrap counts bytes -- a
     * multi-byte sequence carries one repeated attribute and never splits.
     */
    export struct StyledText
    {
        std::string text;
        std::string attributes;

        void append( char c, unsigned char style )
        {
            text += c;
            attributes += static_cast<char>( style );
        }

        void append( std::string_view s, unsigned char style )
        {
            text += s;
            attributes.append( s.size(), static_cast<char>( style ) );
        }

        /// Append another run, keeping its own per-byte attributes.
        void append( const StyledText& other )
        {
            text += other.text;
            attributes += other.attributes;
        }

        void clear()
        {
            text.clear();
            attributes.clear();
        }

        /// The style of the first byte, or None when there is none. Used for the
        /// separator space a wrap inserts, so it takes the style of what follows.
        unsigned char leadingStyle() const
        {
            return attributes.empty()
                ? StyleNone : static_cast<unsigned char>( attributes.front() );
        }

        bool empty() const { return text.empty(); }
    };

    /// One wrapped display line, attributes rebased to the line's own offsets.
    export using StyledLine = StyledText;

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
        /// LaTeX, symbol commands and scripts to Unicode. Pure text-to-text.
        static std::string convertMarkup( std::string_view in )
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

            return s;
        }

        /**
         * @brief formatRich, keeping what the markup meant.
         *
         * The LaTeX and symbol passes above are pure text-to-text and stay that way;
         * only the markdown tail carries meaning worth painting, so it runs as one
         * left-to-right scan that emits text and attributes together. That replaces
         * what was a chain of whole-string replaceAll passes -- which could not have
         * produced attributes at all, since each pass destroyed the offsets of the
         * one before it.
         */
        static StyledText formatRichStyled( std::string_view in, bool at_line_start = true )
        {
            return applyMarkdown( convertMarkup( in ), at_line_start );
        }

        /// Plain text only, for callers with nothing to paint with.
        static std::string formatRich( std::string_view in, bool at_line_start = true )
        {
            return formatRichStyled( in, at_line_start ).text;
        }

        /**
         * @brief Greedy word-wrap with leading-indent preservation and hard breaks
         *        for words longer than the wrap width. Tabs expand to 4 spaces.
         */
        /// Plain text, for callers with nothing to paint with.
        static std::vector<std::string> wordWrap( std::string_view text, int max_width )
        {
            StyledText styled;
            styled.append( text, StyleNone );

            std::vector<std::string> lines;

            for ( const auto& line : wordWrap( styled, max_width ) )
                lines.push_back( line.text );

            return lines;
        }

        static std::vector<StyledLine> wordWrap( const StyledText& styled, int max_width )
        {
            // Expand tabs to 4 spaces so column counting stays accurate.
            StyledText text;
            text.text.reserve( styled.text.size() );
            text.attributes.reserve( styled.text.size() );

            for ( size_t at = 0; at < styled.text.size(); ++at )
            {
                const auto style = static_cast<unsigned char>( styled.attributes[ at ] );

                if ( styled.text[ at ] == '\t' )
                    text.append( "    ", style );
                else
                    text.append( styled.text[ at ], style );
            }

            std::vector<StyledLine> lines;
            StyledText current;
            StyledText word;
            bool at_line_start = true;

            // Append word to current, inserting a space separator only when the last
            // character of current is not already a space (i.e. not a leading-indent run).
            // Slice both strings with identical arithmetic -- that equality is the
            // whole reason attributes are a parallel string rather than a span list.
            auto splitOff = [&]( StyledText& from, size_t count )
            {
                StyledLine head;
                head.text = from.text.substr( 0, count );
                head.attributes = from.attributes.substr( 0, count );

                from.text = from.text.substr( count );
                from.attributes = from.attributes.substr( count );

                return head;
            };

            auto flush = [&]()
            {
                if ( word.empty() )
                    return;

                while ( static_cast<int>( word.text.size() ) > max_width )
                {
                    if ( !current.empty() )
                    {
                        lines.push_back( current );
                        current.clear();
                    }

                    lines.push_back( splitOff( word, static_cast<size_t>( max_width ) ) );
                }

                if ( current.empty() || current.text.back() == ' ' )
                {
                    current.append( word );  // no separator: empty line or following indent
                }
                else if ( static_cast<int>( current.text.size() ) + 1
                    + static_cast<int>( word.text.size() ) <= max_width )
                {
                    // The separator takes the style of what follows, so a run of
                    // styled words reads as one run rather than as gapped fragments.
                    current.append( ' ', word.leadingStyle() );
                    current.append( word );
                }
                else
                {
                    lines.push_back( current );
                    current = word;
                }

                word.clear();
            };

            for ( size_t at = 0; at < text.text.size(); ++at )
            {
                const char c = text.text[ at ];
                const auto style = static_cast<unsigned char>( text.attributes[ at ] );

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
                        current.append( ' ', style );  // preserve leading indentation
                    else
                        flush();
                }
                else
                {
                    at_line_start = false;
                    word.append( c, style );
                }
            }

            flush();

            if ( !current.empty() )
                lines.push_back( current );

            return lines;
        }

    private:

        /**
         * @brief Markdown structure and emphasis, in one pass, as text plus attributes.
         *
         * Line structure first (heading marker, list marker, indentation), then inline
         * emphasis. A heading's flag covers the whole line, so a heading that wraps
         * stays a heading on every display line it occupies.
         *
         * Emphasis markers are consumed rather than kept: `**`/`__` toggle bold, `~~`
         * and a lone `*` and backticks are dropped with their text retained. A lone
         * `_` is left alone -- it is far more often a snake_case identifier than
         * emphasis, and convertMarkup has already had its say about scripts.
         *
         * at_line_start gates the first line: a chunk beginning mid-line must not read
         * its first characters as line structure.
         */
        static StyledText applyMarkdown( const std::string& s, bool at_line_start )
        {
            StyledText out;
            out.text.reserve( s.size() );
            out.attributes.reserve( s.size() );

            size_t i = 0;
            bool bold = false;

            while ( i < s.size() )
            {
                unsigned char line_style = StyleNone;

                if ( at_line_start )
                {
                    const size_t indent_start = i;

                    while ( i < s.size() && (s[ i ] == ' ' || s[ i ] == '\t') )
                        ++i;

                    // ATX heading: one to six hashes then a space. Levels collapse to
                    // one treatment -- models pick depth inconsistently, so honouring
                    // it would encode their noise rather than the section structure.
                    size_t hashes = i;

                    while ( hashes < s.size() && s[ hashes ] == '#' )
                        ++hashes;

                    const size_t hash_count = hashes - i;

                    if ( hash_count >= 1 && hash_count <= 6
                        && hashes < s.size() && s[ hashes ] == ' ' )
                    {
                        // The marker and its space go; the indent goes with them, since
                        // a heading owns its whole line and has nothing to align to.
                        line_style = StyleHeading;
                        i = hashes + 1;
                    }
                    else
                    {
                        const bool is_marker = i + 1 < s.size()
                            && (s[ i ] == '*' || s[ i ] == '-' || s[ i ] == '+')
                            && s[ i + 1 ] == ' ';

                        out.append( std::string_view( s ).substr(
                            indent_start, i - indent_start ), StyleNone );

                        if ( is_marker )
                        {
                            out.append( "\xE2\x80\xA2", StyleNone );  // U+2022 bullet
                            ++i;  // drop the marker; its trailing space stays
                        }
                    }
                }

                while ( i < s.size() && s[ i ] != '\n' )
                {
                    const char c = s[ i ];

                    if ( (c == '*' || c == '_') && i + 1 < s.size() && s[ i + 1 ] == c )
                    {
                        bold = !bold;
                        i += 2;
                        continue;
                    }

                    if ( c == '~' && i + 1 < s.size() && s[ i + 1 ] == '~' )
                    {
                        i += 2;
                        continue;
                    }

                    if ( c == '*' || c == '`' )
                    {
                        ++i;
                        continue;
                    }

                    out.append( c, static_cast<unsigned char>(
                        line_style | (bold ? StyleBold : StyleNone) ) );
                    ++i;
                }

                if ( i < s.size() )
                    out.append( s[ i++ ], StyleNone );  // the newline

                // Emphasis does not survive a line break: an unclosed ** would
                // otherwise bold the rest of the response.
                bold = false;
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
