/**
 * @file Chat.IncrementalRichFormatter.ixx
 * @brief Formats streamed text in pieces that render exactly as the whole text would.
 */

export module Chat.IncrementalRichFormatter;

import std;
import Chat.RichText;

namespace Mila::ChatApp
{
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

        StyledText feed( std::string_view text )
        {
            pending_.append( text );

            return drain( false );
        }

        StyledText flush()
        {
            return drain( true );
        }

    private:

        static constexpr std::size_t kMaximumHoldLength = 512;

        StyledText drain( bool force )
        {
            StyledText out;

            while ( !pending_.empty() )
            {
                const std::size_t hold = force ? std::string::npos : holdPoint( pending_, at_line_start_ );

                if ( hold == std::string::npos )
                {
                    out.append( formatChunk( pending_ ) );
                    pending_.clear();
                    break;
                }

                // A construct is never held across a display line: force the
                // whole line through rather than stalling on malformed markup.
                const std::size_t newline = pending_.find( '\n', hold );

                if ( newline != std::string::npos )
                {
                    out.append( formatChunk( pending_.substr( 0, newline + 1 ) ) );
                    pending_.erase( 0, newline + 1 );
                    continue;
                }

                if ( pending_.size() - hold > kMaximumHoldLength )
                {
                    out.append( formatChunk( pending_ ) );
                    pending_.clear();
                    break;
                }

                if ( hold > 0 )
                {
                    out.append( formatChunk( pending_.substr( 0, hold ) ) );
                    pending_.erase( 0, hold );
                }

                break;
            }

            return out;
        }

        StyledText formatChunk( const std::string& chunk )
        {
            const StyledText formatted = RichText::formatRichStyled( chunk, at_line_start_ );

            // Track line-start context on the emitted text: transforms can create
            // newlines (\\ -> line break) that the raw input does not show.
            if ( !formatted.empty() )
                at_line_start_ = formatted.text.back() == '\n';
            else if ( !chunk.empty() )
                at_line_start_ = chunk.back() == '\n';

            return formatted;
        }

        /**
         * @brief Earliest position in s whose construct may continue past the end
         *        of the buffer; npos when the whole buffer is safe to format now.
         */
        static std::size_t holdPoint( std::string_view s, bool at_line_start )
        {
            bool line_start_run = at_line_start;  // only indent seen so far on this line

            // Where that run began, so it can be held whole. A leading indent is part
            // of the list-marker construct, not text preceding it: emitting the indent
            // alone leaves the next chunk believing it starts mid-line.
            std::size_t indent_start = at_line_start ? 0 : std::string_view::npos;
            std::size_t i = 0;

            while ( i < s.size() )
            {
                const char c = s[ i ];

                if ( c == '\n' )
                {
                    line_start_run = true;
                    ++i;
                    indent_start = i;
                    continue;
                }

                if ( line_start_run && (c == ' ' || c == '\t') )
                {
                    ++i;
                    continue;
                }

                // A leading *, -, + may be a list marker: the decision needs the
                // next character (marker iff followed by a space). Held from the start
                // of the indent, not from the marker -- releasing the indent alone would
                // clear at_line_start_ and cost the marker its bullet on the next chunk.
                if ( line_start_run && (c == '*' || c == '-' || c == '+') && i + 1 == s.size() )
                    return indent_start;

                // A leading # run may be an ATX heading, decided by the character after
                // the run. Hold the whole construct until that character arrives: half a
                // marker formats as literal hashes, which is what the buffered render
                // would never produce.
                if ( line_start_run && c == '#' )
                {
                    std::size_t hashes = i;

                    while ( hashes < s.size() && s[ hashes ] == '#' )
                        ++hashes;

                    if ( hashes == s.size() )
                        return indent_start;

                    line_start_run = false;
                    i = hashes;
                    continue;
                }

                line_start_run = false;

                if ( c == '\\' )
                {
                    std::size_t j = i + 1;

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
                        const std::size_t close = s.find( '}', j + 1 );

                        if ( close == std::string_view::npos )
                            return i;

                        if ( name == "frac" )
                        {
                            if ( close + 1 >= s.size() )
                                return i;  // need to see whether a second group follows

                            if ( s[ close + 1 ] == '{' )
                            {
                                const std::size_t second_close = s.find( '}', close + 2 );

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
                        const std::size_t close = s.find( '}', i + 2 );

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

                // A following twin would make an emphasis pair: ~~ is removed, ** toggles
                // bold. Splitting the pair across chunks would drop the toggle and leave
                // the run unstyled where the buffered render bolds it.
                if ( (c == '~' || c == '*') && i + 1 == s.size() )
                    return i;

                ++i;
            }

            // The buffer ended inside a line's leading indent, so the character that
            // decides whether this is a list item has not arrived. Hold the indent with
            // it: releasing it would emit a run of spaces, leave at_line_start_ false,
            // and cost the next chunk's marker its bullet. Only indented markers can
            // split this way -- a column-zero marker is caught by the i + 1 == s.size()
            // test above -- which is why nested items were the ones that lost the glyph.
            if ( line_start_run && indent_start < s.size() )
                return indent_start;

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
}
