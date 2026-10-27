/**
 * @file Assistant.HtmlText.ixx
 * @brief A web page's HTML as the text a model reads: headings, paragraphs, lists, tables and links kept, scripts,
 *        styles and page furniture dropped. Imports nothing of Mila's, so it is tested on its own.
 */

export module Assistant.HtmlText;

import std;

namespace Mila::Assistant
{
    namespace HtmlTextDetail
    {
        /// Elements whose content is never text for a reader.
        inline constexpr std::array<std::string_view, 12> kSkipped{
            "script", "style", "noscript", "svg", "template", "head", "iframe", "canvas", "nav", "footer", "button", "select" };

        /// Elements that end a line where they open and where they close.
        inline constexpr std::array<std::string_view, 24> kBlocks{
            "p", "div", "br", "li", "tr", "ul", "ol", "table", "section", "article", "header", "main", "aside",
            "blockquote", "pre", "hr", "dd", "dt", "dl", "figure", "figcaption", "h1", "h2", "h3" };

        inline bool contains( std::span<const std::string_view> names, std::string_view name )
        {
            return std::ranges::find( names, name ) != names.end();
        }

        inline std::string lowercase( std::string_view text )
        {
            std::string lowered( text );

            for ( char& character : lowered )
                character = static_cast<char>( std::tolower( static_cast<unsigned char>( character ) ) );

            return lowered;
        }

        inline bool isSpace( char character )
        {
            return character == ' ' || character == '\t' || character == '\n' || character == '\r' || character == '\f';
        }

        inline void appendUtf8( std::string& out, std::uint32_t code )
        {
            if ( code == 0 || code > 0x10FFFF || ( code >= 0xD800 && code <= 0xDFFF ) )
                code = 0xFFFD;

            if ( code < 0x80 )
            {
                out.push_back( static_cast<char>( code ) );
            }
            else if ( code < 0x800 )
            {
                out.push_back( static_cast<char>( 0xC0 | ( code >> 6 ) ) );
                out.push_back( static_cast<char>( 0x80 | ( code & 0x3F ) ) );
            }
            else if ( code < 0x10000 )
            {
                out.push_back( static_cast<char>( 0xE0 | ( code >> 12 ) ) );
                out.push_back( static_cast<char>( 0x80 | ( ( code >> 6 ) & 0x3F ) ) );
                out.push_back( static_cast<char>( 0x80 | ( code & 0x3F ) ) );
            }
            else
            {
                out.push_back( static_cast<char>( 0xF0 | ( code >> 18 ) ) );
                out.push_back( static_cast<char>( 0x80 | ( ( code >> 12 ) & 0x3F ) ) );
                out.push_back( static_cast<char>( 0x80 | ( ( code >> 6 ) & 0x3F ) ) );
                out.push_back( static_cast<char>( 0x80 | ( code & 0x3F ) ) );
            }
        }

        struct NamedEntity
        {
            std::string_view name;
            std::uint32_t code;
        };

        /// The entities pages use for text; any other is kept as written.
        inline constexpr std::array<NamedEntity, 20> kEntities{ {
            { "amp", '&' }, { "lt", '<' }, { "gt", '>' }, { "quot", '"' }, { "apos", '\'' }, { "nbsp", ' ' },
            { "mdash", 0x2014 }, { "ndash", 0x2013 }, { "hellip", 0x2026 }, { "lsquo", 0x2018 }, { "rsquo", 0x2019 },
            { "ldquo", 0x201C }, { "rdquo", 0x201D }, { "copy", 0xA9 }, { "reg", 0xAE }, { "trade", 0x2122 },
            { "middot", 0xB7 }, { "bull", 0x2022 }, { "times", 0xD7 }, { "deg", 0xB0 } } };

        /// Text between tags with its character references written out.
        inline std::string decodeEntities( std::string_view text )
        {
            std::string out;
            out.reserve( text.size() );

            for ( std::size_t at = 0; at < text.size(); ++at )
            {
                if ( text[ at ] != '&' )
                {
                    out.push_back( text[ at ] );

                    continue;
                }

                const std::size_t end = text.find( ';', at + 1 );

                if ( end == std::string_view::npos || end - at > 10 )
                {
                    out.push_back( '&' );

                    continue;
                }

                const std::string_view name = text.substr( at + 1, end - at - 1 );

                if ( name.starts_with( '#' ) && name.size() > 1 )
                {
                    const bool hexadecimal = name[ 1 ] == 'x' || name[ 1 ] == 'X';
                    const std::string_view digits = name.substr( hexadecimal ? 2 : 1 );
                    std::uint32_t code = 0;
                    const auto [stopped, error] = std::from_chars( digits.data(), digits.data() + digits.size(), code,
                        hexadecimal ? 16 : 10 );

                    if ( error == std::errc{} && stopped == digits.data() + digits.size() && !digits.empty() )
                    {
                        appendUtf8( out, code );
                        at = end;

                        continue;
                    }
                }

                const auto named = std::ranges::find( kEntities, name, &NamedEntity::name );

                if ( named != kEntities.end() )
                {
                    appendUtf8( out, named->code );
                    at = end;

                    continue;
                }

                out.push_back( '&' );
            }

            return out;
        }

        /// One tag as read: its name in lower case, whether it closes, and its href when it is a link.
        struct Tag
        {
            std::string name;
            bool closing{ false };
            std::string href;
        };

        /// The value of an attribute in a tag's text, quoted or not; empty when absent.
        inline std::string attribute( std::string_view tag_text, std::string_view name )
        {
            const std::string lowered = lowercase( tag_text );
            std::size_t at = 0;

            while ( ( at = lowered.find( name, at ) ) != std::string::npos )
            {
                const bool starts = at > 0 && isSpace( lowered[ at - 1 ] );
                std::size_t after = at + name.size();

                while ( after < lowered.size() && isSpace( lowered[ after ] ) )
                    ++after;

                if ( !starts || after >= lowered.size() || lowered[ after ] != '=' )
                {
                    at += name.size();

                    continue;
                }

                ++after;

                while ( after < tag_text.size() && isSpace( tag_text[ after ] ) )
                    ++after;

                if ( after < tag_text.size() && ( tag_text[ after ] == '"' || tag_text[ after ] == '\'' ) )
                {
                    const char quote = tag_text[ after ];
                    const std::size_t close = tag_text.find( quote, after + 1 );

                    return std::string( tag_text.substr( after + 1, close == std::string_view::npos ? std::string_view::npos : close - after - 1 ) );
                }

                std::size_t end = after;

                while ( end < tag_text.size() && !isSpace( tag_text[ end ] ) && tag_text[ end ] != '>' )
                    ++end;

                return std::string( tag_text.substr( after, end - after ) );
            }

            return {};
        }

        inline Tag readTag( std::string_view tag_text )
        {
            Tag tag;
            std::size_t at = 0;

            if ( at < tag_text.size() && tag_text[ at ] == '/' )
            {
                tag.closing = true;
                ++at;
            }

            std::size_t end = at;

            while ( end < tag_text.size() && ( std::isalnum( static_cast<unsigned char>( tag_text[ end ] ) ) != 0 ) )
                ++end;

            tag.name = lowercase( tag_text.substr( at, end - at ) );

            if ( tag.name == "a" && !tag.closing )
                tag.href = decodeEntities( attribute( tag_text, "href" ) );

            return tag;
        }

        /// Where the tag that opens at @p open ends, past quoted attribute values; npos when it never does.
        inline std::size_t tagEnd( std::string_view html, std::size_t open )
        {
            char quote = 0;

            for ( std::size_t at = open + 1; at < html.size(); ++at )
            {
                const char character = html[ at ];

                if ( quote != 0 )
                {
                    if ( character == quote )
                        quote = 0;

                    continue;
                }

                if ( character == '"' || character == '\'' )
                {
                    // A quote only opens a value after an equals sign; elsewhere it is text in a broken tag.
                    std::size_t before = at;

                    while ( before > open + 1 && isSpace( html[ before - 1 ] ) )
                        --before;

                    if ( html[ before - 1 ] == '=' )
                        quote = character;

                    continue;
                }

                if ( character == '>' )
                    return at;
            }

            return std::string_view::npos;
        }

        /// The text a reader sees, written as it arrives: spaces collapsed, lines and blank lines kept to one.
        class Writer
        {
        public:

            void text( std::string_view piece, bool preformatted )
            {
                for ( const char character : piece )
                {
                    if ( preformatted )
                    {
                        if ( character == '\r' )
                            continue;

                        out_.push_back( character );
                        pending_space_ = false;

                        continue;
                    }

                    if ( isSpace( character ) )
                    {
                        pending_space_ = !out_.empty() && out_.back() != '\n';

                        continue;
                    }

                    if ( pending_space_ )
                        out_.push_back( ' ' );

                    pending_space_ = false;
                    out_.push_back( character );
                }
            }

            void lineBreak()
            {
                pending_space_ = false;

                while ( !out_.empty() && out_.back() == ' ' )
                    out_.pop_back();

                if ( out_.empty() || out_.back() == '\n' )
                    return;

                out_.push_back( '\n' );
            }

            void blankLine()
            {
                lineBreak();

                if ( !out_.empty() && !out_.ends_with( "\n\n" ) )
                    out_.push_back( '\n' );
            }

            /// A table cell begins: `| ` at the start of a row, ` | ` between cells.
            void cell()
            {
                pending_space_ = false;

                while ( !out_.empty() && out_.back() == ' ' )
                    out_.pop_back();

                out_ += out_.empty() || out_.back() == '\n' ? "| " : " | ";
            }

            void startLine( std::string_view prefix )
            {
                lineBreak();
                out_ += prefix;
            }

            std::string finish()
            {
                lineBreak();

                while ( !out_.empty() && out_.back() == '\n' )
                    out_.pop_back();

                return std::move( out_ );
            }

        private:

            std::string out_;
            bool pending_space_{ false };
        };
    }

    /**
     * @brief The text of a page's `<title>`, its references written out; empty when it has none.
     */
    export std::string pageTitle( std::string_view html )
    {
        using namespace HtmlTextDetail;

        const std::string lowered = lowercase( html );
        const std::size_t open = lowered.find( "<title" );

        if ( open == std::string::npos )
            return {};

        const std::size_t start = lowered.find( '>', open );
        const std::size_t close = lowered.find( "</title", start );

        if ( start == std::string::npos || close == std::string::npos )
            return {};

        Writer writer;
        writer.text( decodeEntities( html.substr( start + 1, close - start - 1 ) ), false );

        return writer.finish();
    }

    /**
     * @brief A page's HTML as readable text.
     *
     * Headings become `#` lines, list items `- ` lines, table cells are separated by ` | `, and a link to a web
     * address keeps it after its text, as `text (address)`, so a model can follow it. Scripts, styles, the head,
     * navigation and footers are dropped. Not a parser: a page broken enough to confuse it reads as its text with
     * stray markup, never as an error.
     */
    export std::string readableText( std::string_view html )
    {
        using namespace HtmlTextDetail;

        Writer writer;
        std::string skipping;
        int preformatted = 0;
        std::vector<std::string> links;
        std::size_t at = 0;

        while ( at < html.size() )
        {
            const std::size_t open = html.find( '<', at );
            const std::size_t text_end = open == std::string_view::npos ? html.size() : open;

            if ( skipping.empty() && text_end > at )
                writer.text( decodeEntities( html.substr( at, text_end - at ) ), preformatted > 0 );

            if ( open == std::string_view::npos )
                break;

            if ( html.substr( open ).starts_with( "<!--" ) )
            {
                const std::size_t close = html.find( "-->", open + 4 );
                at = close == std::string_view::npos ? html.size() : close + 3;

                continue;
            }

            const std::size_t close = tagEnd( html, open );

            if ( close == std::string_view::npos )
            {
                if ( skipping.empty() )
                    writer.text( html.substr( open ), preformatted > 0 );

                break;
            }

            at = close + 1;

            const std::string_view tag_text = html.substr( open + 1, close - open - 1 );

            if ( tag_text.empty() || tag_text.front() == '!' || tag_text.front() == '?' )
                continue;

            const Tag tag = readTag( tag_text );

            if ( tag.name.empty() )
            {
                if ( skipping.empty() )
                    writer.text( html.substr( open, close - open + 1 ), preformatted > 0 );

                continue;
            }

            if ( !skipping.empty() )
            {
                if ( tag.closing && tag.name == skipping )
                    skipping.clear();

                continue;
            }

            const bool self_closing = tag_text.ends_with( '/' );

            if ( !tag.closing && contains( kSkipped, tag.name ) && !self_closing )
            {
                skipping = tag.name;

                continue;
            }

            if ( tag.name.size() == 2 && tag.name[ 0 ] == 'h' && tag.name[ 1 ] >= '1' && tag.name[ 1 ] <= '6' )
            {
                if ( tag.closing )
                    writer.blankLine();
                else
                {
                    writer.blankLine();
                    writer.startLine( std::string( static_cast<std::size_t>( tag.name[ 1 ] - '0' ), '#' ) + " " );
                }

                continue;
            }

            if ( tag.name == "li" && !tag.closing )
            {
                writer.startLine( "- " );

                continue;
            }

            if ( ( tag.name == "td" || tag.name == "th" ) && !tag.closing )
            {
                writer.cell();

                continue;
            }

            if ( tag.name == "pre" )
                preformatted = std::max( 0, preformatted + ( tag.closing ? -1 : 1 ) );

            if ( tag.name == "p" || tag.name == "blockquote" || tag.name == "pre" || tag.name == "table" )
            {
                writer.blankLine();

                continue;
            }

            if ( contains( kBlocks, tag.name ) )
            {
                writer.lineBreak();

                continue;
            }

            if ( tag.name == "a" )
            {
                if ( !tag.closing )
                {
                    links.push_back( tag.href );

                    continue;
                }

                if ( !links.empty() )
                {
                    const std::string href = std::move( links.back() );
                    links.pop_back();

                    if ( href.starts_with( "http://" ) || href.starts_with( "https://" ) )
                        writer.text( " (" + href + ")", true );
                }
            }
        }

        return writer.finish();
    }
}
