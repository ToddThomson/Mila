/**
 * @file RichText.Cpu.cpp
 * @brief Behaviour pins for the chat harness's rich-text formatter.
 *
 * Chat.RichText imports nothing -- pure std string work -- so this target links no
 * Mila, no CUDA and no gtest fixture support beyond the basics. That is the point:
 * the formatter is the cheapest high-value test surface in the adaptor, and until
 * now the only thing checking it was a model happening to emit the right shape.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

import Chat.RichText;

namespace Mila::Tests::ChatApp
{
    using Mila::ChatApp::RichText;

    // ---- wordWrap ---------------------------------------------------------

    TEST( ChatRichTextWordWrap, WrapsGreedilyAtWidth )
    {
        const auto lines = RichText::wordWrap( "aaa bbb ccc ddd", 7 );

        ASSERT_EQ( lines.size(), 2u );
        EXPECT_EQ( lines[ 0 ], "aaa bbb" );
        EXPECT_EQ( lines[ 1 ], "ccc ddd" );
    }

    TEST( ChatRichTextWordWrap, PreservesLeadingIndent )
    {
        const auto lines = RichText::wordWrap( "    indented", 40 );

        ASSERT_EQ( lines.size(), 1u );
        EXPECT_EQ( lines[ 0 ], "    indented" );
    }

    TEST( ChatRichTextWordWrap, HardBreaksAWordLongerThanTheWidth )
    {
        const auto lines = RichText::wordWrap( "abcdefghij", 4 );

        ASSERT_EQ( lines.size(), 3u );
        EXPECT_EQ( lines[ 0 ], "abcd" );
        EXPECT_EQ( lines[ 1 ], "efgh" );
        EXPECT_EQ( lines[ 2 ], "ij" );
    }

    TEST( ChatRichTextWordWrap, KeepsBlankLinesAsEmptyLines )
    {
        const auto lines = RichText::wordWrap( "one\n\ntwo", 40 );

        ASSERT_EQ( lines.size(), 3u );
        EXPECT_EQ( lines[ 0 ], "one" );
        EXPECT_EQ( lines[ 1 ], "" );
        EXPECT_EQ( lines[ 2 ], "two" );
    }

    TEST( ChatRichTextWordWrap, ExpandsTabsToFourColumns )
    {
        const auto lines = RichText::wordWrap( "\tx", 40 );

        ASSERT_EQ( lines.size(), 1u );
        EXPECT_EQ( lines[ 0 ], "    x" );
    }

    // ---- list markers -----------------------------------------------------

    // U+2022, spelled as bytes: the build sets no execution charset, so a literal
    // glyph here would encode by the compiler's codepage rather than as UTF-8.
    constexpr const char* kBullet = "\xE2\x80\xA2";

    TEST( ChatRichTextLists, ConvertsATopLevelMarker )
    {
        EXPECT_EQ( RichText::formatRich( "* item" ), std::string( kBullet ) + " item" );
        EXPECT_EQ( RichText::formatRich( "- item" ), std::string( kBullet ) + " item" );
        EXPECT_EQ( RichText::formatRich( "+ item" ), std::string( kBullet ) + " item" );
    }

    TEST( ChatRichTextLists, ConvertsAnIndentedMarkerAndKeepsTheIndent )
    {
        EXPECT_EQ( RichText::formatRich( "    * nested" ),
            std::string( "    " ) + kBullet + " nested" );
    }

    TEST( ChatRichTextLists, LeavesAMarkerCharacterMidLineAlone )
    {
        // Not a list: no bullet, and the asterisk is erased as emphasis punctuation.
        EXPECT_EQ( RichText::formatRich( "2 * 3 = 6" ), "2  3 = 6" );
    }

    TEST( ChatRichTextLists, RequiresATrailingSpaceToBeAMarker )
    {
        EXPECT_EQ( RichText::formatRich( "*emphasis*" ), "emphasis" );
    }

    // A chunk that begins mid-line must not treat its first characters as a line
    // start -- this is the flag the streaming formatter threads through, and the
    // defect it hid was nested bullets losing their glyph.
    TEST( ChatRichTextLists, HonoursTheAtLineStartFlag )
    {
        EXPECT_EQ( RichText::formatRich( "* item", false ), " item" );
        EXPECT_EQ( RichText::formatRich( "* item", true ), std::string( kBullet ) + " item" );
    }

    TEST( ChatRichTextLists, ALineAfterANewlineIsAlwaysALineStart )
    {
        EXPECT_EQ( RichText::formatRich( "text\n* item", false ),
            std::string( "text\n" ) + kBullet + " item" );
    }

    // ---- LaTeX and symbols ------------------------------------------------

    TEST( ChatRichTextSymbols, ConvertsSymbolCommands )
    {
        EXPECT_EQ( RichText::formatRich( "3 \\times 4" ), "3 \xC3\x97 4" );
        EXPECT_EQ( RichText::formatRich( "a \\leq b" ), "a \xE2\x89\xA4 b" );
        EXPECT_EQ( RichText::formatRich( "x \\to y" ), "x \xE2\x86\x92 y" );
    }

    TEST( ChatRichTextSymbols, ConvertsAFractionToASlash )
    {
        EXPECT_EQ( RichText::formatRich( "\\frac{a}{b}" ), "a/b" );
    }

    TEST( ChatRichTextSymbols, DropsMathDelimiters )
    {
        EXPECT_EQ( RichText::formatRich( "$x + y$" ), "x + y" );
        EXPECT_EQ( RichText::formatRich( "$$x$$" ), "x" );
    }

    TEST( ChatRichTextSymbols, ConvertsDigitSubscriptsAndSuperscripts )
    {
        EXPECT_EQ( RichText::formatRich( "H_2O" ), "H\xE2\x82\x82O" );
        EXPECT_EQ( RichText::formatRich( "x^2" ), "x\xC2\xB2" );
    }

    // A snake_case identifier is not a subscript -- the underscore is followed by a
    // letter, which is overwhelmingly more likely an identifier than a script.
    TEST( ChatRichTextSymbols, LeavesSnakeCaseIdentifiersReadable )
    {
        EXPECT_EQ( RichText::formatRich( "max_width" ), "max_width" );
    }

    // ---- emphasis ---------------------------------------------------------

    TEST( ChatRichTextEmphasis, KeepsTheTextAndDropsTheMarkers )
    {
        EXPECT_EQ( RichText::formatRich( "**bold**" ), "bold" );
        EXPECT_EQ( RichText::formatRich( "__bold__" ), "bold" );
        EXPECT_EQ( RichText::formatRich( "~~struck~~" ), "struck" );
        EXPECT_EQ( RichText::formatRich( "`code`" ), "code" );
    }

    TEST( ChatRichTextEmphasis, KeepsABoldLabelInsideAListItem )
    {
        EXPECT_EQ( RichText::formatRich( "* **Birth:** gravity pulls" ),
            std::string( kBullet ) + " Birth: gravity pulls" );
    }

    // ---- style attributes -------------------------------------------------

    /// Attributes as one readable character per byte: b bold, H heading, . plain.
    std::string sketch( const Mila::ChatApp::StyledText& styled )
    {
        std::string out;

        for ( char attribute : styled.attributes )
        {
            const auto style = static_cast<unsigned char>( attribute );

            out += (style & Mila::ChatApp::StyleHeading) ? 'H'
                 : (style & Mila::ChatApp::StyleBold)    ? 'b'
                 : '.';
        }

        return out;
    }

    TEST( ChatRichTextStyle, EveryByteCarriesAnAttribute )
    {
        const auto styled = RichText::formatRichStyled( "plain" );

        EXPECT_EQ( styled.text, "plain" );
        EXPECT_EQ( styled.attributes.size(), styled.text.size() );
        EXPECT_EQ( sketch( styled ), "....." );
    }

    TEST( ChatRichTextStyle, AHeadingLosesItsMarkerAndFlagsTheLine )
    {
        const auto styled = RichText::formatRichStyled( "### Nuclear Fusion" );

        EXPECT_EQ( styled.text, "Nuclear Fusion" );
        EXPECT_EQ( sketch( styled ), "HHHHHHHHHHHHHH" );
    }

    TEST( ChatRichTextStyle, AllHeadingLevelsGetOneTreatment )
    {
        for ( const auto* source : { "# T", "## T", "### T", "#### T", "##### T", "###### T" } )
        {
            const auto styled = RichText::formatRichStyled( source );

            EXPECT_EQ( styled.text, "T" ) << source;
            EXPECT_EQ( sketch( styled ), "H" ) << source;
        }
    }

    TEST( ChatRichTextStyle, SevenHashesIsNotAHeading )
    {
        const auto styled = RichText::formatRichStyled( "####### T" );

        EXPECT_EQ( styled.text, "####### T" );
        EXPECT_EQ( sketch( styled ), "........." );
    }

    TEST( ChatRichTextStyle, AHashWithoutASpaceIsNotAHeading )
    {
        const auto styled = RichText::formatRichStyled( "#hashtag" );

        EXPECT_EQ( styled.text, "#hashtag" );
        EXPECT_EQ( sketch( styled ), "........" );
    }

    TEST( ChatRichTextStyle, AHeadingDropsItsIndent )
    {
        const auto styled = RichText::formatRichStyled( "  ## Title" );

        EXPECT_EQ( styled.text, "Title" );
        EXPECT_EQ( sketch( styled ), "HHHHH" );
    }

    TEST( ChatRichTextStyle, HeadingAppliesToThatLineOnly )
    {
        const auto styled = RichText::formatRichStyled( "## Title\nbody" );

        EXPECT_EQ( styled.text, "Title\nbody" );
        EXPECT_EQ( sketch( styled ), "HHHHH....." );
    }

    TEST( ChatRichTextStyle, BoldMarkersGoAndTheirTextIsFlagged )
    {
        const auto styled = RichText::formatRichStyled( "**Birth:** gravity" );

        EXPECT_EQ( styled.text, "Birth: gravity" );
        EXPECT_EQ( sketch( styled ), "bbbbbb........" );
    }

    TEST( ChatRichTextStyle, UnderscoreBoldBehavesLikeAsteriskBold )
    {
        EXPECT_EQ( sketch( RichText::formatRichStyled( "__x__ y" ) ), "b.." );
    }

    TEST( ChatRichTextStyle, BoldInsideAListItemFlagsOnlyTheLabel )
    {
        const auto styled = RichText::formatRichStyled( "* **Birth:** gravity" );

        EXPECT_EQ( styled.text, std::string( kBullet ) + " Birth: gravity" );
        //          bullet is 3 bytes, then a space, then the bold label.
        EXPECT_EQ( sketch( styled ), "....bbbbbb........" );
    }

    // An unclosed ** would otherwise bold every remaining line of the response.
    TEST( ChatRichTextStyle, BoldDoesNotSurviveALineBreak )
    {
        const auto styled = RichText::formatRichStyled( "**open\nnext" );

        EXPECT_EQ( styled.text, "open\nnext" );
        EXPECT_EQ( sketch( styled ), "bbbb....." );
    }

    // ---- attributes through the wrap --------------------------------------

    TEST( ChatRichTextStyledWrap, CarriesAttributesOntoEachLine )
    {
        const auto lines = RichText::wordWrap(
            RichText::formatRichStyled( "**aaa bbb** ccc" ), 7 );

        ASSERT_EQ( lines.size(), 2u );
        EXPECT_EQ( lines[ 0 ].text, "aaa bbb" );
        EXPECT_EQ( sketch( lines[ 0 ] ), "bbbbbbb" );
        EXPECT_EQ( lines[ 1 ].text, "ccc" );
        EXPECT_EQ( sketch( lines[ 1 ] ), "..." );
    }

    // A heading long enough to wrap is still a heading on its continuation.
    TEST( ChatRichTextStyledWrap, AWrappedHeadingStaysAHeading )
    {
        const auto lines = RichText::wordWrap(
            RichText::formatRichStyled( "## aaa bbb ccc" ), 7 );

        ASSERT_EQ( lines.size(), 2u );
        EXPECT_EQ( sketch( lines[ 0 ] ), "HHHHHHH" );
        EXPECT_EQ( sketch( lines[ 1 ] ), "HHH" );
    }

    TEST( ChatRichTextStyledWrap, EveryLineHasOneAttributePerByte )
    {
        const auto lines = RichText::wordWrap(
            RichText::formatRichStyled( "## Title\n\n* **a** bb cc dd ee" ), 8 );

        for ( const auto& line : lines )
            EXPECT_EQ( line.attributes.size(), line.text.size() ) << line.text;
    }

    TEST( ChatRichTextStyledWrap, PlainOverloadAgreesWithTheStyledOne )
    {
        const std::string source = "## Title\n\n* **Birth:** gravity pulls gas together";

        const auto plain  = RichText::wordWrap( RichText::formatRich( source ), 12 );
        const auto styled = RichText::wordWrap( RichText::formatRichStyled( source ), 12 );

        ASSERT_EQ( plain.size(), styled.size() );

        for ( size_t line = 0; line < plain.size(); ++line )
            EXPECT_EQ( plain[ line ], styled[ line ].text ) << "line " << line;
    }
}
