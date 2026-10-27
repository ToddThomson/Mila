/**
 * @file HtmlText.Cpu.cpp
 * @brief A web page's HTML read as text: what is kept, what is dropped, and markup broken enough to confuse it.
 */

#include <gtest/gtest.h>
#include <string>

import Desktop.HtmlText;

namespace Mila::Tests::DesktopApp
{
    using Mila::DesktopApp::pageTitle;
    using Mila::DesktopApp::readableText;

    TEST( HtmlText, ParagraphsAreSeparatedByABlankLineAndSpacesCollapse )
    {
        EXPECT_EQ( readableText( "<p>One   two\n three</p><p>Four</p>" ), "One two three\n\nFour" );
    }

    TEST( HtmlText, HeadingsListsAndBreaksKeepTheirShape )
    {
        const std::string html = "<h1>Title</h1><p>Intro</p><h3>Part</h3><ul><li>first</li><li>second</li></ul>a<br>b";

        EXPECT_EQ( readableText( html ), "# Title\n\nIntro\n\n### Part\n\n- first\n- second\na\nb" );
    }

    TEST( HtmlText, ScriptsStylesTheHeadAndNavigationAreDropped )
    {
        const std::string html = "<html><head><title>T</title><style>p{color:red}</style></head><body>"
            "<nav><a href=\"/\">Home</a></nav><script>var a = '<p>no</p>';</script><p>Kept</p>"
            "<footer>Copyright</footer></body></html>";

        EXPECT_EQ( readableText( html ), "Kept" );
    }

    TEST( HtmlText, ALinkToAWebAddressKeepsItAfterItsText )
    {
        const std::string html = "<p>See <a href=\"https://example.org/a?b=1&amp;c=2\">the docs</a> and "
            "<a href=\"/local\">this</a>.</p>";

        EXPECT_EQ( readableText( html ), "See the docs (https://example.org/a?b=1&c=2) and this." );
    }

    TEST( HtmlText, CharacterReferencesAreWrittenOut )
    {
        EXPECT_EQ( readableText( "<p>a &lt; b &amp;&amp; c &#8212; d &#x41; &mdash; &unknown; &nbsp;e</p>" ),
            "a < b && c \xE2\x80\x94 d A \xE2\x80\x94 &unknown; e" );
    }

    TEST( HtmlText, PreformattedTextKeepsItsLines )
    {
        EXPECT_EQ( readableText( "<p>Code:</p><pre>int a;\n  int b;</pre><p>After</p>" ),
            "Code:\n\nint a;\n  int b;\n\nAfter" );
    }

    TEST( HtmlText, TableCellsAreSeparated )
    {
        EXPECT_EQ( readableText( "<table><tr><th>Name</th><th>Size</th></tr><tr><td>a</td><td>1</td></tr></table>" ),
            "| Name | Size\n| a | 1" );
    }

    TEST( HtmlText, CommentsAndAQuotedGreaterThanInAnAttributeDoNotEndText )
    {
        EXPECT_EQ( readableText( "<!-- <p>hidden</p> --><p title=\"a > b\">Shown</p>" ), "Shown" );
    }

    TEST( HtmlText, BrokenMarkupReadsAsTextNotAnError )
    {
        EXPECT_EQ( readableText( "<p>Fine</p><p>Cut off <b" ), "Fine\n\nCut off <b" );
        EXPECT_EQ( readableText( "a < b and c > d" ), "a < b and c > d" );
    }

    TEST( HtmlText, TheTitleIsReadFromTheHead )
    {
        EXPECT_EQ( pageTitle( "<html><HEAD><Title> Mila &amp; models </Title></HEAD></html>" ), "Mila & models" );
        EXPECT_EQ( pageTitle( "<p>No title</p>" ), "" );
    }
}
