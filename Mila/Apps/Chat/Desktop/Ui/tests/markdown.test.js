import { test } from 'node:test';
import assert from 'node:assert/strict';

import { renderMarkdown, renderInline } from '../markdown.js';

function rawHtmlRendersAsText()
{
    const html = renderMarkdown( '<img src=x onerror=alert(1)> and <script>alert(2)</script>' );

    assert.ok( !html.includes( '<img' ) );
    assert.ok( !html.includes( '<script' ) );
    assert.ok( html.includes( '&lt;img src=x onerror=alert(1)&gt;' ) );
}

test( 'raw HTML renders as text', rawHtmlRendersAsText );

function aLinkBecomesAnElementThePageOpensInTheBrowserNeverAnHref()
{
    const html = renderInline( 'See [the docs](https://example.com/a?b=1&c="2").' );

    assert.ok( html.includes( 'data-url="https://example.com/a?b=1&amp;c=&quot;2&quot;"' ) );
    assert.ok( !html.includes( 'href' ) );
}

test( 'a link becomes an element the page opens in the browser, never an href', aLinkBecomesAnElementThePageOpensInTheBrowserNeverAnHref );

function aLinkToAnythingButHttpHttpsOrMailtoStaysText()
{
    const html = renderInline( '[click](javascript:alert(1))' );

    assert.ok( !html.includes( 'data-url' ) );
}

test( 'a link to anything but http, https or mailto stays text', aLinkToAnythingButHttpHttpsOrMailtoStaysText );

function anImageIsShownAsItsDescriptionNotFetched()
{
    const html = renderInline( '![a cat](https://example.com/cat.png)' );

    assert.ok( !html.includes( '<img' ) );
    assert.ok( html.includes( 'Image: a cat' ) );
}

test( 'an image is shown as its description, not fetched', anImageIsShownAsItsDescriptionNotFetched );

function headingsEmphasisAndInlineCode()
{
    const html = renderMarkdown( '## Title\n\nSome **bold**, *italic* and `code <b>`.' );

    assert.ok( html.includes( '<h2>Title</h2>' ) );
    assert.ok( html.includes( '<strong>bold</strong>' ) );
    assert.ok( html.includes( '<em>italic</em>' ) );
    assert.ok( html.includes( '<code>code &lt;b&gt;</code>' ) );
}

test( 'headings, emphasis and inline code', headingsEmphasisAndInlineCode );

function underscoresInsideAWordAreNotEmphasis()
{
    const html = renderInline( 'call get_weather_now and _this_' );

    assert.ok( html.includes( 'get_weather_now' ) );
    assert.ok( html.includes( '<em>this</em>' ) );
}

test( 'underscores inside a word are not emphasis', underscoresInsideAWordAreNotEmphasis );

function aFencedCodeBlockKeepsItsTextAndNamesItsLanguage()
{
    const html = renderMarkdown( '```python\nif a < b:\n    print("x")\n```' );

    assert.ok( html.includes( '<span class="code-language">python</span>' ) );
    assert.ok( html.includes( 'if a &lt; b:\n    print(&quot;x&quot;)' ) );
    assert.ok( html.includes( 'data-action="copy-code"' ) );
}

test( 'a fenced code block keeps its text and names its language', aFencedCodeBlockKeepsItsTextAndNamesItsLanguage );

function anUnclosedFenceAsWhileStreamingIsStillACodeBlock()
{
    const html = renderMarkdown( 'Here:\n\n```js\nconst a = 1;' );

    assert.ok( html.includes( '<code class="language-js">const a = 1;</code>' ) );
}

test( 'an unclosed fence, as while streaming, is still a code block', anUnclosedFenceAsWhileStreamingIsStillACodeBlock );

function aTableWithAlignment()
{
    const html = renderMarkdown( '| a | b |\n|:--|--:|\n| 1 | `x|y` |' );

    assert.ok( html.includes( '<th class="align-left">a</th>' ) );
    assert.ok( html.includes( '<td class="align-right"><code>x|y</code></td>' ) );
}

test( 'a table with alignment', aTableWithAlignment );

function nestedAndOrderedLists()
{
    const html = renderMarkdown( '1. one\n2. two\n   - inner\n3. three' );

    assert.ok( html.startsWith( '<ol><li>one</li><li>two<ul><li>inner</li></ul></li><li>three</li></ol>' ) );
}

test( 'nested and ordered lists', nestedAndOrderedLists );

function anOrderedListKeepsItsStartNumber()
{
    const html = renderMarkdown( '3. three\n4. four' );

    assert.ok( html.startsWith( '<ol start="3">' ) );
}

test( 'an ordered list keeps its start number', anOrderedListKeepsItsStartNumber );

function mathIsShownAsItsSource()
{
    const html = renderMarkdown( 'Energy is $E = mc^2$.' );

    assert.ok( html.includes( '$E = mc^2$' ) );
}

test( 'math is shown as its source', mathIsShownAsItsSource );

function theCaretGoesAtTheEndOfTheLastTextInsideListsAndCodeToo()
{
    const caret = '<span class="caret" aria-hidden="true"></span>';
    const options = { caret: true };

    assert.ok( renderMarkdown( 'Hello', options ).endsWith( 'Hello' + caret + '</p>' ) );
    assert.ok( renderMarkdown( '- a\n- b', options ).endsWith( 'b' + caret + '</li></ul>' ) );
    assert.ok( renderMarkdown( '```\nx', options ).includes( 'x' + caret + '</code>' ) );
    assert.equal( renderMarkdown( '', options ), caret );
}

test( 'the caret goes at the end of the last text, inside lists and code too', theCaretGoesAtTheEndOfTheLastTextInsideListsAndCodeToo );

function aBareUrlBecomesALinkWithoutItsTrailingPunctuation()
{
    const html = renderInline( 'Go to https://mila.dev/docs.' );

    assert.ok( html.includes( 'data-url="https://mila.dev/docs"' ) );
    assert.ok( html.endsWith( '</a>.' ) );
}

test( 'a bare URL becomes a link, without its trailing punctuation', aBareUrlBecomesALinkWithoutItsTrailingPunctuation );

function aBlockquoteAndARule()
{
    const html = renderMarkdown( '> quoted\n\n---\n\nafter' );

    assert.ok( html.startsWith( '<blockquote><p>quoted</p></blockquote><hr>' ) );
}

test( 'a blockquote and a rule', aBlockquoteAndARule );
