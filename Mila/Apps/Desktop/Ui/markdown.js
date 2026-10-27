// Markdown to HTML for the subset MilaApp.md section 3 names. The input is model text, which
// is untrusted: every character of it is escaped on the way out, raw HTML renders as text, a link
// is an element the page turns into link.open, and an image is never fetched.

const fencePattern = /^ {0,3}(`{3,}|~{3,})\s*([^`\s]*)[^`]*$/;
const headingPattern = /^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$/;
const rulePattern = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
const quotePattern = /^ {0,3}> ?(.*)$/;
const listItemPattern = /^( *)([-*+]|\d{1,9}[.)])(?:[ \t]+(.*))?$/;
const tableDelimiterPattern = /^ *\|? *:?-+:? *(?:\| *:?-+:? *)*\|? *$/;
const safeUrlPattern = /^(?:https?:|mailto:)/i;
const bareUrlPattern = /^https?:\/\/[^\s<>"'`]+/i;

export function escapeHtml( text )
{
    return text
        .replace( /&/g, '&amp;' )
        .replace( /</g, '&lt;' )
        .replace( />/g, '&gt;' )
        .replace( /"/g, '&quot;' )
        .replace( /'/g, '&#39;' );
}

const caretHtml = '<span class="caret" aria-hidden="true"></span>';

/**
 * Render markdown to an HTML string. With options.caret the streaming caret is placed at the end
 * of the last text, wherever the last block is.
 */
export function renderMarkdown( text, options )
{
    const caret = options !== undefined && options.caret === true;
    const normalized = text.replace( /\r\n?/g, '\n' );
    const lines = normalized.split( '\n' );
    const blocks = parseBlocks( lines );

    return renderBlocks( blocks, caret );
}

// ---- Blocks ----------------------------------------------------------------------------------

function isBlank( line )
{
    return line.trim().length === 0;
}

function parseBlocks( lines )
{
    const blocks = [];
    let index = 0;

    while ( index < lines.length )
    {
        const line = lines[ index ];

        if ( isBlank( line ) )
        {
            index = index + 1;

            continue;
        }

        const fence = fencePattern.exec( line );

        if ( fence !== null )
        {
            const parsed = parseFence( lines, index, fence );
            blocks.push( parsed.block );
            index = parsed.next;

            continue;
        }

        const heading = headingPattern.exec( line );

        if ( heading !== null )
        {
            const content = heading[ 2 ] === undefined ? '' : heading[ 2 ];
            const block = { kind: 'heading', level: heading[ 1 ].length, text: content };
            blocks.push( block );
            index = index + 1;

            continue;
        }

        if ( rulePattern.test( line ) )
        {
            const block = { kind: 'rule' };
            blocks.push( block );
            index = index + 1;

            continue;
        }

        if ( quotePattern.test( line ) )
        {
            const parsed = parseQuote( lines, index );
            blocks.push( parsed.block );
            index = parsed.next;

            continue;
        }

        if ( startsTable( lines, index ) )
        {
            const parsed = parseTable( lines, index );
            blocks.push( parsed.block );
            index = parsed.next;

            continue;
        }

        if ( listItemPattern.test( line ) )
        {
            const parsed = parseList( lines, index );
            blocks.push( parsed.block );
            index = parsed.next;

            continue;
        }

        const parsed = parseParagraph( lines, index );
        blocks.push( parsed.block );
        index = parsed.next;
    }

    return blocks;
}

function parseFence( lines, start, fence )
{
    const marker = fence[ 1 ];
    const language = fence[ 2 ];
    const body = [];
    let index = start + 1;
    let closed = false;

    while ( index < lines.length )
    {
        const line = lines[ index ];
        const trimmed = line.trim();

        if ( trimmed.length >= marker.length && trimmed[ 0 ] === marker[ 0 ] && /^(`+|~+)$/.test( trimmed ) )
        {
            closed = true;
            index = index + 1;

            break;
        }

        body.push( line );
        index = index + 1;
    }

    const block = { kind: 'code', language, text: body.join( '\n' ), closed };

    return { block, next: index };
}

function parseQuote( lines, start )
{
    const inner = [];
    let index = start;

    while ( index < lines.length )
    {
        const match = quotePattern.exec( lines[ index ] );

        if ( match === null )
            break;

        inner.push( match[ 1 ] );
        index = index + 1;
    }

    const block = { kind: 'quote', blocks: parseBlocks( inner ) };

    return { block, next: index };
}

function splitRow( line )
{
    let trimmed = line.trim();

    if ( trimmed.startsWith( '|' ) )
        trimmed = trimmed.slice( 1 );

    if ( trimmed.endsWith( '|' ) && !trimmed.endsWith( '\\|' ) )
        trimmed = trimmed.slice( 0, -1 );

    const cells = [];
    let current = '';
    let inCode = false;

    for ( let at = 0; at < trimmed.length; at = at + 1 )
    {
        const character = trimmed[ at ];

        if ( character === '\\' && trimmed[ at + 1 ] === '|' )
        {
            current = current + '|';
            at = at + 1;

            continue;
        }

        if ( character === '`' )
            inCode = !inCode;

        if ( character === '|' && !inCode )
        {
            cells.push( current.trim() );
            current = '';

            continue;
        }

        current = current + character;
    }

    cells.push( current.trim() );

    return cells;
}

function startsTable( lines, index )
{
    if ( index + 1 >= lines.length )
        return false;

    const header = lines[ index ];
    const delimiter = lines[ index + 1 ];

    if ( !header.includes( '|' ) || !delimiter.includes( '-' ) )
        return false;

    return tableDelimiterPattern.test( delimiter );
}

function alignmentOf( cell )
{
    const left = cell.startsWith( ':' );
    const right = cell.endsWith( ':' );

    if ( left && right )
        return 'center';

    if ( right )
        return 'right';

    return left ? 'left' : null;
}

function parseTable( lines, start )
{
    const header = splitRow( lines[ start ] );
    const delimiters = splitRow( lines[ start + 1 ] );
    const alignments = delimiters.map( alignmentOf );
    const rows = [];
    let index = start + 2;

    while ( index < lines.length )
    {
        const line = lines[ index ];

        if ( isBlank( line ) || !line.includes( '|' ) )
            break;

        const row = splitRow( line );
        rows.push( row );
        index = index + 1;
    }

    const block = { kind: 'table', header, alignments, rows };

    return { block, next: index };
}

function indentOf( line )
{
    const match = /^ */.exec( line );

    return match[ 0 ].length;
}

function parseList( lines, start )
{
    const first = listItemPattern.exec( lines[ start ] );
    const baseIndent = first[ 1 ].length;
    const ordered = /\d/.test( first[ 2 ][ 0 ] );
    const startNumber = ordered ? parseInt( first[ 2 ], 10 ) : 1;
    const items = [];
    let current = null;
    let contentIndent = 0;
    let index = start;
    let sawBlank = false;
    let loose = false;

    while ( index < lines.length )
    {
        const line = lines[ index ];
        const item = listItemPattern.exec( line );

        if ( item !== null && item[ 1 ].length === baseIndent && /\d/.test( item[ 2 ][ 0 ] ) === ordered )
        {
            if ( sawBlank && current !== null )
                loose = true;

            const content = item[ 3 ] === undefined ? '' : item[ 3 ];
            current = { lines: [ content ] };
            items.push( current );
            contentIndent = baseIndent + item[ 2 ].length + 1;
            sawBlank = false;
            index = index + 1;

            continue;
        }

        if ( isBlank( line ) )
        {
            sawBlank = true;
            current.lines.push( '' );
            index = index + 1;

            continue;
        }

        const indent = indentOf( line );

        if ( indent > baseIndent )
        {
            const strip = Math.min( indent, contentIndent );
            const stripped = line.slice( strip );
            current.lines.push( stripped );
            index = index + 1;

            continue;
        }

        if ( sawBlank || item !== null || startsAnotherBlock( line ) )
            break;

        current.lines.push( line.trim() );
        index = index + 1;
    }

    const parsedItems = items.map( parseListItem );
    const block = { kind: 'list', ordered, start: startNumber, loose, items: parsedItems };

    return { block, next: index };
}

function parseListItem( item )
{
    return parseBlocks( item.lines );
}

function startsAnotherBlock( line )
{
    return fencePattern.test( line ) || headingPattern.test( line ) || rulePattern.test( line ) || quotePattern.test( line );
}

function parseParagraph( lines, start )
{
    const body = [];
    let index = start;

    while ( index < lines.length )
    {
        const line = lines[ index ];

        if ( isBlank( line ) )
            break;

        if ( index > start && ( startsAnotherBlock( line ) || listItemPattern.test( line ) || startsTable( lines, index ) ) )
            break;

        body.push( line.trim() );
        index = index + 1;
    }

    const block = { kind: 'paragraph', text: body.join( '\n' ) };

    return { block, next: index };
}

// ---- Rendering -------------------------------------------------------------------------------

function renderBlocks( blocks, caret )
{
    const parts = [];

    for ( let index = 0; index < blocks.length; index = index + 1 )
    {
        const last = caret && index === blocks.length - 1;
        const html = renderBlock( blocks[ index ], last, false );
        parts.push( html );
    }

    if ( caret && blocks.length === 0 )
        parts.push( caretHtml );

    return parts.join( '' );
}

function renderBlock( block, caret, tight )
{
    const tail = caret ? caretHtml : '';

    switch ( block.kind )
    {
        case 'heading':
            return '<h' + block.level + '>' + renderInline( block.text ) + tail + '</h' + block.level + '>';

        case 'rule':
            return '<hr>' + tail;

        case 'code':
            return renderCode( block, tail );

        case 'quote':
            return '<blockquote>' + renderBlocks( block.blocks, caret ) + '</blockquote>';

        case 'table':
            return renderTable( block ) + tail;

        case 'list':
            return renderList( block, caret );

        default:
            return tight
                ? renderInline( block.text ) + tail
                : '<p>' + renderInline( block.text ) + tail + '</p>';
    }
}

function renderCode( block, tail )
{
    const language = block.language.length > 0 ? escapeHtml( block.language ) : 'text';
    const languageClass = block.language.length > 0 ? ' class="language-' + escapeHtml( block.language ) + '"' : '';

    return '<div class="code-block">'
        + '<div class="code-header"><span class="code-language">' + language + '</span>'
        + '<button class="code-copy" type="button" data-action="copy-code">Copy</button></div>'
        + '<pre><code' + languageClass + '>' + escapeHtml( block.text ) + tail + '</code></pre>'
        + '</div>';
}

function alignmentClass( alignment )
{
    return alignment === null || alignment === undefined ? '' : ' class="align-' + alignment + '"';
}

function renderTable( block )
{
    const headerCells = block.header.map( ( cell, column ) => '<th' + alignmentClass( block.alignments[ column ] ) + '>' + renderInline( cell ) + '</th>' );
    const bodyRows = block.rows.map( ( row ) => renderTableRow( row, block ) );

    return '<div class="table-scroll"><table><thead><tr>' + headerCells.join( '' ) + '</tr></thead>'
        + '<tbody>' + bodyRows.join( '' ) + '</tbody></table></div>';
}

function renderTableRow( row, block )
{
    const cells = [];

    for ( let column = 0; column < block.header.length; column = column + 1 )
    {
        const cell = column < row.length ? row[ column ] : '';
        const html = '<td' + alignmentClass( block.alignments[ column ] ) + '>' + renderInline( cell ) + '</td>';
        cells.push( html );
    }

    return '<tr>' + cells.join( '' ) + '</tr>';
}

function renderList( block, caret )
{
    const tag = block.ordered ? 'ol' : 'ul';
    const start = block.ordered && block.start !== 1 ? ' start="' + block.start + '"' : '';
    const items = [];

    for ( let index = 0; index < block.items.length; index = index + 1 )
    {
        const last = caret && index === block.items.length - 1;
        const html = renderListItem( block.items[ index ], block.loose, last );
        items.push( html );
    }

    return '<' + tag + start + '>' + items.join( '' ) + '</' + tag + '>';
}

function renderListItem( blocks, loose, caret )
{
    const parts = [];

    for ( let index = 0; index < blocks.length; index = index + 1 )
    {
        const last = caret && index === blocks.length - 1;
        const tight = !loose && blocks[ index ].kind === 'paragraph';
        const html = renderBlock( blocks[ index ], last, tight );
        parts.push( html );
    }

    if ( caret && blocks.length === 0 )
        parts.push( caretHtml );

    return '<li>' + parts.join( '' ) + '</li>';
}

// ---- Inline ----------------------------------------------------------------------------------

function renderLink( label, url )
{
    const escapedUrl = escapeHtml( url );

    return '<a class="link" role="link" tabindex="0" data-url="' + escapedUrl + '" title="' + escapedUrl + '">' + label + '</a>';
}

function findClosingBracket( text, start )
{
    let depth = 0;

    for ( let at = start; at < text.length; at = at + 1 )
    {
        const character = text[ at ];

        if ( character === '\\' )
        {
            at = at + 1;

            continue;
        }

        if ( character === '[' )
            depth = depth + 1;

        if ( character === ']' )
        {
            depth = depth - 1;

            if ( depth === 0 )
                return at;
        }
    }

    return -1;
}

/**
 * A [label](destination) at text[start]: the label, the destination and where it ends, or null.
 */
function matchLink( text, start )
{
    const close = findClosingBracket( text, start );

    if ( close < 0 || text[ close + 1 ] !== '(' )
        return null;

    const end = text.indexOf( ')', close + 2 );

    if ( end < 0 )
        return null;

    const label = text.slice( start + 1, close );
    const inside = text.slice( close + 2, end ).trim();
    const destination = inside.split( /\s+/ )[ 0 ].replace( /^<|>$/g, '' );

    return { label, destination, next: end + 1 };
}

function isWordCharacter( character )
{
    return character !== undefined && /[\p{L}\p{N}]/u.test( character );
}

function isWhitespace( character )
{
    return character === undefined || /\s/.test( character );
}

/**
 * The closing delimiter for an emphasis run opened at text[start], or -1. An underscore does
 * not open or close inside a word, so snake_case stays as written.
 */
function findEmphasisClose( text, start, delimiter )
{
    const width = delimiter.length;
    const after = text[ start + width ];

    if ( isWhitespace( after ) )
        return -1;

    if ( delimiter[ 0 ] === '_' && isWordCharacter( text[ start - 1 ] ) )
        return -1;

    let at = start + width;

    while ( at < text.length )
    {
        const found = text.indexOf( delimiter, at );

        if ( found < 0 )
            return -1;

        const before = text[ found - 1 ];
        const following = text[ found + width ];
        const closesWord = delimiter[ 0 ] !== '_' || !isWordCharacter( following );
        const doubled = width === 1 && following === delimiter;

        if ( found > start + width && !isWhitespace( before ) && closesWord && !doubled )
            return found;

        at = doubled ? found + 2 : found + 1;
    }

    return -1;
}

export function renderInline( text )
{
    let html = '';
    let at = 0;

    while ( at < text.length )
    {
        const character = text[ at ];

        if ( character === '\\' && at + 1 < text.length && /[!-\/:-@\[-`{-~]/.test( text[ at + 1 ] ) )
        {
            const escaped = text[ at + 1 ];
            html = html + escapeHtml( escaped );
            at = at + 2;

            continue;
        }

        if ( character === '\n' )
        {
            html = html + '<br>';
            at = at + 1;

            continue;
        }

        if ( character === '`' )
        {
            const rest = text.slice( at );
            const run = /^`+/.exec( rest )[ 0 ];
            const close = text.indexOf( run, at + run.length );

            if ( close > at )
            {
                const raw = text.slice( at + run.length, close );
                const code = raw.replace( /\n/g, ' ' );
                const trimmed = code.length > 2 && code.startsWith( ' ' ) && code.endsWith( ' ' ) ? code.slice( 1, -1 ) : code;
                html = html + '<code>' + escapeHtml( trimmed ) + '</code>';
                at = close + run.length;

                continue;
            }

            html = html + escapeHtml( run );
            at = at + run.length;

            continue;
        }

        if ( character === '!' && text[ at + 1 ] === '[' )
        {
            const image = matchLink( text, at + 1 );

            if ( image !== null )
            {
                html = html + renderImageReference( image );
                at = image.next;

                continue;
            }
        }

        if ( character === '[' )
        {
            const link = matchLink( text, at );

            if ( link !== null && safeUrlPattern.test( link.destination ) )
            {
                const label = renderInline( link.label );
                html = html + renderLink( label, link.destination );
                at = link.next;

                continue;
            }
        }

        if ( character === '<' )
        {
            const rest = text.slice( at );
            const autolink = /^<((?:https?:|mailto:)[^\s<>]+)>/i.exec( rest );

            if ( autolink !== null )
            {
                const label = escapeHtml( autolink[ 1 ] );
                html = html + renderLink( label, autolink[ 1 ] );
                at = at + autolink[ 0 ].length;

                continue;
            }
        }

        if ( ( character === 'h' || character === 'H' ) && !isWordCharacter( text[ at - 1 ] ) )
        {
            const rest = text.slice( at );
            const bare = bareUrlPattern.exec( rest );

            if ( bare !== null )
            {
                const url = bare[ 0 ].replace( /[.,;:!?)\]]+$/, '' );
                const label = escapeHtml( url );
                html = html + renderLink( label, url );
                at = at + url.length;

                continue;
            }
        }

        if ( character === '~' && text[ at + 1 ] === '~' )
        {
            const close = findEmphasisClose( text, at, '~~' );

            if ( close > 0 )
            {
                const content = text.slice( at + 2, close );
                const inner = renderInline( content );
                html = html + '<del>' + inner + '</del>';
                at = close + 2;

                continue;
            }
        }

        if ( character === '*' || character === '_' )
        {
            const doubled = character + character;
            const strong = text[ at + 1 ] === character ? findEmphasisClose( text, at, doubled ) : -1;

            if ( strong > 0 )
            {
                const content = text.slice( at + 2, strong );
                const inner = renderInline( content );
                html = html + '<strong>' + inner + '</strong>';
                at = strong + 2;

                continue;
            }

            const emphasis = text[ at + 1 ] !== character ? findEmphasisClose( text, at, character ) : -1;

            if ( emphasis > 0 )
            {
                const content = text.slice( at + 1, emphasis );
                const inner = renderInline( content );
                html = html + '<em>' + inner + '</em>';
                at = emphasis + 1;

                continue;
            }
        }

        html = html + escapeHtml( character );
        at = at + 1;
    }

    return html;
}

function renderImageReference( image )
{
    const alternative = image.label.length > 0 ? image.label : image.destination;
    const label = '<span class="image-reference">Image: ' + escapeHtml( alternative ) + '</span>';

    if ( safeUrlPattern.test( image.destination ) )
        return renderLink( label, image.destination );

    return label;
}
