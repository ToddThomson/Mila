// The conversation: user turns, assistant turns built from their blocks in order of occurrence,
// and the session's quiet lines. An entry or a block is rebuilt only when the reducer gave it a
// new object, so a streamed token touches one element.

import { element, button, iconButton, setIcon, clear } from '../dom.js';
import { renderMarkdown } from '../markdown.js';
import { summarizeTurn, isBusy } from '../state.js';
import { thoughtLabel, statsLine, finishNotice, readableJson } from '../format.js';

const stickDistance = 80;

export function createTranscriptView( scroller, container, actions )
{
    const records = new Map();
    const emptyState = element( 'div', 'empty-state' );
    const waiting = element( 'div', 'turn assistant waiting' );
    const waitingCaret = element( 'span', 'caret' );
    let lastLength = 0;

    waiting.append( waitingCaret );

    function nearBottom()
    {
        const remaining = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;

        return remaining < stickDistance;
    }

    function onClick( event )
    {
        const target = event.target.closest( '[data-action]' );

        if ( target === null || !container.contains( target ) )
            return;

        const action = target.dataset.action;

        if ( action === 'toggle' )
            actions.toggleExpanded( target.dataset.identifier );

        if ( action === 'copy-code' )
            copyCode( target );

        if ( action === 'copy-reply' )
            copyReply( target );
    }

    function copyReply( target )
    {
        const record = records.get( target.dataset.identifier );

        if ( record === undefined )
            return;

        const text = replyText( record.entry );
        copyText( text, target );
    }

    container.addEventListener( 'click', onClick );

    function render( state, surface )
    {
        const stick = nearBottom();
        const seen = new Set();
        const children = [];

        for ( const entry of state.transcript )
        {
            seen.add( entry.id );

            const record = renderEntry( entry, surface );
            children.push( record.element );
        }

        for ( const identifier of Array.from( records.keys() ) )
        {
            if ( !seen.has( identifier ) )
                records.delete( identifier );
        }

        const last = state.transcript.length > 0 ? state.transcript[ state.transcript.length - 1 ] : null;
        const awaitingReply = isBusy( state ) && last !== null && last.kind === 'user';

        if ( awaitingReply )
            children.push( waiting );

        if ( state.transcript.length === 0 )
        {
            renderEmptyState( state );
            children.push( emptyState );
        }

        placeChildren( container, children );

        const grew = state.transcript.length > lastLength;
        const sent = grew && last !== null && last.kind === 'user';
        lastLength = state.transcript.length;

        if ( stick || sent )
            scroller.scrollTop = scroller.scrollHeight;
    }

    function renderEmptyState( state )
    {
        clear( emptyState );

        if ( state.session === null )
            return;

        if ( state.session.model === null )
        {
            const heading = element( 'p', 'empty-heading', 'No model is loaded' );
            const reason = state.session.no_model_reason === null ? 'Choose one from the model menu above to start.' : state.session.no_model_reason;
            const detail = element( 'p', 'empty-detail', reason );
            emptyState.append( heading, detail );

            return;
        }

        const heading = element( 'p', 'empty-heading', 'New conversation' );
        const detailText = 'Replies come from ' + state.session.model + ', running on this computer.';
        const detail = element( 'p', 'empty-detail', detailText );
        emptyState.append( heading, detail );
    }

    function renderEntry( entry, surface )
    {
        const existing = records.get( entry.id );

        if ( entry.kind !== 'assistant' )
        {
            if ( existing !== undefined && existing.entry === entry )
                return existing;

            const built = buildSimpleEntry( entry );
            const record = { entry, element: built };
            records.set( entry.id, record );

            return record;
        }

        const record = existing !== undefined ? existing : createAssistantRecord( entry );
        updateAssistant( record, entry, surface );
        records.set( entry.id, record );

        return record;
    }

    return { render };
}

function placeChildren( container, children )
{
    for ( let index = 0; index < children.length; index = index + 1 )
    {
        const child = children[ index ];

        if ( container.children[ index ] !== child )
            container.insertBefore( child, container.children[ index ] || null );
    }

    while ( container.children.length > children.length )
        container.lastElementChild.remove();
}

// ---- User turns and quiet lines --------------------------------------------------------------

function buildSimpleEntry( entry )
{
    if ( entry.kind === 'user' )
        return buildUserTurn( entry );

    if ( entry.kind === 'error' )
        return element( 'p', 'notice error', entry.text );

    return element( 'p', 'notice', entry.text );
}

function buildUserTurn( entry )
{
    const turn = element( 'article', 'turn user' );

    if ( entry.attachments.length > 0 )
    {
        const strip = element( 'div', 'user-attachments' );

        for ( const attachment of entry.attachments )
        {
            const chip = buildAttachmentReference( attachment );
            strip.append( chip );
        }

        turn.append( strip );
    }

    const bubble = element( 'div', 'user-bubble', entry.text );
    turn.append( bubble );

    return turn;
}

function buildAttachmentReference( attachment )
{
    const chip = element( 'span', 'attachment-reference' );
    const icon = element( 'span', 'attachment-icon' );
    const name = element( 'span', 'attachment-name', attachment.name );

    setIcon( icon, attachment.kind === 'audio' ? 'audio' : 'image' );
    chip.append( icon, name );

    return chip;
}

// ---- Assistant turns -------------------------------------------------------------------------

function createAssistantRecord( entry )
{
    const turn = element( 'article', 'turn assistant' );
    const body = element( 'div', 'turn-body' );
    const notice = element( 'p', 'finish-notice' );
    const footer = element( 'div', 'turn-footer' );
    const copy = iconButton( 'quiet-icon copy-reply', 'copy', 'Copy reply', 'copy-reply' );
    const stats = element( 'span', 'stats' );

    copy.dataset.identifier = entry.id;
    footer.append( copy, stats );
    turn.append( body, notice, footer );

    return { entry: null, element: turn, body, notice, footer, stats, blocks: new Map() };
}

function updateAssistant( record, entry, surface )
{
    const lastIndex = entry.blocks.length - 1;
    const children = [];

    for ( let index = 0; index < entry.blocks.length; index = index + 1 )
    {
        const block = entry.blocks[ index ];
        const live = entry.streaming && index === lastIndex;
        const expanded = surface.expanded.has( block.id );
        const blockRecord = renderBlock( record, block, live, expanded );
        children.push( blockRecord.element );
    }

    if ( entry.streaming && entry.blocks.length === 0 )
    {
        const caret = record.pendingCaret !== undefined ? record.pendingCaret : element( 'span', 'caret' );
        record.pendingCaret = caret;
        children.push( caret );
    }

    placeChildren( record.body, children );

    if ( record.entry === entry )
        return;

    record.entry = entry;
    record.element.classList.toggle( 'streaming', entry.streaming );

    const notice = entry.streaming ? null : finishNotice( entry.finish );
    record.notice.textContent = notice === null ? '' : notice;
    record.notice.hidden = notice === null;
    record.notice.classList.toggle( 'cancelled', entry.finish === 'cancelled' );

    record.footer.hidden = entry.streaming;

    const summary = summarizeTurn( entry );
    record.stats.textContent = statsLine( summary );
}

function renderBlock( record, block, live, expanded )
{
    const existing = record.blocks.get( block.id );

    if ( existing !== undefined && existing.block === block && existing.live === live && existing.expanded === expanded )
        return existing;

    const built = existing !== undefined ? existing : createBlockRecord( block );
    built.block = block;
    built.live = live;
    built.expanded = expanded;
    updateBlock( built, block, live, expanded );
    record.blocks.set( block.id, built );

    return built;
}

function createBlockRecord( block )
{
    if ( block.kind === 'thought' )
        return createThoughtRecord( block );

    if ( block.kind === 'tool' )
        return createToolRecord( block );

    const answer = element( 'div', 'answer markdown' );

    return { element: answer };
}

function updateBlock( built, block, live, expanded )
{
    if ( block.kind === 'thought' )
    {
        updateThought( built, block, live, expanded );

        return;
    }

    if ( block.kind === 'tool' )
    {
        updateTool( built, block, expanded );

        return;
    }

    const options = { caret: live };
    built.element.innerHTML = renderMarkdown( block.text, options );
}

// ---- Thought blocks --------------------------------------------------------------------------

function createThoughtRecord( block )
{
    const container = element( 'div', 'thought' );
    const toggle = button( 'thought-toggle', null, 'toggle' );
    const chevron = element( 'span', 'chevron' );
    const label = element( 'span', 'thought-label' );
    const caret = element( 'span', 'caret' );
    const body = element( 'div', 'thought-body markdown' );

    toggle.dataset.identifier = block.id;
    setIcon( chevron, 'chevronRight' );
    toggle.append( chevron, label, caret );
    container.append( toggle, body );

    return { element: container, toggle, label, caret, body };
}

function updateThought( built, block, live, expanded )
{
    built.element.classList.toggle( 'open', expanded );
    built.element.classList.toggle( 'live', live );
    built.toggle.setAttribute( 'aria-expanded', expanded ? 'true' : 'false' );
    built.label.textContent = thoughtLabel( block, live );
    built.caret.hidden = !live || expanded;
    built.body.hidden = !expanded;

    // A closed block is not kept current while it streams: opening it renders it whole.
    if ( !expanded )
        return;

    const options = { caret: live };
    built.body.innerHTML = renderMarkdown( block.text, options );
}

// ---- Tool cards ------------------------------------------------------------------------------

function createToolRecord( block )
{
    const card = element( 'div', 'tool-card' );
    const toggle = button( 'tool-toggle', null, 'toggle' );
    const icon = element( 'span', 'tool-icon' );
    const summary = element( 'span', 'tool-summary', block.summary );
    const name = element( 'span', 'tool-name', block.name );
    const status = element( 'span', 'tool-status' );
    const chevron = element( 'span', 'chevron' );
    const detail = element( 'div', 'tool-detail' );
    const argumentsLabel = element( 'div', 'tool-section-label', 'Arguments' );
    const argumentsBody = element( 'pre', 'tool-text' );
    const resultLabel = element( 'div', 'tool-section-label', 'Result' );
    const resultBody = element( 'pre', 'tool-text' );

    toggle.dataset.identifier = block.id;
    setIcon( icon, 'tool' );
    setIcon( chevron, 'chevronRight' );
    toggle.append( icon, summary, name, status, chevron );
    detail.append( argumentsLabel, argumentsBody, resultLabel, resultBody );
    card.append( toggle, detail );

    return { element: card, toggle, status, detail, argumentsBody, resultBody };
}

function updateTool( built, block, expanded )
{
    const finished = block.result !== null;

    built.element.classList.toggle( 'open', expanded );
    built.element.classList.toggle( 'failed', block.isError );
    built.toggle.setAttribute( 'aria-expanded', expanded ? 'true' : 'false' );
    built.status.textContent = finished ? ( block.isError ? 'Failed' : '' ) : 'Running';
    built.detail.hidden = !expanded;
    built.argumentsBody.textContent = readableJson( block.arguments );
    built.resultBody.textContent = finished ? readableJson( block.result ) : 'Waiting for the result.';
}

// ---- Copying ---------------------------------------------------------------------------------

function replyText( entry )
{
    if ( entry.answer !== null )
        return entry.answer;

    const answers = entry.blocks.filter( ( block ) => block.kind === 'answer' );
    const texts = answers.map( ( block ) => block.text );

    return texts.join( '\n\n' );
}

function copyCode( target )
{
    const block = target.closest( '.code-block' );
    const code = block === null ? null : block.querySelector( 'code' );

    if ( code === null )
        return;

    copyText( code.textContent, target );
}

function copyText( text, target )
{
    const confirm = () => showCopied( target );
    const ignore = () => null;
    const written = navigator.clipboard.writeText( text );

    written.then( confirm, ignore );
}

function showCopied( target )
{
    const restore = () => markCopied( target, false );

    markCopied( target, true );
    window.setTimeout( restore, 1200 );
}

function markCopied( target, copied )
{
    target.classList.toggle( 'copied', copied );

    if ( target.dataset.action === 'copy-code' )
    {
        target.textContent = copied ? 'Copied' : 'Copy';

        return;
    }

    setIcon( target, copied ? 'check' : 'copy' );
}
