// The conversation: user turns, assistant turns built from their blocks in order of occurrence,
// and the session's quiet lines. An entry or a block is rebuilt only when the reducer gave it a
// new object, so a streamed token touches one element.

import { element, button, iconButton, setIcon, clear, previewOrIcon } from '../dom.js';
import { renderMarkdown } from '../markdown.js';
import { summarizeTurn, isBusy, emptyScreen, firstRunStep, suggestedModel } from '../state.js';
import { thoughtLabel, statsLine, finishNotice, readableJson, activityCard, formatBytes, modelAbilities,
    suggestionFacts } from '../format.js';

const stickDistance = 80;

export function createTranscriptView( scroller, container, actions )
{
    const records = new Map();
    const emptyState = element( 'div', 'empty-state' );
    const waiting = element( 'div', 'turn assistant waiting' );
    let lastLength = 0;

    waiting.append( readingLine() );

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

        if ( action === 'tool-decision' )
            actions.decideTool( target.dataset.turn, target.dataset.call, target.dataset.decision );

        if ( action === 'start-activity' )
            actions.startActivity( target.dataset.activity );

        if ( action === 'load-model' )
        {
            const message = { type: 'model.load', name: target.dataset.name, quantization: null };
            actions.send( message );
        }

        // A first model is chosen from the published list, so it is fetched as Settings opens unless it is here.
        if ( action === 'open-models' )
        {
            const current = actions.getState();

            if ( current.publishedModels === null && !current.publishedRequested )
            {
                const refresh = { type: 'models.refresh', online: true };
                actions.send( refresh );
            }

            actions.openSettings( 'models' );
        }

        if ( action === 'retry-models' )
        {
            const refresh = { type: 'models.refresh', online: true };
            actions.send( refresh );
        }

        if ( action === 'install-model' )
        {
            const message = { type: 'model.install', name: target.dataset.name };
            actions.send( message );
        }
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

        const screen = emptyScreen( state );

        if ( screen === null )
            return;

        if ( screen === 'choose-model' )
        {
            renderChooseModel( state );

            return;
        }

        if ( screen === 'first-run' )
        {
            renderFirstRun( state );

            return;
        }

        const activity = state.session.activity;

        if ( screen === 'activity' )
        {
            const card = activityCard( activity );
            const detailText = state.session.folder ? 'Working in ' + state.session.folder : card.line;
            const heading = element( 'p', 'empty-heading', card.name );
            const detail = element( 'p', 'empty-detail', detailText );
            emptyState.append( heading, detail );

            return;
        }

        renderHome( state );
    }

    /** Why no model is loaded, when there is a reason: a load just refused, or the remembered model not opening. */
    function noModelReason( state )
    {
        if ( state.modelsError !== null && state.modelsError.during === 'load' )
            return state.modelsError.text;

        return state.session.no_model_reason;
    }

    /** Models are installed and none is loaded: each one a click from loading. */
    function renderChooseModel( state )
    {
        const heading = element( 'p', 'empty-heading', 'Choose a model to start' );
        const reason = noModelReason( state );
        const list = element( 'div', 'start-models' );

        emptyState.append( heading );

        // Why the person is asked: a refusal when a load was refused, else that nothing is loaded.
        if ( reason !== null )
        {
            const refusal = element( 'p', 'empty-detail start-refusal', reason );
            refusal.setAttribute( 'role', 'alert' );
            emptyState.append( refusal );
        }
        else
        {
            emptyState.append( element( 'p', 'empty-detail', 'Mila needs a model loaded to answer you.' ) );
        }

        for ( const model of state.installedModels )
        {
            const row = element( 'div', 'model-row' );
            const text = element( 'div', 'model-text' );
            const name = element( 'div', 'model-name', model.name );
            const facts = element( 'div', 'model-facts', formatBytes( model.size_bytes ) + ' · ' + model.quantization );
            const load = button( 'chip-button', 'Load', 'load-model' );

            load.dataset.name = model.name;
            text.append( name, facts );
            row.append( text, load );
            list.append( row );
        }

        const more = button( 'link-button', 'Install another model', 'open-models' );
        emptyState.append( list, more );
    }

    /**
     * Nothing is installed: Mila's suggestion for this computer, installed with one click, or the way through
     * when there is none (AssistantOnboarding.md 4). The publisher is asked as the screen first shows.
     */
    function renderFirstRun( state )
    {
        const step = firstRunStep( state );

        if ( step === 'not-asked' )
        {
            const refresh = { type: 'models.refresh', online: true };
            queueMicrotask( () => actions.send( refresh ) );
        }

        const heading = element( 'p', 'empty-heading', 'Welcome to Mila' );
        const detail = element( 'p', 'empty-detail',
            'Mila runs a language model on this computer, so what you ask stays here. To start, install one.' );

        emptyState.append( heading, detail );

        if ( state.modelsError !== null && state.modelsError.during === 'install' )
        {
            const refusal = element( 'p', 'empty-detail start-refusal', state.modelsError.text );
            refusal.setAttribute( 'role', 'alert' );
            emptyState.append( refusal );
        }

        if ( step === 'not-asked' || step === 'asking' )
        {
            const finding = element( 'div', 'reading start-finding' );
            const ring = element( 'span', 'caret' );
            ring.setAttribute( 'aria-hidden', 'true' );
            finding.append( ring, element( 'span', 'reading-label', 'Finding the model that suits this computer' ) );
            emptyState.append( finding );

            return;
        }

        if ( step === 'unreachable' )
        {
            const problem = element( 'p', 'empty-detail start-refusal', 'Mila could not reach the models published for it. '
                + 'Check the internet connection, then try again.' );
            problem.setAttribute( 'role', 'alert' );
            const again = button( 'chip-button start-choose', 'Try again', 'retry-models' );
            emptyState.append( problem, again );

            return;
        }

        if ( step === 'none-fits' )
        {
            renderNoneFits( state );

            return;
        }

        renderSuggestion( state, suggestedModel( state ) );
    }

    function renderSuggestion( state, model )
    {
        const card = element( 'div', 'start-suggestion' );
        const label = element( 'div', 'suggestion-label', 'Suggested for this computer' );
        const name = element( 'div', 'suggestion-name', model.name );
        const abilities = element( 'div', 'suggestion-line', modelAbilities( model ) );
        const facts = element( 'div', 'suggestion-facts', suggestionFacts( model ) );

        card.append( label, name, abilities, facts );

        if ( model.license !== '' )
            card.append( element( 'div', 'suggestion-facts', 'Licence: ' + model.license ) );

        const progress = state.progress;
        const installing = progress !== null && progress.task === 'install'
            && ( progress.total === null || progress.done < progress.total );

        if ( installing )
        {
            const shown = element( 'div', 'progress' );
            const bar = element( 'progress', 'progress-bar' );

            if ( progress.total !== null )
            {
                bar.max = progress.total;
                bar.value = progress.done;
            }

            shown.append( element( 'div', 'progress-label', progress.label ), bar );
            card.append( shown );
        }
        else if ( isBusy( state ) )
        {
            card.append( element( 'div', 'progress-label', 'Starting the download' ) );
        }
        else
        {
            const install = button( 'chip-button start-install', 'Install', 'install-model' );
            install.dataset.name = model.name;
            card.append( install );
        }

        const other = button( 'link-button', 'Choose another model', 'open-models' );
        emptyState.append( card, other );
    }

    /** The publisher answered and nothing it offers runs here: why, for the model nearest to running, and the list. */
    function renderNoneFits( state )
    {
        const reason = element( 'p', 'empty-detail start-refusal', 'None of the models published for Mila runs on this computer.' );
        reason.setAttribute( 'role', 'alert' );
        emptyState.append( reason );

        const nearest = state.publishedModels.find( ( model ) => model.why_not !== null );

        if ( nearest !== undefined )
            emptyState.append( element( 'p', 'empty-detail start-why', nearest.name + ': ' + nearest.why_not ) );

        const all = button( 'link-button', 'See every model', 'open-models' );
        emptyState.append( all );
    }

    /**
     * The home screen: what the person came to do, as the box below and a card for each activity the
     * session can start (MilaDesktop.md 3.4).
     */
    function renderHome( state )
    {
        const heading = element( 'p', 'empty-heading', 'What would you like to do?' );
        const cards = element( 'div', 'activity-cards' );
        const offered = state.session.activities === undefined ? [ 'ask' ] : state.session.activities;

        for ( const activity of offered )
        {
            const card = activityCard( activity );
            const tile = button( 'activity-card', null, 'start-activity' );
            const name = element( 'span', 'activity-name', card.name );
            const line = element( 'span', 'activity-line', card.line );

            tile.dataset.activity = activity;
            tile.append( name, line );
            cards.append( tile );
        }

        emptyState.append( heading, cards );
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
    chip.append( previewOrIcon( attachment, icon ), name );

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

    // Before the first token, and after a call's result, the model is reading what it will answer.
    const last = entry.blocks.length > 0 ? entry.blocks[ entry.blocks.length - 1 ] : null;
    const reading = last === null || ( last.kind === 'tool' && last.result !== null );

    if ( entry.streaming && reading )
    {
        if ( record.readingLine === undefined )
            record.readingLine = readingLine();

        children.push( record.readingLine );
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

    // No ring at the end of streaming text: the text moving is the sign of life.
    built.element.innerHTML = renderMarkdown( block.text );
}

// ---- Reading ---------------------------------------------------------------------------------

/** The ring and "Reading": the model taking in the message, or a call's result, before it writes. */
function readingLine()
{
    const line = element( 'div', 'reading' );
    const ring = element( 'span', 'caret' );

    ring.setAttribute( 'aria-hidden', 'true' );
    line.append( ring, element( 'span', 'reading-label', 'Reading' ) );

    return line;
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
    caret.setAttribute( 'aria-hidden', 'true' );
    toggle.append( caret, chevron, label );
    container.append( toggle, body );

    return { element: container, toggle, chevron, label, caret, body };
}

function updateThought( built, block, live, expanded )
{
    built.element.classList.toggle( 'open', expanded );
    built.element.classList.toggle( 'live', live );
    built.toggle.setAttribute( 'aria-expanded', expanded ? 'true' : 'false' );
    built.label.textContent = thoughtLabel( block, live );

    // While it is written the ring leads, in the chevron's place: ring, "Thinking", the count.
    built.caret.hidden = !live;
    built.chevron.hidden = live;
    built.body.hidden = !expanded;

    // A closed block is not kept current while it streams: opening it renders it whole.
    if ( !expanded )
        return;

    built.body.innerHTML = renderMarkdown( block.text );
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

    const approval = element( 'div', 'tool-approval' );

    toggle.dataset.identifier = block.id;
    setIcon( icon, 'tool' );
    setIcon( chevron, 'chevronRight' );
    toggle.append( icon, summary, name, status, chevron );
    detail.append( argumentsLabel, argumentsBody, resultLabel, resultBody );
    card.append( toggle, detail, approval );

    return { element: card, toggle, status, detail, argumentsBody, resultBody, approval, choices: null };
}

const choiceLabels = {
    allow_once: 'Allow once',
    allow_conversation: 'Allow in this conversation',
    deny: 'Deny'
};

function toolStatus( block )
{
    if ( block.choices !== null )
        return 'Waiting for you';

    if ( block.decision === 'deny' && block.result !== null )
        return 'Declined';

    if ( block.notRun )
        return 'Not run';

    if ( block.result === null )
        return 'Running';

    return block.isError ? 'Failed' : '';
}

function toolResultText( block )
{
    if ( block.choices !== null )
        return 'Runs once you allow it.';

    if ( block.notRun )
        return 'Not run.';

    return block.result === null ? 'Waiting for the result.' : readableJson( block.result );
}

/**
 * A call waiting for the person is shown open, whatever its toggle says: its arguments are what is
 * being approved.
 */
function updateTool( built, block, expanded )
{
    const waiting = block.choices !== null;
    const declined = block.decision === 'deny';
    const open = expanded || waiting;

    built.element.classList.toggle( 'open', open );
    built.element.classList.toggle( 'waiting', waiting );
    built.element.classList.toggle( 'failed', block.isError && !declined );
    built.toggle.setAttribute( 'aria-expanded', open ? 'true' : 'false' );
    built.status.textContent = toolStatus( block );
    built.detail.hidden = !open;
    built.argumentsBody.textContent = readableJson( block.arguments );
    built.resultBody.textContent = toolResultText( block );
    updateApproval( built, block );
}

function updateApproval( built, block )
{
    built.approval.hidden = block.choices === null;

    if ( built.choices === block.choices )
        return;

    built.choices = block.choices;
    clear( built.approval );

    if ( block.choices === null )
        return;

    const reaches = reachesList( block.reaches );

    if ( reaches !== null )
        built.approval.append( reaches );

    const question = element( 'span', 'tool-question', 'Run this tool?' );
    built.approval.append( question );

    for ( const choice of block.choices )
    {
        const style = choice === 'deny' ? 'chip-button quiet' : 'chip-button';
        const choiceButton = button( style, choiceLabels[ choice ], 'tool-decision' );
        choiceButton.dataset.turn = block.turnId;
        choiceButton.dataset.call = block.callId;
        choiceButton.dataset.decision = choice;
        built.approval.append( choiceButton );
    }
}

/**
 * What the call would reach if allowed, as the session says: its folders, the folders of the
 * programs it may run, and whether the internet. Nothing for a call that reaches none of them.
 */
function reachesList( reaches )
{
    if ( !reaches )
        return null;

    const rows = [];

    for ( const folder of reaches.folders )
        rows.push( [ folder.access === 'read_write' ? 'Reads and writes' : 'Reads', folder.path ] );

    // Many folders, so they are counted, and listed on opening, beside the command they serve.
    if ( reaches.program_folders.length > 0 )
    {
        const count = reaches.program_folders.length;
        const folders = element( 'details', 'tool-program-folders' );
        folders.append( element( 'summary', null, count === 1 ? '1 folder' : count + ' folders' ),
            element( 'div', null, reaches.program_folders.join( '\n' ) ) );
        rows.push( [ 'Runs programs from', folders ] );
    }

    if ( reaches.runs_programs || reaches.network )
        rows.push( [ 'Internet', reaches.network ? 'Yes' : 'No' ] );

    if ( rows.length === 0 )
        return null;

    const list = element( 'dl', 'tool-reaches' );

    for ( const [ label, value ] of rows )
    {
        const definition = element( 'dd', null, typeof value === 'string' ? value : null );

        if ( typeof value !== 'string' )
            definition.append( value );

        list.append( element( 'dt', null, label ), definition );
    }

    return list;
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
