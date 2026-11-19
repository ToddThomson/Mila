// The conversation: user turns, assistant turns built from their blocks in order of occurrence,
// and the session's quiet lines. An entry or a block is rebuilt only when the reducer gave it a
// new object, so a streamed token touches one element.

import { element, button, iconButton, setIcon, clear, previewOrIcon } from '../dom.js';
import { renderMarkdown } from '../markdown.js';
import { summarizeTurn, isBusy, emptyScreen, firstRunStep, suggestedModel, setupPane } from '../state.js';
import { thoughtLabel, statsLine, finishNotice, readableJson, formatBytes, modelAbilities, fitLine, modelTitle, namedTitle,
    licenseParagraphs, approvalWords } from '../format.js';

const stickDistance = 80;

export function createTranscriptView( scroller, container, actions )
{
    const records = new Map();
    const emptyState = element( 'div', 'empty-state' );
    const waiting = element( 'div', 'turn assistant waiting' );
    let lastLength = 0;
    let showingEmpty = false;

    // Kept across renders, so the place the person has read to stays while the pane is drawn again.
    const licenseBox = element( 'div', 'license-text' );
    licenseBox.tabIndex = 0;
    licenseBox.setAttribute( 'role', 'document' );
    licenseBox.addEventListener( 'scroll', onLicenseScroll );

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

        if ( action === 'choose-folder' )
            actions.chooseFolder( target.dataset.turn, target.dataset.call );

        if ( action === 'load-model' )
        {
            const message = { type: 'model.load', name: target.dataset.name, quantization: null };
            actions.send( message );
        }

        // The publisher is asked only when the person says to, there or here (AssistantOnboarding.md 4.2).
        if ( action === 'open-models' )
            actions.openSettings( 'models' );

        if ( action === 'look-for-models' || action === 'retry-models' )
        {
            const refresh = { type: 'models.refresh', online: true };
            actions.send( refresh );
        }

        if ( action === 'install-model' )
        {
            const message = { type: 'model.install', name: target.dataset.name };
            actions.send( message );
        }

        if ( action === 'setup-continue' )
            actions.dispatch( { type: 'setup.continue' } );

        if ( action === 'setup-back' )
            actions.dispatch( { type: 'setup.back' } );

        if ( action === 'setup-done' )
            actions.dispatch( { type: 'setup.done' } );

        if ( action === 'accept-license' )
        {
            const accept = { type: 'license.accept', sha256: actions.getState().license.sha256 };
            actions.send( accept );
        }

        // Declining the licence is quitting, which the window does.
        if ( action === 'quit' )
            actions.transmit( { type: 'app.quit' } );
    }

    /** The end of the licence has been in view once its last line is: within a pixel, for a fractional scroll. */
    function licenseEndSeen()
    {
        return licenseBox.scrollHeight - licenseBox.scrollTop - licenseBox.clientHeight <= 1;
    }

    function onLicenseScroll()
    {
        if ( !actions.getState().licenseRead && licenseEndSeen() )
            actions.dispatch( { type: 'license.read' } );
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
        // The first-run setup has the window to itself: what was said before it ends waits in the transcript.
        const pane = setupPane( state );

        if ( pane !== null )
        {
            renderSetup( state, pane );
            placeChildren( container, [ emptyState ] );

            return;
        }

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

        // A screen asking for a model reads from the top, and keeps where the person scrolled to while it is redrawn;
        // only a conversation keeps to its latest turn.
        if ( state.transcript.length === 0 )
        {
            if ( !showingEmpty )
                scroller.scrollTop = 0;

            showingEmpty = true;
            lastLength = 0;

            return;
        }

        showingEmpty = false;

        const grew = state.transcript.length > lastLength;
        const sent = grew && last !== null && last.kind === 'user';
        lastLength = state.transcript.length;

        if ( stick || sent )
            scroller.scrollTop = scroller.scrollHeight;
    }

    function renderEmptyState( state )
    {
        clear( emptyState );
        emptyState.classList.remove( 'first-run' );

        const screen = emptyScreen( state );

        if ( screen === null )
            return;

        if ( screen === 'choose-model' )
        {
            renderChooseModel( state );

            return;
        }

        // A first run whose setup is done or refused at its load: the way to a model, as Settings has it.
        if ( screen === 'first-run' )
        {
            renderInstallModel();

            return;
        }

        // A conversation not yet begun shows nothing: the bar below is where it begins (AssistantInterface.md 5.1).
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
            const name = element( 'div', 'model-name', modelTitle( model ) );
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
     * The first-run setup (AssistantOnboarding.md 4.1): one pane at a time, each with Go Back and its way forward
     * at the bottom right, and the progress of what it is doing beside them.
     */
    function renderSetup( state, pane )
    {
        // Where the person has read the licence to, kept while the pane is drawn again.
        const readTo = licenseBox.scrollTop;

        clear( emptyState );
        emptyState.classList.add( 'first-run' );

        const body = element( 'div', 'setup-body' );
        const footer = element( 'div', 'setup-footer' );
        const progressSlot = element( 'div', 'setup-progress' );
        footer.append( progressSlot );

        if ( pane === 'welcome' )
            renderWelcomePane( state, body, footer );
        else if ( pane === 'license' )
            renderLicensePane( state, body, footer );
        else if ( pane === 'model' )
            renderModelPane( state, body, footer, progressSlot );
        else
            renderConclusionPane( state, body, footer );

        emptyState.append( body, footer );

        if ( pane !== 'license' )
            return;

        licenseBox.scrollTop = readTo;

        // A licence that fits its box is read once it is shown.
        if ( !state.licenseRead && state.license.text !== null )
            requestAnimationFrame( onLicenseScroll );
    }

    function setupButton( label, action, enabled, primary )
    {
        const made = button( primary ? 'setup-button primary' : 'setup-button', label, action );
        made.disabled = !enabled;

        return made;
    }

    function renderWelcomePane( state, body, footer )
    {
        const mark = element( 'img', 'first-run-mark' );
        mark.src = 'mila-mark.svg';
        mark.alt = '';
        const heading = element( 'h1', 'empty-heading', 'Welcome to Mila' );
        const byline = element( 'p', 'first-run-byline', 'Local LLMs on hardware you own — at the metal' );
        const line = element( 'p', 'first-run-introduction setup-line', welcomeLine( state ) );

        body.append( mark, heading, byline, line );
        footer.append( setupButton( 'Go Back', 'setup-back', false, false ),
            setupButton( 'Continue', 'setup-continue', true, true ) );
    }

    /** What the panes after the welcome ask, in the order they ask it. */
    function welcomeLine( state )
    {
        if ( state.license.accepted )
            return 'To start, install the model that suits this computer.';

        if ( state.installedModels.length > 0 )
            return 'To start, accept Mila’s licence.';

        return 'Two steps to start: accept Mila’s licence, then install the model that suits this computer.';
    }

    function renderLicensePane( state, body, footer )
    {
        const license = state.license;
        const heading = element( 'h1', 'empty-heading', 'Licence' );

        body.append( heading );

        if ( license.text === null )
        {
            const problem = element( 'p', 'empty-detail start-refusal', license.problem );
            problem.setAttribute( 'role', 'alert' );
            body.append( problem );
            footer.append( setupButton( 'Go Back', 'setup-back', true, false ), setupButton( 'Quit', 'quit', true, false ) );

            return;
        }

        const line = element( 'p', 'first-run-introduction setup-line',
            'Mila is yours to use under this licence. Read it to the end to accept it.' );

        const paragraphs = licenseParagraphs( license.text );

        if ( licenseBox.textContent !== paragraphs )
            licenseBox.textContent = paragraphs;

        body.append( line, licenseBox );

        if ( state.licenseError !== null )
        {
            const refusal = element( 'p', 'empty-detail start-refusal', state.licenseError );
            refusal.setAttribute( 'role', 'alert' );
            body.append( refusal );
        }

        const forward = license.accepted
            ? setupButton( 'Continue', 'setup-continue', true, true )
            : setupButton( 'Accept', 'accept-license', state.licenseRead, true );

        footer.append( setupButton( 'Go Back', 'setup-back', true, false ), setupButton( 'Quit', 'quit', true, false ), forward );
    }

    /**
     * Nothing is installed: Mila's suggestion for this computer, or the way through when there is none
     * (AssistantOnboarding.md 4). The publisher is asked only once the person presses Look for models (4.2); Install is
     * then the way forward.
     */
    function renderModelPane( state, body, footer, progressSlot )
    {
        const step = firstRunStep( state );

        body.append( element( 'h1', 'empty-heading', 'A model for this computer' ) );

        if ( step === 'not-asked' )
        {
            body.append( element( 'p', 'first-run-introduction setup-line', 'To suggest a model, Mila asks Hugging Face, where '
                + 'its models are published, what is available. Hugging Face sees this computer’s internet address and '
                + 'which files Mila reads. Nothing you have written is sent, and Mila works out here what fits this computer.' ) );
            footer.append( setupButton( 'Go Back', 'setup-back', !isBusy( state ), false ),
                setupButton( 'Look for models', 'look-for-models', !isBusy( state ), true ) );

            return;
        }

        if ( state.modelsError !== null && state.modelsError.during === 'install' )
        {
            const refusal = element( 'p', 'empty-detail start-refusal', state.modelsError.text );
            refusal.setAttribute( 'role', 'alert' );
            body.append( refusal );
        }

        const working = setupWork( state );

        if ( working !== null )
            progressSlot.append( working );
        else if ( isBusy( state ) && state.setupInstalling )
            progressSlot.append( element( 'div', 'progress-label', 'Starting the download' ) );

        const model = step === 'suggested' ? suggestedModel( state ) : null;
        const install = setupButton( 'Install', 'install-model', model !== null && working === null && !isBusy( state ), true );

        if ( model !== null )
            install.dataset.name = model.name;

        footer.append( setupButton( 'Go Back', 'setup-back', working === null && !isBusy( state ), false ), install );

        if ( step === 'asking' )
        {
            const finding = element( 'div', 'reading start-finding' );
            const ring = element( 'span', 'caret' );
            ring.setAttribute( 'aria-hidden', 'true' );
            finding.append( ring, element( 'span', 'reading-label', 'Finding the model that suits this computer' ) );
            body.append( finding );

            return;
        }

        if ( step === 'unreachable' )
        {
            const problem = element( 'p', 'empty-detail start-refusal', 'Mila could not reach the models published for it. '
                + 'Check the internet connection, then try again.' );
            problem.setAttribute( 'role', 'alert' );
            const again = button( 'chip-button start-choose', 'Try again', 'retry-models' );
            body.append( problem, again );

            return;
        }

        if ( step === 'none-fits' )
        {
            renderNoneFits( state, body );

            return;
        }

        renderSuggestion( state, model, body, working === null && !isBusy( state ) );
    }

    /** The install, then the load it ends in, while either runs: its label and a bar when its size is known. */
    function setupWork( state )
    {
        const progress = state.progress;
        const running = progress !== null && ( progress.task === 'install' || progress.task === 'load' )
            && ( progress.total === null || progress.done < progress.total );

        if ( !running && !state.session.loading )
            return null;

        const shown = element( 'div', 'progress' );
        const label = running ? progress.label : 'Loading';
        shown.append( element( 'div', 'progress-label', label ) );

        if ( running && progress.total !== null )
        {
            const bar = element( 'progress', 'progress-bar' );
            bar.max = progress.total;
            bar.value = progress.done;
            shown.append( bar );
        }

        return shown;
    }

    function renderConclusionPane( state, body, footer )
    {
        const title = namedTitle( state, state.session.model );
        const heading = element( 'h1', 'empty-heading', 'Mila is ready' );
        const line = element( 'p', 'first-run-introduction setup-line', title + ' is installed and loaded.' );
        const change = element( 'p', 'first-run-introduction setup-line', 'To change the model, open Settings, then Models.' );

        body.append( heading, line, change );
        footer.append( setupButton( 'Go Back', 'setup-back', false, false ), setupButton( 'Done', 'setup-done', true, true ) );
    }

    /** A first run whose setup is passed: the way to a model, without the suggestion. */
    function renderInstallModel()
    {
        const heading = element( 'p', 'empty-heading', 'Install a model to start' );
        const detail = element( 'p', 'empty-detail', 'Mila needs a model to answer you.' );
        const choose = button( 'link-button', 'Choose a model to install', 'open-models' );

        emptyState.append( heading, detail, choose );
    }

    /** The model the planner suggests, described by what it can do here; Install is the pane's own button. */
    function renderSuggestion( state, model, target, offerOther )
    {
        const introduction = element( 'p', 'first-run-introduction',
            'To start, install the model Mila suggests for this computer.' );
        const card = element( 'div', 'start-suggestion' );
        const name = element( 'div', 'suggestion-name', modelTitle( model ) );
        const abilities = element( 'div', 'suggestion-line', modelAbilities( model ) );
        const fit = fitLine( model );

        target.append( introduction );
        card.append( name, abilities );

        if ( fit !== null )
            card.append( element( 'div', 'suggestion-facts', fit ) );

        if ( model.license !== '' )
            card.append( element( 'div', 'suggestion-facts', 'Licence: ' + model.license ) );

        target.append( card );

        // Once Install is pressed the choice is made: the way to another model goes.
        if ( offerOther )
        {
            const other = button( 'link-button start-other', 'Choose another model', 'open-models' );
            target.append( other );
        }
    }

    /** The publisher answered and nothing it offers runs here: why, for the model nearest to running, and the list. */
    function renderNoneFits( state, target )
    {
        const reason = element( 'p', 'empty-detail start-refusal', 'None of the models published for Mila runs on this computer.' );
        reason.setAttribute( 'role', 'alert' );
        target.append( reason );

        const nearest = state.publishedModels.find( ( model ) => model.why_not !== null );

        if ( nearest !== undefined )
            target.append( element( 'p', 'empty-detail start-why', nearest.name + ': ' + nearest.why_not ) );

        const all = button( 'link-button', 'See every model', 'open-models' );
        target.append( all );
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

    if ( block.name === 'request_folder' )
    {
        folderQuestion( built, block );

        return;
    }

    const words = approvalWords( block );

    if ( words.facts.length > 0 )
    {
        const sends = element( 'dl', 'tool-reaches tool-sends' );

        for ( const [ label, value ] of words.facts )
            sends.append( element( 'dt', null, label ), element( 'dd', null, value ) );

        built.approval.append( sends );
    }

    const reaches = reachesList( block.reaches );

    if ( reaches !== null )
        built.approval.append( reaches );

    for ( const note of words.notes )
        built.approval.append( element( 'p', 'tool-note', note ) );

    const question = element( 'span', 'tool-question', words.question );
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
 * Mila asking for a folder (AssistantInterface.md 5.5): the call's line already says why, and the answer is the
 * picker, since the folder chosen is what the answer gives.
 */
function folderQuestion( built, block )
{
    const question = element( 'span', 'tool-question', 'Give this chat a folder?' );
    const choose = button( 'chip-button', 'Choose a folder', 'choose-folder' );
    choose.dataset.turn = block.turnId;
    choose.dataset.call = block.callId;

    const decline = button( 'chip-button quiet', 'Not now', 'tool-decision' );
    decline.dataset.turn = block.turnId;
    decline.dataset.call = block.callId;
    decline.dataset.decision = 'deny';

    built.approval.append( question, choose, decline );
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
