// The first-run setup (AssistantOnboarding.md 4.1): a sheet of one size in an empty window, the workspace starting when
// it ends. One pane at a time, each with its way forward at the bottom right and the progress of what it is doing on
// the left.

import { element, button, clear } from '../dom.js';
import { isBusy, firstRunStep, suggestedModel, setupPane, installUnderWay, onboardingRunning, updateShowing } from '../state.js';
import { modelAbilities, fitLine, modelTitle, namedTitle, licenseParagraphs, downloadLine, installAmount, formatBytes } from '../format.js';
import { downloadProgress } from './download-progress.js';

export function createSetupView( layer, sheet, actions )
{
    let shownPane = null;
    let shownSignature = null;
    let askedOnShowing = false;

    // Kept across renders, so the place the person has read to stays while the pane is drawn again.
    const licenseBox = element( 'div', 'license-text' );
    licenseBox.tabIndex = 0;
    licenseBox.setAttribute( 'role', 'document' );
    licenseBox.addEventListener( 'scroll', onLicenseScroll );

    function onClick( event )
    {
        const target = event.target.closest( '[data-action]' );

        if ( target === null || !sheet.contains( target ) )
            return;

        const action = target.dataset.action;

        if ( action === 'setup-continue' )
            actions.dispatch( { type: 'setup.continue' } );

        if ( action === 'setup-done' )
            actions.dispatch( { type: 'setup.done' } );

        if ( action === 'accept-license' )
        {
            const accept = { type: 'license.accept', sha256: actions.getState().license.sha256 };
            actions.send( accept );
        }

        if ( action === 'retry-models' )
            lookForModels();

        // The person agrees to the download, and goes on while it runs (AssistantOnboarding.md 4.3, round 9).
        if ( action === 'agree-download' )
            actions.send( { type: 'onboarding.download' } );

        if ( action === 'panes-done' )
            actions.send( { type: 'onboarding.panes_done' } );

        if ( action === 'install-model' )
        {
            const message = { type: 'model.install', name: target.dataset.name };
            actions.send( message );
        }

        if ( action === 'stop-install' )
            actions.send( { type: 'model.install_stop' } );
    }

    sheet.addEventListener( 'click', onClick );

    // Each pane gives the keyboard to its way forward; the focus ring shows only once the person moves with Tab, as the
    // engine draws it for given focus before the mouse is first used and not after (Todd, 2026-11-25).
    document.addEventListener( 'keydown', ( event ) =>
    {
        if ( event.key === 'Tab' )
            sheet.dataset.keyboard = 'true';
    }, true );

    document.addEventListener( 'pointerdown', () => { delete sheet.dataset.keyboard; }, true );

    function lookForModels()
    {
        const refresh = { type: 'models.refresh', online: true };
        actions.send( refresh );
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

    function render( state )
    {
        const pane = setupPane( state );
        const showing = pane !== null;

        layer.hidden = !showing;

        if ( !showing )
        {
            if ( shownPane !== null )
            {
                clear( sheet );
                delete sheet.dataset.pane;
                shownPane = null;
            }

            return;
        }

        // Progress arrives many times a second while Mila gets ready. Drawing the pane again for each would replace its
        // buttons between a press and its release, so a click never lands: the same pane showing the same thing has
        // only its indicator and bars moved.
        const signature = paneSignature( state, pane );

        if ( pane === shownPane && signature !== null && signature === shownSignature )
        {
            moveProgress( state );

            return;
        }

        shownSignature = signature;

        // Where the person has read the licence to, and what held the keyboard, kept while the pane is drawn again.
        const readTo = licenseBox.scrollTop;
        const held = heldControl();

        clear( sheet );
        sheet.dataset.pane = pane;

        const body = element( 'div', 'setup-body' );
        const footer = element( 'div', 'setup-footer' );
        const progressSlot = element( 'div', 'setup-progress' );
        footer.append( progressSlot );

        // Once the download is under way, the onboarding's window says how far it has come on every pane (4.3, round 9).
        const indicator = readyIndicator( state );

        if ( indicator !== null )
            sheet.append( indicator );

        if ( pane === 'welcome' )
            renderWelcomePane( state, body, footer );
        else if ( pane === 'license' )
            renderLicensePane( state, body, footer );
        else if ( pane === 'ready' )
            renderReadyPane( state, body, footer );
        else if ( pane === 'capabilities' )
            renderCapabilitiesPane( state, body, footer );
        else if ( pane === 'getting-ready' )
            renderGettingReadyPane( state, body, footer );
        else if ( pane === 'model' )
            renderModelPane( state, body, footer, progressSlot );
        else
            renderConclusionPane( state, body, footer );

        sheet.append( body, footer );

        if ( pane === 'license' )
            licenseBox.scrollTop = readTo;

        // A new pane takes the keyboard at its way forward, as a default button does; the same pane drawn again gives
        // it back to what held it.
        if ( pane !== shownPane )
        {
            shownPane = pane;
            focusForward( footer );
        }
        else if ( held !== null )
        {
            restoreControl( held, footer );
        }

        if ( pane !== 'license' )
            return;

        // A licence that fits its box is read once it is shown.
        if ( !state.licenseRead && state.license.text !== null )
            requestAnimationFrame( onLicenseScroll );
    }

    /**
     * What the onboarding's panes show apart from how far getting ready has come; null for the panes drawn whole on every
     * change, whose model card carries its own progress.
     */
    function paneSignature( state, pane )
    {
        if ( pane === 'model' || pane === 'conclusion' )
            return null;

        const progress = readyProgress( state );
        const entry = onboardingEntry( state );
        const license = state.license;

        return JSON.stringify( [
            license === null ? null : [ license.accepted, license.sha256, license.problem ],
            state.licenseRead, state.licenseError, state.modelsError, firstRunStep( state ), isBusy( state ),
            entry === null ? null : [ entry.name, entry.size_bytes, entry.agentic_here ],
            state.session === null ? null : [ state.session.onboarding, state.session.onboarding_agreed, state.session.whats_new ],
            progress === null ? null : [ progress.phase, progress.fraction === null, progress.download === null || progress.download.total === null ] ] );
    }

    /** How far getting ready has come, moved in place: the window's indicator. */
    function moveProgress( state )
    {
        const old = sheet.querySelector( '.setup-indicator' );
        const indicator = readyIndicator( state );

        if ( old !== null )
            old.remove();

        if ( indicator !== null )
            sheet.prepend( indicator );
    }

    /** The control in the sheet that holds the keyboard, by what it does, since drawing the pane replaces it. */
    function heldControl()
    {
        const active = document.activeElement;

        if ( active === null || !sheet.contains( active ) )
            return null;

        if ( active === licenseBox )
            return 'license-text';

        return active.dataset.action ?? null;
    }

    function restoreControl( held, footer )
    {
        if ( held === 'license-text' )
        {
            licenseBox.focus( { preventScroll: true } );

            return;
        }

        const control = sheet.querySelector( '[data-action="' + held + '"]' );

        // A control dimmed since cannot hold the keyboard: the way forward takes it, as when the pane opened.
        if ( control !== null && !control.disabled )
            control.focus();
        else
            focusForward( footer );
    }

    function focusForward( footer )
    {
        const forward = footer.querySelector( '.setup-button.primary:not(:disabled)' );

        if ( forward !== null )
            forward.focus();
        else
            sheet.focus();
    }

    function setupButton( label, action, enabled, primary )
    {
        const made = button( primary ? 'setup-button primary' : 'setup-button', label, action );
        made.disabled = !enabled;

        return made;
    }

    // The first pane has nothing before it, so no Go Back. On the onboarding's path it says what Mila Assistant is and
    // what the steps after it do; the words are a draft until the process is settled (4.3, round 9).
    function renderWelcomePane( state, body, footer )
    {
        // The lockup, the mark and the wordmark as one object as on the website, then Welcome beneath it as the
        // pane's title (Todd, 2026-11-25).
        const lockup = element( 'div', 'setup-lockup' );
        const mark = element( 'img', '' );
        mark.src = 'mila-mark.svg';
        mark.alt = '';
        lockup.append( mark, element( 'span', '', 'Mila' ) );
        const heading = element( 'h1', 'setup-heading setup-welcome-heading', 'Welcome' );

        body.classList.add( 'setup-welcome' );
        body.append( lockup, heading );

        if ( onboardingRunning( state ) )
        {
            const what = 'Mila Assistant is a personal assistant that runs on this computer.';
            const how = 'To set it up, accept the Terms and Conditions and let Mila get ready; then Mila asks you a few '
                + 'questions to set itself up for you.';
            body.append( element( 'p', 'setup-line', what ), element( 'p', 'setup-line', how ) );
        }
        else if ( updateShowing( state ) )
        {
            body.append( element( 'p', 'setup-line', 'Mila Assistant has been updated.' ) );
        }
        else
        {
            body.append( element( 'p', 'setup-byline', 'Local LLMs on hardware you own — at the metal' ) );
        }

        footer.append( setupButton( 'Continue', 'setup-continue', true, true ) );
    }

    /** The published entry of the model the setup installs, for its size and what it can do here; null until the list is read. */
    function onboardingEntry( state )
    {
        if ( state.publishedModels === null || state.onboardingModel === null )
            return null;

        return state.publishedModels.find( ( model ) => model.name === state.onboardingModel ) ?? null;
    }

    /**
     * How far getting Mila ready has come: the download, then the start of the model. Null when nothing is under way.
     */
    function readyProgress( state )
    {
        const download = installUnderWay( state );

        if ( download !== null )
            return { phase: 'download', fraction: download.total === null ? null : download.done / download.total, download };

        const progress = state.progress;
        const loading = ( progress !== null && progress.task === 'load' && ( progress.total === null || progress.done < progress.total ) )
            || ( state.session !== null && state.session.loading );

        if ( loading )
            return { phase: 'load', fraction: null, download: null };

        return null;
    }

    /** The onboarding window's indicator, a ring at its corner as a browser shows a download; null when idle. */
    function readyIndicator( state )
    {
        const progress = readyProgress( state );

        if ( progress === null )
            return null;

        const ring = element( 'div', 'setup-indicator' );
        const said = progress.phase === 'download'
            ? 'Getting Mila ready' + ( progress.download.total === null ? '' : ': ' + installAmount( progress.download ) )
            : 'Starting Mila';
        ring.title = said;
        ring.setAttribute( 'role', 'progressbar' );
        ring.setAttribute( 'aria-label', said );

        const circumference = 2 * Math.PI * 9;
        const shown = progress.fraction === null ? 0.25 : Math.max( progress.fraction, 0.02 );
        ring.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true">'
            + '<circle class="setup-indicator-track" cx="12" cy="12" r="9"></circle>'
            + '<circle class="setup-indicator-done" cx="12" cy="12" r="9" stroke-dasharray="' + ( circumference * shown ).toFixed( 2 )
            + ' ' + circumference.toFixed( 2 ) + '"></circle>'
            + '<path class="setup-indicator-arrow" d="M12 8v7m-3-3l3 3 3-3"></path></svg>';

        if ( progress.fraction === null )
            ring.classList.add( 'moving' );

        return ring;
    }

    /**
     * Getting Mila Ready, after the terms: the model Mila chose for this computer and the space it takes, and the
     * person's agreement, which starts the download (AssistantOnboarding.md 4.3, round 9). The model is spoken of and
     * not named (4.4, round 13). Declining is closing the window.
     */
    function renderReadyPane( state, body, footer )
    {
        body.append( element( 'h1', 'setup-heading', 'Getting Mila Ready' ) );

        const entry = onboardingEntry( state );

        // Declining is closing the window, which quits Mila (Todd, 2026-11-25).
        footer.append( setupButton( 'Download', 'agree-download', entry !== null, true ) );

        if ( firstRunStep( state ) === 'unreachable' )
        {
            const problem = element( 'p', 'empty-detail start-refusal', 'Mila could not reach the internet to see what it needs. '
                + 'Check the internet connection, then try again.' );
            problem.setAttribute( 'role', 'alert' );
            body.append( problem, button( 'chip-button start-choose', 'Try again', 'retry-models' ) );

            return;
        }

        if ( entry === null )
        {
            const finding = element( 'div', 'reading start-finding' );
            const caret = element( 'span', 'caret' );
            caret.setAttribute( 'aria-hidden', 'true' );
            finding.append( caret, element( 'span', 'reading-label', 'Finding the model that suits this computer' ) );
            body.append( finding );

            return;
        }

        body.append( element( 'p', 'setup-line', 'Mila has chosen a model that suits this computer. It takes about '
            + formatBytes( entry.size_bytes ) + ' of space, and downloads in the background while you go on.' ) );
    }

    /**
     * What Mila can do on this computer, from what is measured of the model here, never from the computer's name; or,
     * on an update, what is new (AssistantOnboarding.md 4.3, round 9). Read while the download runs.
     */
    function renderCapabilitiesPane( state, body, footer )
    {
        footer.append( setupButton( 'Continue', 'panes-done', true, true ) );

        if ( !onboardingRunning( state ) && updateShowing( state ) )
        {
            body.append( element( 'h1', 'setup-heading', 'What’s New' ) );

            for ( const paragraph of state.session.whats_new.split( /\n\s*\n/ ) )
                body.append( element( 'p', 'setup-line', paragraph.trim() ) );

            return;
        }

        body.append( element( 'h1', 'setup-heading', 'What Mila Can Do' ) );

        const list = element( 'ul', 'setup-list' );

        for ( const line of capabilityLines( onboardingEntry( state ) ) )
            list.append( element( 'li', 'setup-line', line ) );

        body.append( element( 'p', 'setup-line', 'On this computer, Mila can:' ), list );
    }

    /** What the setup's model can do here, in the person's words; the conversation alone when nothing more is measured. */
    function capabilityLines( entry )
    {
        if ( entry === null )
            return [ 'Answer your questions and hold a conversation' ];

        const lines = [];

        if ( entry.agentic_here )
        {
            lines.push( 'Work through a task for you, step by step' );
            lines.push( 'Read and change files in the folders you choose, asking before it changes anything' );
            lines.push( 'Find what your earlier chats said, and keep related chats together in projects' );
        }
        else
        {
            lines.push( 'Answer your questions and hold a conversation' );
        }

        if ( entry.thinking )
            lines.push( 'Think a question through before it answers' );

        if ( entry.modalities.includes( 'image' ) )
            lines.push( 'Look at the images you give it' );

        if ( entry.modalities.includes( 'audio' ) )
            lines.push( 'Listen to the audio you give it' );

        return lines;
    }

    /**
     * The person reached the end of the panes before Mila was ready: the download, then the start, as a list with
     * checks (AssistantOnboarding.md 4.3, round 9). Ends when the interview's chat opens; closing the window keeps
     * what has arrived.
     */
    function renderGettingReadyPane( state, body, footer )
    {
        body.append( element( 'h1', 'setup-heading', 'Almost Ready' ) );

        if ( firstRunStep( state ) === 'unreachable' )
        {
            const problem = element( 'p', 'empty-detail start-refusal', 'Mila could not reach the internet to get ready. '
                + 'Check the internet connection, then try again.' );
            problem.setAttribute( 'role', 'alert' );
            body.append( problem, button( 'chip-button start-choose', 'Try again', 'retry-models' ) );

            return;
        }

        const progress = readyProgress( state );

        // A download that stopped, or a start refused, says why; Try again continues the download from what arrived, or
        // starts Mila again.
        const failed = state.modelsError !== null && ( state.modelsError.during === 'install' || state.modelsError.during === 'load' );

        if ( progress === null && failed && !isBusy( state ) )
        {
            const problem = element( 'p', 'empty-detail start-refusal', state.modelsError.text );
            problem.setAttribute( 'role', 'alert' );
            body.append( problem, button( 'chip-button start-choose', 'Try again', 'retry-models' ) );

            return;
        }

        body.append( element( 'p', 'setup-line', 'Mila is almost ready. Your first chat opens as soon as it is.' ) );

        // A simple list with checks (Todd, 2026-11-25); the window's indicator says how far the download has come.
        const downloading = progress !== null && progress.phase === 'download';
        const starting = progress !== null && progress.phase === 'load';
        const list = element( 'ul', 'setup-checklist' );
        list.append( checkItem( 'Download the model', downloading ? 'now' : 'done' ) );
        list.append( checkItem( 'Start Mila', starting ? 'now' : downloading ? 'waiting' : 'done' ) );
        list.append( checkItem( 'Open your first chat', 'waiting' ) );
        body.append( list );
    }

    /** One step of getting ready, checked once done. */
    function checkItem( label, status )
    {
        const item = element( 'li', 'setup-check', label );
        item.dataset.status = status;

        return item;
    }

    function renderLicensePane( state, body, footer )
    {
        const license = state.license;
        const heading = element( 'h1', 'setup-heading', !onboardingRunning( state ) ? 'License' : 'Terms and Conditions' );

        body.append( heading );

        if ( license.text === null )
        {
            const problem = element( 'p', 'empty-detail start-refusal', license.problem );
            problem.setAttribute( 'role', 'alert' );
            body.append( problem );

            return;
        }

        const said = !onboardingRunning( state ) ? 'Mila is yours to use under this license. Read it to the end to accept it.'
            : 'Read the Terms and Conditions to the end to accept them.';
        const line = element( 'p', 'setup-line', said );

        const paragraphs = licenseParagraphs( license.text );

        if ( licenseBox.textContent !== paragraphs )
            licenseBox.textContent = paragraphs;

        body.classList.add( 'setup-license' );
        body.append( line, licenseBox );

        if ( state.licenseError !== null )
        {
            const refusal = element( 'p', 'empty-detail start-refusal', state.licenseError );
            refusal.setAttribute( 'role', 'alert' );
            body.append( refusal );
        }

        // Accepted, the License is done: the setup moves past it and never returns (AssistantOnboarding.md 4.1).
        const forward = setupButton( 'Accept', 'accept-license', state.licenseRead, true );

        // No Go Back, as the welcome before it holds nothing to change, and no Quit: declining is closing the window,
        // which quits Mila (Todd, 2026-11-25).
        footer.append( forward );
    }

    /**
     * Nothing is installed: Mila's suggestion for this computer, or why there is none (AssistantOnboarding.md 4). The
     * publisher is asked as the pane first shows (4.2); Install is the way forward, and Skip goes to the workspace with
     * no model, where Settings' Models offers every one. The License before it is accepted and done, so there is no Go
     * Back.
     */
    function renderModelPane( state, body, footer, progressSlot )
    {
        const step = firstRunStep( state );

        body.append( element( 'h1', 'setup-heading', 'Install Model' ) );

        // Once, after the pane is drawn, since asking changes the state it was drawn from.
        if ( step === 'not-asked' && !askedOnShowing )
        {
            askedOnShowing = true;
            queueMicrotask( lookForModels );
        }

        if ( state.modelsError !== null && state.modelsError.during === 'install' )
        {
            const refusal = element( 'p', 'empty-detail start-refusal', state.modelsError.text );
            refusal.setAttribute( 'role', 'alert' );
            body.append( refusal );
        }

        // The download is the card's to show, from the press, with its Cancel; the load it ends in is the footer's.
        const download = installUnderWay( state );
        const working = setupLoad( state );

        if ( working !== null )
            progressSlot.append( working );

        // The card is the model downloading, which may be one chosen in Settings, or else the suggestion.
        const downloadingModel = download === null || state.publishedModels === null ? undefined
            : state.publishedModels.find( ( listed ) => listed.name === download.name );
        const model = step !== 'suggested' ? null : downloadingModel ?? suggestedModel( state );
        const idle = download === null && working === null && !isBusy( state );
        const install = setupButton( 'Install', 'install-model', model !== null && idle, true );

        if ( model !== null )
            install.dataset.name = model.name;

        footer.append( setupButton( 'Skip', 'setup-done', idle, false ), install );

        if ( step === 'not-asked' || step === 'asking' )
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

        if ( model === null )
            return;

        const downloading = download !== null && download.name === model.name ? download : null;
        renderSuggestion( model, body, downloading );
    }

    /** The load an install ends in, while it runs: its label and a bar when its size is known. */
    function setupLoad( state )
    {
        const progress = state.progress;
        const running = progress !== null && progress.task === 'load' && ( progress.total === null || progress.done < progress.total );

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

    /** The last pane: nothing after it to undo from, so no Go Back. */
    function renderConclusionPane( state, body, footer )
    {
        const title = namedTitle( state, state.session.model );
        const heading = element( 'h1', 'setup-heading', 'Ready' );
        const line = element( 'p', 'setup-line', title + ' is installed and loaded.' );
        const change = element( 'p', 'setup-line', 'To change the model, open Settings, then Models.' );

        body.append( heading, line, change );
        footer.append( setupButton( 'Done', 'setup-done', true, true ) );
    }

    /**
     * The model the planner suggests, described by what it can do here and what it downloads; Install is the pane's
     * own button. While it downloads, the card shows how far, and Cancel.
     */
    function renderSuggestion( model, target, downloading )
    {
        const introduction = element( 'p', 'setup-line', 'To start, install the model Mila suggests for this computer.' );
        const card = element( 'div', 'start-suggestion' );
        const name = element( 'div', 'model-suggestion-name', modelTitle( model ) );
        const abilities = element( 'div', 'suggestion-line', modelAbilities( model ) );
        const fit = fitLine( model );
        const size = downloadLine( model );

        target.append( introduction );
        card.append( name, abilities );

        if ( fit !== null )
            card.append( element( 'div', 'suggestion-facts', fit ) );

        if ( size !== null )
            card.append( element( 'div', 'suggestion-facts', size ) );

        if ( model.license !== '' )
            card.append( element( 'div', 'suggestion-facts', 'License: ' + model.license ) );

        if ( downloading !== null )
            card.append( downloadProgress( downloading ) );

        target.append( card );
    }

    /** The publisher answered and nothing it offers runs here: why, for the model nearest to running. */
    function renderNoneFits( state, target )
    {
        const reason = element( 'p', 'empty-detail start-refusal', 'None of the models published for Mila runs on this computer.' );
        reason.setAttribute( 'role', 'alert' );
        target.append( reason );

        const nearest = state.publishedModels.find( ( model ) => model.why_not !== null );

        if ( nearest !== undefined )
            target.append( element( 'p', 'empty-detail start-why', nearest.name + ': ' + nearest.why_not ) );
    }

    return { render };
}
