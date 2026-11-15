// The main pane's header: the loaded model as a picker, and the session's facts as quiet text.

import { element, button, iconButton, setIcon, clear } from '../dom.js';
import { contextShare, contextDetails, thinkingFact, activityCard, folderName, modelTitle, namedTitle } from '../format.js';
import { appStatus, isAskingForModel } from '../state.js';

// The ring's arc: a circle of radius 7 drawn from twelve o'clock, filled by the share of the context used.
const ringRadius = 7;
const ringLength = 2 * Math.PI * ringRadius;
const ringMarkup = '<svg viewBox="0 0 20 20" aria-hidden="true">'
    + '<circle class="context-ring-track" cx="10" cy="10" r="7"></circle>'
    + '<circle class="context-ring-used" cx="10" cy="10" r="7" transform="rotate(-90 10 10)"></circle></svg>';

export function createHeaderView( root, actions )
{
    const toggle = iconButton( 'quiet-icon sidebar-toggle', 'sidebar', 'Show conversations', 'toggle-sidebar' );
    const picker = element( 'div', 'model-picker' );
    const pickerButton = button( 'model-picker-button', null, 'toggle-picker' );
    const pickerLabel = element( 'span', 'model-picker-label' );
    const pickerChevron = element( 'span', 'chevron' );
    const menu = element( 'div', 'menu model-menu' );
    const facts = element( 'div', 'session-facts' );
    const conversationTitle = element( 'div', 'conversation-title' );
    const status = element( 'span', 'app-status' );
    const statusLabel = element( 'span', 'app-status-label' );
    const context = element( 'div', 'context-control' );
    const contextButton = button( 'context-ring', null, 'toggle-context' );
    const contextPopup = element( 'div', 'menu context-popup' );
    let open = false;
    let contextOpen = false;
    let lastKey = '';

    setIcon( pickerChevron, 'chevronDown' );
    pickerButton.setAttribute( 'aria-haspopup', 'menu' );
    pickerButton.append( pickerLabel, pickerChevron );
    picker.append( pickerButton, menu );
    status.hidden = true;
    status.setAttribute( 'role', 'status' );
    const ring = element( 'span', 'caret' );
    ring.setAttribute( 'aria-hidden', 'true' );
    status.append( ring, statusLabel );
    contextButton.innerHTML = ringMarkup;
    contextButton.setAttribute( 'aria-haspopup', 'dialog' );
    context.append( contextButton, contextPopup );
    context.hidden = true;
    root.append( toggle, picker, status, conversationTitle, facts, context );
    menu.hidden = true;
    contextPopup.hidden = true;

    function setOpen( value )
    {
        open = value;
        menu.hidden = !open;
        pickerButton.setAttribute( 'aria-expanded', open ? 'true' : 'false' );
    }

    function setContextOpen( value )
    {
        contextOpen = value;
        contextPopup.hidden = !contextOpen;
        contextButton.setAttribute( 'aria-expanded', contextOpen ? 'true' : 'false' );
    }

    /** The ring filled by the share used, and the popup's rows, for the session as it stands. */
    function renderContext( session )
    {
        const share = contextShare( session );
        const used = contextButton.querySelector( '.context-ring-used' );
        const filled = share === null ? 0 : share * ringLength;

        used.setAttribute( 'stroke-dasharray', filled + ' ' + ringLength );
        contextButton.classList.toggle( 'full', share !== null && share >= 0.9 );

        const rows = contextDetails( session );
        const label = 'Context: ' + rows[ 0 ][ 1 ];
        contextButton.setAttribute( 'aria-label', label );
        contextButton.title = label;

        clear( contextPopup );
        contextPopup.append( element( 'div', 'menu-label', 'Context' ) );

        for ( const [ name, value ] of rows )
        {
            const row = element( 'div', 'context-row' );
            row.append( element( 'span', 'context-row-name', name ), element( 'span', 'context-row-value', value ) );
            contextPopup.append( row );
        }
    }

    function loadModel( name )
    {
        const state = actions.getState();

        if ( state.session !== null && state.session.model === name )
            return;

        const message = { type: 'model.load', name, quantization: null };
        actions.send( message );
    }

    function onClick( event )
    {
        const target = event.target.closest( '[data-action]' );

        if ( target === null )
            return;

        const action = target.dataset.action;

        if ( action === 'toggle-sidebar' )
            actions.toggleSidebar();

        if ( action === 'toggle-picker' )
            setOpen( !open );

        if ( action === 'toggle-context' )
            setContextOpen( !contextOpen );

        if ( action === 'load-model' )
        {
            setOpen( false );
            loadModel( target.dataset.name );
        }

        if ( action === 'manage-models' )
        {
            setOpen( false );
            actions.openSettings( 'models' );
        }
    }

    function onDocumentClick( event )
    {
        if ( open && !picker.contains( event.target ) )
            setOpen( false );

        if ( contextOpen && !context.contains( event.target ) )
            setContextOpen( false );
    }

    root.addEventListener( 'click', onClick );
    document.addEventListener( 'click', onDocumentClick );

    function buildMenu( state )
    {
        clear( menu );

        const current = state.session === null ? null : state.session.model;
        const heading = element( 'div', 'menu-label', 'Installed' );
        menu.append( heading );

        for ( const model of state.installedModels )
        {
            const item = button( 'menu-item model-menu-item', null, 'load-model' );
            const check = element( 'span', 'menu-check' );
            const name = element( 'span', 'menu-item-name', modelTitle( model ) );

            item.dataset.name = model.name;
            item.classList.toggle( 'current', model.name === current );

            if ( model.name === current )
                setIcon( check, 'check' );

            item.append( check, name );
            menu.append( item );
        }

        const separator = element( 'div', 'menu-separator' );
        const manage = button( 'menu-item', 'Manage models…', 'manage-models' );
        menu.append( separator, manage );
    }

    function render( state, surface )
    {
        // A wide window's collapsed sidebar is a rail, which has the way back itself.
        toggle.hidden = !surface.narrow || surface.sidebarOverlay;
        picker.hidden = isAskingForModel( state );

        // Ahead of the facts' early return: the status and the title change while they do not.
        const working = appStatus( state );
        status.hidden = working === null;
        statusLabel.textContent = working === null ? '' : working;

        const current = state.conversations.find( ( item ) => item.id === state.conversationId );
        conversationTitle.textContent = current === undefined ? '' : current.title;
        conversationTitle.title = conversationTitle.textContent;

        const key = JSON.stringify( [ state.session, state.installedModels ] );

        if ( key === lastKey )
            return;

        lastKey = key;
        buildMenu( state );
        clear( facts );

        if ( state.session === null )
        {
            pickerLabel.textContent = '';
            context.hidden = true;
            setContextOpen( false );

            return;
        }

        // A model being loaded is named at once: the header's status says it is loading.
        const session = state.session;
        const named = session.loading ? session.loading : session.model;
        pickerButton.classList.toggle( 'empty', !named );
        pickerLabel.textContent = named ? namedTitle( state, named ) : 'No model loaded';

        // The ring is the resident model's: with none, there is no context to fill.
        context.hidden = session.model === null;

        if ( session.model === null )
        {
            setContextOpen( false );

            return;
        }

        renderContext( session );

        if ( session.activity !== undefined && session.activity !== 'ask' )
        {
            const where = session.folder ? ' · ' + folderName( session.folder ) : '';
            const activity = element( 'span', 'fact', activityCard( session.activity ).name + where );
            const separator = element( 'span', 'fact-separator', '·' );

            if ( session.folder )
                activity.title = session.folder;

            facts.append( activity );
        }

        const thinking = thinkingFact( session );

        if ( thinking !== null )
        {
            if ( facts.childElementCount > 0 )
                facts.append( element( 'span', 'fact-separator', '·' ) );

            facts.append( element( 'span', 'fact', thinking ) );
        }
    }

    return { render };
}
