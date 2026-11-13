// The main pane's header: the loaded model as a picker, and the session's facts as quiet text.

import { element, button, iconButton, setIcon, clear } from '../dom.js';
import { contextFact, thinkingFact, activityCard, folderName } from '../format.js';
import { appStatus, isAskingForModel } from '../state.js';

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
    let open = false;
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
    root.append( toggle, picker, status, conversationTitle, facts );
    menu.hidden = true;

    function setOpen( value )
    {
        open = value;
        menu.hidden = !open;
        pickerButton.setAttribute( 'aria-expanded', open ? 'true' : 'false' );
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
            const name = element( 'span', 'menu-item-name', model.name );

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

            return;
        }

        // A model being loaded is named at once: the header's status says it is loading.
        const session = state.session;
        const named = session.loading ? session.loading : session.model;
        pickerButton.classList.toggle( 'empty', !named );
        pickerLabel.textContent = named ? named : 'No model loaded';

        if ( session.model === null )
            return;

        if ( session.activity !== undefined && session.activity !== 'ask' )
        {
            const where = session.folder ? ' · ' + folderName( session.folder ) : '';
            const activity = element( 'span', 'fact', activityCard( session.activity ).name + where );
            const separator = element( 'span', 'fact-separator', '·' );

            if ( session.folder )
                activity.title = session.folder;

            facts.append( activity, separator );
        }

        const contextText = contextFact( session );
        const context = element( 'span', 'fact', contextText );
        facts.append( context );

        const thinking = thinkingFact( session );

        if ( thinking !== null )
        {
            const separator = element( 'span', 'fact-separator', '·' );
            const fact = element( 'span', 'fact', thinking );
            facts.append( separator, fact );
        }
    }

    return { render };
}
