// The sidebar (AssistantInterface.md 4.1): always the same places, with where the person is marked among them; the
// recent chats; and at its foot Settings, the model as a quiet fact, and the person.

import { element, button, setIcon, clear, milaMark } from '../dom.js';
import { recentChats, appStatus } from '../state.js';
import { contextShare, contextDetails, thinkingFact, namedTitle, initials } from '../format.js';

// The ring's arc: a circle of radius 7 drawn from twelve o'clock, filled by the share of the context used.
const ringRadius = 7;
const ringLength = 2 * Math.PI * ringRadius;
const ringMarkup = '<svg viewBox="0 0 20 20" aria-hidden="true">'
    + '<circle class="context-ring-track" cx="10" cy="10" r="7"></circle>'
    + '<circle class="context-ring-used" cx="10" cy="10" r="7" transform="rotate(-90 10 10)"></circle></svg>';

export function createSidebarView( root, actions )
{
    const top = element( 'div', 'sidebar-top' );
    const foot = element( 'div', 'sidebar-foot' );
    const development = element( 'div', 'development' );
    const local = { popover: null };
    let lastKey = '';

    development.id = 'development';
    development.hidden = true;
    root.append( top, foot );

    function onClick( event )
    {
        const target = event.target.closest( '[data-action]' );

        if ( target === null || !root.contains( target ) )
            return;

        const action = target.dataset.action;

        if ( action === 'toggle-model' || action === 'toggle-person' )
        {
            const popover = action === 'toggle-model' ? 'model' : 'person';
            togglePopover( popover );

            return;
        }

        closePopover();

        if ( action === 'open-place' )
            actions.openPlace( target.dataset.place );

        if ( action === 'open-conversation' )
            actions.openConversation( target.dataset.identifier );

        if ( action === 'open-search' )
            actions.openSearch();

        if ( action === 'open-settings' )
            actions.openSettings( null );

        if ( action === 'manage-models' )
            actions.openSettings( 'models' );

        if ( action === 'open-license' )
            actions.openLicense();

        if ( action === 'open-about' )
            actions.openAbout();

        if ( action === 'quit' )
            actions.transmit( { type: 'app.quit' } );
    }

    function togglePopover( popover )
    {
        local.popover = local.popover === popover ? null : popover;
        lastKey = '';
        render( actions.getState(), null );
    }

    function closePopover()
    {
        if ( local.popover === null )
            return;

        local.popover = null;
        lastKey = '';
        render( actions.getState(), null );
    }

    function onDocumentClick( event )
    {
        if ( local.popover !== null && !event.target.closest( '.sidebar-popover, .model-button, .person-button' ) )
            closePopover();
    }

    root.addEventListener( 'click', onClick );
    document.addEventListener( 'click', onDocumentClick );

    function placeItem( place, label, icon, current, count )
    {
        const item = button( 'side-item', null, 'open-place' );
        const holder = element( 'span', 'side-icon' );
        const name = element( 'span', 'side-label', label );

        setIcon( holder, icon );
        item.dataset.place = place;
        item.append( holder, name );

        if ( count !== null )
            item.append( element( 'span', 'side-count', String( count ) ) );

        if ( current )
            item.setAttribute( 'aria-current', 'page' );

        return item;
    }

    function renderTop( state )
    {
        const workspace = button( 'side-workspace', null, 'open-place' );
        const name = element( 'span', 'side-name' );

        workspace.dataset.place = 'workspace';
        name.append( element( 'b', null, 'Your Workspace' ), element( 'span', null, 'Mila Assistant' ) );
        workspace.append( milaMark( 'side-mark' ), name );

        const search = button( 'side-item', null, 'open-search' );
        const searchIcon = element( 'span', 'side-icon' );
        const shortcut = element( 'kbd' );
        setIcon( searchIcon, 'search' );
        shortcut.dataset.shortcut = 'K';
        shortcut.textContent = actions.shortcutLabel( 'K' );
        search.append( searchIcon, element( 'span', 'side-label', 'Search' ), shortcut );

        const place = state.place;
        const places = [
            placeItem( 'workspace', 'Workspace', 'workspace', place === 'workspace', null ),
            placeItem( 'chats', 'Chats', 'chats', place === 'chats', state.conversations.length ),
            placeItem( 'activity', 'Activity', 'activity', place === 'activity', null )
        ];

        clear( top );
        top.append( workspace, search, element( 'div', 'side-rule' ), ...places );

        const recent = recentChats( state );

        if ( recent.length === 0 )
            return;

        top.append( element( 'div', 'side-title', 'Recent chats' ) );

        for ( const item of recent )
        {
            const row = button( 'side-item side-chat', null, 'open-conversation' );
            row.dataset.identifier = item.id;
            row.title = item.title;
            row.append( element( 'span', 'side-dot' ), element( 'span', 'side-label', item.title ) );

            if ( place === 'chat' && item.id === state.conversationId )
                row.setAttribute( 'aria-current', 'page' );

            top.append( row );
        }
    }

    /** The model's row: what the app is doing to it while it works, else its name and the open chat's share of the context. */
    function modelButton( state )
    {
        const session = state.session;
        const model = button( 'side-item model-button', null, 'toggle-model' );
        const working = appStatus( state );
        const named = session === null ? null : session.loading ? session.loading : session.model;

        model.setAttribute( 'aria-haspopup', 'dialog' );
        model.setAttribute( 'aria-expanded', local.popover === 'model' ? 'true' : 'false' );

        if ( working !== null )
        {
            const ring = element( 'span', 'caret' );
            ring.setAttribute( 'aria-hidden', 'true' );
            model.append( ring, element( 'span', 'side-label', named ? namedTitle( state, named ) : working ),
                element( 'span', 'side-note', working ) );

            return model;
        }

        if ( !named )
        {
            model.append( element( 'span', 'side-dot empty' ), element( 'span', 'side-label', 'No model loaded' ) );

            return model;
        }

        model.append( element( 'span', 'side-dot model' ), element( 'span', 'side-label', namedTitle( state, named ) ) );

        const share = contextShare( session );

        if ( share === null )
        {
            model.append( element( 'span', 'side-note', 'Model' ) );

            return model;
        }

        const ring = element( 'span', 'context-ring' );
        ring.innerHTML = ringMarkup;
        ring.querySelector( '.context-ring-used' ).setAttribute( 'stroke-dasharray', share * ringLength + ' ' + ringLength );
        ring.classList.toggle( 'full', share >= 0.9 );
        model.title = 'Context: ' + contextDetails( session )[ 0 ][ 1 ];
        model.append( ring );

        return model;
    }

    /** What Mila is running on, and the open chat's share of its context; the model is changed in Settings. */
    function modelPopover( state )
    {
        const session = state.session;
        const popover = element( 'div', 'sidebar-popover menu' );
        const named = session === null ? null : session.loading ? session.loading : session.model;

        popover.setAttribute( 'role', 'dialog' );
        popover.setAttribute( 'aria-label', 'Model' );
        popover.append( element( 'div', 'menu-label', 'Model' ),
            element( 'div', 'popover-title', named ? namedTitle( state, named ) : 'No model loaded' ) );

        if ( session !== null && session.model !== null )
        {
            const rows = contextDetails( session );
            const thinking = thinkingFact( session );

            if ( thinking !== null )
                rows.push( [ 'Thinking', thinking.replace( /^Thinking /, '' ) ] );

            for ( const [ name, value ] of rows )
            {
                const row = element( 'div', 'context-row' );
                row.append( element( 'span', 'context-row-name', name ), element( 'span', 'context-row-value', value ) );
                popover.append( row );
            }
        }

        popover.append( element( 'div', 'menu-separator' ), button( 'menu-item', 'Manage models…', 'manage-models' ) );

        return popover;
    }

    function personButton( state )
    {
        const name = state.about === null || state.about.userName === '' ? 'You' : state.about.userName;
        const person = button( 'person-button', null, 'toggle-person' );
        const text = element( 'span', 'side-name' );
        const chevrons = element( 'span', 'side-chevrons' );

        person.setAttribute( 'aria-haspopup', 'menu' );
        person.setAttribute( 'aria-expanded', local.popover === 'person' ? 'true' : 'false' );
        text.append( element( 'b', null, name ), element( 'span', null, 'Workspace owner' ) );
        setIcon( chevrons, 'chevronUpDown' );
        person.append( element( 'span', 'avatar', initials( name ) ), text, chevrons );

        return person;
    }

    function personMenu( state )
    {
        const menu = element( 'div', 'sidebar-popover menu' );
        const settings = button( 'menu-item', null, 'open-settings' );
        const shortcut = element( 'kbd' );

        menu.setAttribute( 'role', 'menu' );
        shortcut.textContent = actions.shortcutLabel( ',' );
        settings.append( element( 'span', 'menu-item-name', 'Settings' ), shortcut );
        const reach = button( 'menu-item', 'What Mila can reach', 'open-place' );

        reach.dataset.place = 'reach';
        menu.append( settings, reach, button( 'menu-item', 'Mila’s licence', 'open-license' ),
            element( 'div', 'menu-separator' ),
            button( 'menu-item', 'About Mila Assistant', 'open-about' ),
            button( 'menu-item', 'Quit Mila', 'quit' ) );

        return menu;
    }

    function renderFoot( state )
    {
        const settings = button( 'side-item', null, 'open-settings' );
        const settingsIcon = element( 'span', 'side-icon' );
        const shortcut = element( 'kbd' );

        setIcon( settingsIcon, 'settings' );
        shortcut.textContent = actions.shortcutLabel( ',' );
        settings.append( settingsIcon, element( 'span', 'side-label', 'Settings' ), shortcut );

        clear( foot );
        foot.append( settings, modelButton( state ), personButton( state ), development );

        if ( local.popover === 'model' )
            foot.append( modelPopover( state ) );

        if ( local.popover === 'person' )
            foot.append( personMenu( state ) );
    }

    function render( state )
    {
        const chats = state.conversations.map( ( item ) => [ item.id, item.title, item.updated ] );
        const key = JSON.stringify( [ chats, state.place, state.conversationId, state.session, state.progress, state.restore,
            state.installedModels, state.publishedModels === null, state.about, local ] );

        if ( key === lastKey )
            return;

        lastKey = key;
        renderTop( state );
        renderFoot( state );
    }

    return { render, development };
}
