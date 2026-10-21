// The sidebar: a new conversation, the stored conversations by date, and the way to the models
// view and settings. A row's menu renames in place and asks before deleting.

import { element, button, iconButton, clear } from '../dom.js';
import { groupConversations } from '../state.js';

export function createSidebarView( root, actions )
{
    const list = root.querySelector( '.conversation-list' );
    const local = { menuFor: null, renaming: null, confirming: null };
    const modelsPanel = { modelsOpen: true, settingsOpen: false };
    const settingsPanel = { settingsOpen: true };
    const noConfirmation = { confirming: null };
    const noRename = { renaming: null };
    const noMenu = { menuFor: null };
    let lastConversations = null;
    let lastConversationId = null;
    let lastLocal = '';

    function changeLocal( change )
    {
        Object.assign( local, change );
        lastConversations = null;

        const state = actions.getState();
        renderList( state );
    }

    function onClick( event )
    {
        const target = event.target.closest( '[data-action]' );

        if ( target === null || !root.contains( target ) )
            return;

        const action = target.dataset.action;
        const identifier = target.dataset.identifier;

        if ( action === 'new-conversation' )
            actions.newConversation();

        if ( action === 'collapse-sidebar' )
            actions.toggleSidebar();

        if ( action === 'open-models' )
            openPanel( modelsPanel );

        if ( action === 'open-settings' )
            openPanel( settingsPanel );

        if ( action === 'open-conversation' )
            openConversation( identifier );

        if ( action === 'row-menu' )
            toggleMenu( identifier );

        if ( action === 'rename' )
            startRename( identifier );

        if ( action === 'delete' )
            confirmDelete( identifier );

        if ( action === 'delete-confirmed' )
            deleteConversation( identifier );

        if ( action === 'delete-cancelled' )
            changeLocal( noConfirmation );
    }

    function openPanel( change )
    {
        const closed = { ...change, sidebarOverlay: false };

        actions.changeSurface( closed );
    }

    function openConversation( identifier )
    {
        const message = { type: 'conversation.open', id: identifier };
        const closed = { sidebarOverlay: false, modelsOpen: false };

        actions.send( message );
        actions.changeSurface( closed );
    }

    function toggleMenu( identifier )
    {
        const menuFor = local.menuFor === identifier ? null : identifier;
        const change = { menuFor, confirming: null };

        changeLocal( change );
    }

    function startRename( identifier )
    {
        const change = { menuFor: null, renaming: identifier };

        changeLocal( change );
    }

    function confirmDelete( identifier )
    {
        const change = { menuFor: null, confirming: identifier };

        changeLocal( change );
    }

    function deleteConversation( identifier )
    {
        const message = { type: 'conversation.delete', id: identifier };
        const change = { confirming: null };

        actions.send( message );
        changeLocal( change );
    }

    function commitRename( input )
    {
        const identifier = input.dataset.identifier;
        const title = input.value.trim();
        const state = actions.getState();
        const conversation = state.conversations.find( ( item ) => item.id === identifier );
        const change = { renaming: null };

        if ( conversation !== undefined && title.length > 0 && title !== conversation.title )
        {
            const message = { type: 'conversation.rename', id: identifier, title };
            actions.send( message );
        }

        changeLocal( change );
    }

    function onKeyDown( event )
    {
        const input = event.target.closest( '.rename-input' );

        if ( input === null )
            return;

        event.stopPropagation();

        if ( event.key === 'Enter' )
            commitRename( input );

        if ( event.key === 'Escape' )
            changeLocal( noRename );
    }

    function onFocusOut( event )
    {
        const input = event.target.closest( '.rename-input' );

        if ( input !== null && local.renaming !== null )
            commitRename( input );
    }

    function onDocumentClick( event )
    {
        if ( local.menuFor === null || event.target.closest( '.row-menu, .row-menu-button' ) !== null )
            return;

        changeLocal( noMenu );
    }

    root.addEventListener( 'click', onClick );
    root.addEventListener( 'keydown', onKeyDown );
    root.addEventListener( 'focusout', onFocusOut );
    document.addEventListener( 'click', onDocumentClick );

    function buildRow( item, active )
    {
        const row = element( 'div', 'conversation-row' );
        row.classList.toggle( 'active', active );
        row.title = 'Held with ' + item.model;

        if ( local.renaming === item.id )
        {
            const input = element( 'input', 'rename-input' );
            input.value = item.title;
            input.dataset.identifier = item.id;
            input.setAttribute( 'aria-label', 'Conversation title' );
            row.append( input );

            return row;
        }

        if ( local.confirming === item.id )
        {
            row.classList.add( 'confirming' );

            const question = element( 'span', 'confirm-question', 'Delete this conversation?' );
            const confirm = button( 'confirm-delete', 'Delete', 'delete-confirmed' );
            const cancel = button( 'confirm-cancel', 'Cancel', 'delete-cancelled' );
            confirm.dataset.identifier = item.id;
            row.append( question, confirm, cancel );

            return row;
        }

        const open = button( 'conversation-open', item.title, 'open-conversation' );
        const menuButton = iconButton( 'row-menu-button quiet-icon', 'more', 'Conversation options', 'row-menu' );
        open.dataset.identifier = item.id;
        menuButton.dataset.identifier = item.id;
        row.append( open, menuButton );

        if ( local.menuFor === item.id )
        {
            row.classList.add( 'menu-open' );

            const menu = element( 'div', 'row-menu menu' );
            const rename = button( 'menu-item', 'Rename', 'rename' );
            const remove = button( 'menu-item danger', 'Delete', 'delete' );
            rename.dataset.identifier = item.id;
            remove.dataset.identifier = item.id;
            menu.append( rename, remove );
            row.append( menu );
        }

        return row;
    }

    function renderList( state )
    {
        const localKey = JSON.stringify( local );

        if ( state.conversations === lastConversations && state.conversationId === lastConversationId && localKey === lastLocal )
            return;

        lastConversations = state.conversations;
        lastConversationId = state.conversationId;
        lastLocal = localKey;

        const now = new Date();
        const groups = groupConversations( state.conversations, now );

        clear( list );

        if ( groups.length === 0 )
        {
            const empty = element( 'p', 'conversation-empty', 'Your conversations will appear here.' );
            list.append( empty );
        }

        for ( const group of groups )
        {
            const section = element( 'section', 'conversation-group' );
            const heading = element( 'h2', 'conversation-group-label', group.label );
            section.append( heading );

            for ( const item of group.items )
            {
                const active = item.id === state.conversationId;
                const row = buildRow( item, active );
                section.append( row );
            }

            list.append( section );
        }

        const input = list.querySelector( '.rename-input' );

        if ( input !== null )
        {
            input.focus();
            input.select();
        }
    }

    function render( state )
    {
        renderList( state );
    }

    return { render };
}
