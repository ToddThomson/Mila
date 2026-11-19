// The places of the workspace (AssistantInterface.md 4): the overview the window opens on, all chats, activity, and
// what Mila can reach. Each is drawn from the session's records; the open chat is the transcript's.

import { element, button, iconButton, clear } from '../dom.js';
import { workspaceOverview, matchingChats, activityDays, activityKinds, reachOverview } from '../state.js';
import { whenLabel, attentionLine, plainLine, activityText, longDate, dayLabel, clockLabel, toolWord } from '../format.js';

const activityFilters = { all: 'Everything', chat: 'Chats', file: 'Files', plan: 'Plans', sent: 'Sent from this computer' };

// What the person can do about a reply that ended early: one cut short can be finished; one that ran out of room or
// failed is opened, since asking again there would end the same way.
const replyActions = { length: 'Ask Mila to finish', context: 'Open the chat', error: 'Open the chat' };
const finishRequest = 'Please finish your last reply.';

export function createPlacesView( scroller, root, actions )
{
    const local = { query: '', kind: 'all', menuFor: null, renaming: null, confirming: null };
    const search = element( 'input', 'search-input' );
    const list = element( 'div', 'rows chat-rows' );
    let lastKey = '';
    let lastPlace = null;

    search.type = 'search';
    search.placeholder = 'Search chats';
    search.setAttribute( 'aria-label', 'Search chats' );
    search.autocomplete = 'off';
    search.addEventListener( 'input', onSearchInput );

    function onSearchInput()
    {
        local.query = search.value;
        redraw();
    }

    function redraw()
    {
        lastKey = '';
        render( actions.getState() );
    }

    function changeLocal( change )
    {
        Object.assign( local, change );
        redraw();
    }

    function onClick( event )
    {
        const target = event.target.closest( '[data-action]' );

        if ( target === null || !root.contains( target ) )
            return;

        const action = target.dataset.action;
        const identifier = target.dataset.identifier;

        if ( action === 'open-place' )
            actions.openPlace( target.dataset.place );

        if ( action === 'open-conversation' )
            actions.openConversation( identifier );

        if ( action === 'finish-reply' )
            actions.openConversation( identifier, finishRequest );

        if ( action === 'show-section' )
            root.querySelector( '#' + target.dataset.section )?.scrollIntoView( { block: 'start' } );

        if ( action === 'activity-kind' )
            changeLocal( { kind: target.dataset.kind } );

        if ( action === 'show-sent' )
        {
            local.kind = 'sent';
            actions.openPlace( 'activity' );
        }

        if ( action === 'remove-folder' )
            actions.send( { type: 'reach.remove', kind: 'folder', id: identifier ?? '' } );

        if ( action === 'remove-allowed' )
            actions.send( { type: 'reach.remove', kind: 'allowed', name: target.dataset.name } );

        if ( action === 'row-menu' )
            changeLocal( { menuFor: local.menuFor === identifier ? null : identifier, confirming: null } );

        if ( action === 'rename' )
            changeLocal( { menuFor: null, renaming: identifier } );

        if ( action === 'delete' )
            changeLocal( { menuFor: null, confirming: identifier } );

        if ( action === 'delete-confirmed' )
        {
            actions.send( { type: 'conversation.delete', id: identifier } );
            changeLocal( { confirming: null } );
        }

        if ( action === 'delete-cancelled' )
            changeLocal( { confirming: null } );
    }

    function commitRename( input )
    {
        const identifier = input.dataset.identifier;
        const title = input.value.trim();
        const conversation = actions.getState().conversations.find( ( item ) => item.id === identifier );

        if ( conversation !== undefined && title.length > 0 && title !== conversation.title )
            actions.send( { type: 'conversation.rename', id: identifier, title } );

        changeLocal( { renaming: null } );
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
            changeLocal( { renaming: null } );
    }

    function onFocusOut( event )
    {
        const input = event.target.closest( '.rename-input' );

        if ( input !== null && local.renaming !== null )
            commitRename( input );
    }

    function onDocumentClick( event )
    {
        if ( local.menuFor !== null && event.target.closest( '.row-menu, .row-menu-button' ) === null )
            changeLocal( { menuFor: null } );
    }

    root.addEventListener( 'click', onClick );
    root.addEventListener( 'keydown', onKeyDown );
    root.addEventListener( 'focusout', onFocusOut );
    document.addEventListener( 'click', onDocumentClick );

    // ---- Shared pieces -------------------------------------------------------------------------

    function head( eyebrow, title )
    {
        const made = element( 'div', 'place-head' );
        made.append( element( 'div', 'eyebrow', eyebrow ), element( 'h1', 'place-title', title ) );

        return made;
    }

    function section( title, identifier, aside )
    {
        const made = element( 'section', 'section' );
        const top = element( 'div', 'section-head' );

        if ( identifier )
            made.id = identifier;

        top.append( element( 'h2', 'section-title', title ) );

        if ( aside )
            top.append( aside );

        made.append( top );

        return made;
    }

    /** A chat as a row: its title, the start of Mila's last reply, and when. */
    function chatRow( item, now )
    {
        const row = button( 'row', null, 'open-conversation' );
        const main = element( 'span', 'row-main' );
        const preview = plainLine( item.reply_start );

        row.dataset.identifier = item.id;
        main.append( element( 'span', 'row-title', item.title ) );
        main.append( element( 'span', 'row-sub', preview.length > 0 ? preview : 'No reply yet' ) );
        row.append( main, element( 'span', 'row-when', whenLabel( item.updated, now ) ) );

        return row;
    }

    function figure( value, label, note, action, modifier )
    {
        const made = button( modifier ? 'figure ' + modifier : 'figure', null, action.action );

        Object.assign( made.dataset, action.data );
        made.append( element( 'span', 'figure-value', String( value ) ), element( 'span', 'figure-label', label ),
            element( 'span', 'figure-note', note ) );

        return made;
    }

    // ---- The workspace overview (4.3) -----------------------------------------------------------

    function noticeFor( row )
    {
        const notice = element( 'div', 'notice-card waiting' );
        const actionsRow = element( 'div', 'notice-actions' );
        const item = row.item;
        const finishing = row.kind === 'reply' && item.finish === 'length';
        const label = row.kind === 'plan' ? 'Answer and continue' : replyActions[ item.finish ];
        const act = button( 'button', label, finishing ? 'finish-reply' : 'open-conversation' );

        act.dataset.identifier = item.id;
        actionsRow.append( act );
        notice.append( element( 'div', 'notice-kind', 'Needs you' ), element( 'div', 'notice-title', item.title ),
            element( 'div', 'notice-text', attentionLine( row ) ), actionsRow );

        return notice;
    }

    function renderWorkspace( state, now )
    {
        const overview = workspaceOverview( state );
        const waiting = overview.waiting;
        const glance = element( 'div', 'glance' );
        const newest = overview.newest === null ? 'None yet' : 'Newest: ' + overview.newest.title;
        const waitingNote = waiting.length === 0 ? 'Nothing right now' : waiting.map( ( row ) => row.item.title ).join( ', ' );
        const toWaiting = waiting.length === 0 ? { action: null, data: {} } : { action: 'show-section', data: { section: 'needs-you' } };

        glance.append( figure( overview.chatCount, 'Chats', newest, { action: 'open-place', data: { place: 'chats' } }, null ),
            figure( waiting.length, 'Waiting on you', waitingNote, toWaiting, waiting.length > 0 ? 'attention' : null ) );

        const atGlance = section( 'At a glance', null, null );
        atGlance.append( glance );
        root.append( head( longDate( now ), 'Your Workspace' ), atGlance );

        if ( waiting.length > 0 )
        {
            const needs = section( 'Needs you', 'needs-you', null );
            const notices = element( 'div', 'notices' );

            notices.append( ...waiting.map( noticeFor ) );
            needs.append( notices );
            root.append( needs );
        }

        const all = button( 'link', 'All ' + overview.chatCount + ( overview.chatCount === 1 ? ' chat' : ' chats' ), 'open-place' );
        all.dataset.place = 'chats';

        const chats = section( 'Chats', 'chats', overview.chatCount > 0 ? all : null );
        const rows = element( 'div', 'rows' );

        if ( overview.chats.length === 0 )
            rows.append( element( 'div', 'empty', 'No chats yet.' ) );
        else
            rows.append( ...overview.chats.map( ( item ) => chatRow( item, now ) ) );

        chats.append( rows );
        root.append( chats );
    }

    // ---- All chats (4.6) --------------------------------------------------------------------------

    function managedRow( item, now )
    {
        const holder = element( 'div', 'row-holder' );

        if ( local.renaming === item.id )
        {
            const input = element( 'input', 'rename-input' );
            input.value = item.title;
            input.dataset.identifier = item.id;
            input.setAttribute( 'aria-label', 'Chat title' );
            holder.append( input );

            return holder;
        }

        if ( local.confirming === item.id )
        {
            const confirm = button( 'button danger', 'Delete', 'delete-confirmed' );
            confirm.dataset.identifier = item.id;
            holder.classList.add( 'confirming' );
            holder.append( element( 'span', 'confirm-question', 'Delete “' + item.title + '”?' ), confirm,
                button( 'button quiet', 'Cancel', 'delete-cancelled' ) );

            return holder;
        }

        const menuButton = iconButton( 'row-menu-button quiet-icon', 'more', 'Chat options', 'row-menu' );
        menuButton.dataset.identifier = item.id;
        holder.append( chatRow( item, now ), menuButton );

        if ( local.menuFor === item.id )
        {
            const menu = element( 'div', 'row-menu menu' );
            const rename = button( 'menu-item', 'Rename', 'rename' );
            const remove = button( 'menu-item danger', 'Delete', 'delete' );

            holder.classList.add( 'menu-open' );
            rename.dataset.identifier = item.id;
            remove.dataset.identifier = item.id;
            menu.append( rename, remove );
            holder.append( menu );
        }

        return holder;
    }

    function renderChats( state, now )
    {
        const shown = matchingChats( state, local.query );
        const tools = element( 'div', 'tools' );

        if ( search.value !== local.query )
            search.value = local.query;

        tools.append( search );
        clear( list );

        if ( shown.length === 0 )
            list.append( element( 'div', 'empty', state.conversations.length === 0 ? 'No chats yet.' : 'No chat matches.' ) );
        else
            list.append( ...shown.map( ( item ) => managedRow( item, now ) ) );

        root.append( head( 'Your Workspace', 'All chats' ), tools, list );

        const input = list.querySelector( '.rename-input' );

        if ( input !== null )
        {
            input.focus();
            input.select();
        }
    }

    // ---- Activity (4.8) ---------------------------------------------------------------------------

    function renderActivity( state, now )
    {
        const tools = element( 'div', 'tools' );

        for ( const kind of activityKinds )
        {
            const filter = button( 'filter', activityFilters[ kind ], 'activity-kind' );
            filter.dataset.kind = kind;
            filter.setAttribute( 'aria-pressed', local.kind === kind ? 'true' : 'false' );
            tools.append( filter );
        }

        root.append( head( 'Your Workspace', 'Activity' ), tools );

        const days = activityDays( state, local.kind );

        if ( days.length === 0 )
        {
            root.append( element( 'div', 'empty activity-empty', 'Nothing yet.' ) );

            return;
        }

        for ( const day of days )
        {
            const block = element( 'section', 'day' );
            const timeline = element( 'div', 'timeline' );

            for ( const event of day.events )
            {
                const entry = element( 'div', 'event ' + event.who );
                const text = element( 'div', 'event-text' );
                const open = button( 'link', activityText( event ), 'open-conversation' );
                const where = eventWhere( event );

                open.dataset.identifier = event.item.id;
                text.append( open, element( 'div', 'event-where', where ) );
                entry.append( element( 'span', 'event-time', clockLabel( event.time ) ), text );
                timeline.append( entry );
            }

            block.append( element( 'h2', 'section-title', dayLabel( day.day, now ) ), timeline );
            root.append( block );
        }
    }

    function eventWhere( event )
    {
        if ( event.kind === 'sent' )
            return ( event.who === 'mila' ? 'Mila' : 'You' ) + ' · in ' + event.item.title + ' · to ' + event.sent.where;

        return event.who === 'mila' ? 'Mila · in ' + event.item.title : 'You';
    }

    // ---- What Mila can reach (4.12) ------------------------------------------------------------------

    function reachRow( title, note, remove )
    {
        const row = element( 'div', 'reach-row' );
        const main = element( 'div', 'reach-main' );

        main.append( element( 'div', 'reach-title', title ), element( 'div', 'row-sub reach-note', note ) );
        row.append( main );

        if ( remove !== null )
            row.append( remove );

        return row;
    }

    function renderReach( state )
    {
        const reach = reachOverview( state );
        const folders = section( 'Folders', null, null );
        const folderRows = element( 'div', 'rows' );

        if ( reach.folders.length === 0 )
            folderRows.append( element( 'div', 'empty', 'No chat has a folder.' ) );

        for ( const folder of reach.folders )
        {
            const remove = button( 'button quiet', 'Remove', 'remove-folder' );
            const access = folder.changes ? 'Read and change' : 'Read';

            if ( folder.id !== null )
                remove.dataset.identifier = folder.id;

            remove.setAttribute( 'aria-label', 'Remove ' + folder.path );
            folderRows.append( reachRow( folder.path, access + ' · ' + folder.title, remove ) );
        }

        folders.append( folderRows );

        const chat = section( 'Allowed for this chat', null, null );
        const allowedRows = element( 'div', 'rows' );

        if ( reach.allowed.length === 0 )
            allowedRows.append( element( 'div', 'empty', 'Nothing. Mila asks each time.' ) );

        for ( const name of reach.allowed )
        {
            const remove = button( 'button quiet', 'Remove', 'remove-allowed' );

            remove.dataset.name = name;
            allowedRows.append( reachRow( toolWord( name ), 'Without asking, until this chat ends', remove ) );
        }

        chat.append( allowedRows );

        const internet = section( 'The internet', null, button( 'link', 'See what was sent', 'show-sent' ) );
        const internetRows = element( 'div', 'rows' );

        internetRows.append( reachRow( 'Web pages', 'Mila asks before each page, showing its address', null ),
            reachRow( 'Web search', reach.webSearch ? 'Mila asks before each search, showing its words' : 'Off: no search key is set', null ),
            reachRow( 'Commands', reach.commands ? 'Mila asks before each one' : 'Off', null ) );
        internet.append( internetRows );

        root.append( head( 'Your Workspace', 'What Mila can reach' ), folders, chat, internet );
    }

    function render( state )
    {
        const place = state.place;

        if ( place === 'chat' )
        {
            lastPlace = place;

            return;
        }

        const now = new Date();
        const session = state.session;
        const reachKey = session === null ? null : [ session.folder, session.folder_access, session.allowed_for_chat, session.web_search, session.run_commands ];
        const key = JSON.stringify( [ place, state.conversations, state.conversationId, session === null ? null : session.busy, reachKey,
            local, now.getMinutes() ] );

        if ( key === lastKey )
            return;

        lastKey = key;

        // Typing in the search keeps its place; anything else redrawn keeps where the person scrolled to.
        const typing = document.activeElement === search;
        const scrolled = place === lastPlace ? scroller.scrollTop : 0;

        clear( root );

        if ( place === 'chats' )
            renderChats( state, now );
        else if ( place === 'activity' )
            renderActivity( state, now );
        else if ( place === 'reach' )
            renderReach( state );
        else
            renderWorkspace( state, now );

        scroller.scrollTop = scrolled;
        lastPlace = place;

        if ( typing )
            search.focus();
    }

    /** A place shown again is read from the top. */
    function reset()
    {
        local.menuFor = null;
        local.renaming = null;
        local.confirming = null;
        lastPlace = null;
        lastKey = '';
    }

    return { render, reset };
}
