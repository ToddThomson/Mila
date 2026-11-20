// The places of the workspace (AssistantInterface.md 4): the overview the window opens on, all projects and a project,
// all chats, activity, and what Mila can reach. Each is drawn from the session's records; the open chat is the
// transcript's.

import { element, button, iconButton, clear } from '../dom.js';
import { workspaceOverview, filteredChats, chatFilters, activityDays, activityKinds, reachOverview, projectOfChat } from '../state.js';
import { attentionLine, activityText, longDate, dayLabel, clockLabel, toolWord } from '../format.js';
import { head, section, chatRow, figure } from './place-parts.js';
import { renderProjects, renderProject, projectCard } from './projects.js';

const activityFilters = { all: 'Everything', project: 'Projects', chat: 'Chats', file: 'Files', plan: 'Plans', sent: 'Sent from this computer' };
const chatFilterWords = { all: 'All', loose: 'Not in a project', placed: 'In a project' };

// What the person can do about a reply that ended early: one cut short can be finished; one that ran out of room or
// failed is opened, since asking again there would end the same way.
const replyActions = { length: 'Ask Mila to finish', context: 'Open the chat', error: 'Open the chat' };
const finishRequest = 'Please finish your last reply.';

export function createPlacesView( scroller, root, actions )
{
    const local = {
        query: '', kind: 'all', chatFilter: 'all', menuFor: null, movingFor: null, renaming: null, confirming: null,
        projectQuery: '', projectFilter: 'all', projectSort: 'recent', projectMenu: false, editing: null, confirmingProject: null
    };
    const search = element( 'input', 'search-input' );
    const projectSearch = element( 'input', 'search-input' );
    const list = element( 'div', 'rows chat-rows' );
    let lastKey = '';
    let lastPlace = null;

    search.type = 'search';
    search.placeholder = 'Search chats';
    search.setAttribute( 'aria-label', 'Search chats' );
    search.autocomplete = 'off';
    search.addEventListener( 'input', onSearchInput );

    projectSearch.type = 'search';
    projectSearch.placeholder = 'Search projects';
    projectSearch.setAttribute( 'aria-label', 'Search projects' );
    projectSearch.autocomplete = 'off';
    projectSearch.addEventListener( 'input', () =>
    {
        local.projectQuery = projectSearch.value;
        redraw();
    } );

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
            changeLocal( { menuFor: local.menuFor === identifier ? null : identifier, movingFor: null, confirming: null } );

        if ( action === 'move' )
            changeLocal( { movingFor: identifier } );

        if ( action === 'move-to' )
        {
            const project = target.dataset.project === '' ? null : target.dataset.project;

            actions.send( { type: 'conversation.move', id: identifier, project } );
            changeLocal( { menuFor: null, movingFor: null } );
        }

        if ( action === 'chat-filter' )
            changeLocal( { chatFilter: target.dataset.filter } );

        if ( action === 'activity-earlier' )
            actions.send( { type: 'activity.open', kind: null, before: actions.getState().eventsBefore } );

        onProjectClick( action, target );

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

    /** What the projects' places do (4.4, 4.5); every change is one `project.update` or `project.delete`. */
    function onProjectClick( action, target )
    {
        const project = target.dataset.project;

        if ( action === 'open-project' )
            actions.openProject( project );

        if ( action === 'new-project' )
            actions.newProject();

        if ( action === 'project-filter' )
            changeLocal( { projectFilter: target.dataset.filter } );

        if ( action === 'project-sort' )
            changeLocal( { projectSort: target.dataset.sort } );

        if ( action === 'project-menu' )
            changeLocal( { projectMenu: !local.projectMenu } );

        if ( action === 'project-rename' )
            changeLocal( { projectMenu: false, editing: 'name' } );

        if ( action === 'project-purpose' )
            changeLocal( { projectMenu: false, editing: 'purpose' } );

        if ( action === 'project-add-folder' )
        {
            changeLocal( { projectMenu: false } );
            actions.chooseProjectFolder( project );
        }

        if ( action === 'project-remove-folder' )
            changeFolders( project, ( folders ) => folders.filter( ( folder ) => folder.path !== target.dataset.path ) );

        if ( action === 'project-delete' )
            changeLocal( { projectMenu: false, confirmingProject: project } );

        if ( action === 'project-delete-keep' || action === 'project-delete-all' )
        {
            actions.send( { type: 'project.delete', id: project, conversations: action === 'project-delete-all' ? 'delete' : 'keep' } );
            changeLocal( { confirmingProject: null } );
        }

        if ( action === 'project-delete-cancel' )
            changeLocal( { confirmingProject: null } );
    }

    /** A project's folders changed whole, as `project.update` takes them. */
    function changeFolders( projectId, change )
    {
        const project = actions.getState().projects.find( ( candidate ) => candidate.id === projectId );

        if ( project === undefined )
            return;

        const folders = Array.isArray( project.folders ) ? project.folders : [];

        actions.send( { type: 'project.update', id: projectId, folders: change( folders.map( ( folder ) => ( { ...folder } ) ) ) } );
    }

    function onChange( event )
    {
        const select = event.target.closest( '.folder-access' );

        if ( select === null )
            return;

        const path = select.dataset.path;

        changeFolders( select.dataset.project, ( folders ) => folders.map( ( folder ) =>
            ( folder.path === path ? { ...folder, access: select.value } : folder ) ) );
    }

    /** A project's name or purpose, as edited in its head; an empty name is not one. */
    function commitProjectEdit( input )
    {
        const kind = input.dataset.edit;
        const state = actions.getState();
        const project = state.projects.find( ( candidate ) => candidate.id === state.projectId );
        const value = input.value.trim();

        local.editing = null;

        if ( project !== undefined && kind === 'name' && value.length > 0 && value !== project.name )
            actions.send( { type: 'project.update', id: project.id, name: value } );

        if ( project !== undefined && kind === 'purpose' && value !== ( project.purpose ?? '' ) )
            actions.send( { type: 'project.update', id: project.id, purpose: value } );

        redraw();
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

        // A project's name or purpose; a purpose takes a new line with Shift.
        if ( input.dataset.edit !== undefined )
        {
            if ( event.key === 'Enter' && !event.shiftKey )
            {
                event.preventDefault();
                commitProjectEdit( input );
            }

            if ( event.key === 'Escape' )
                changeLocal( { editing: null } );

            return;
        }

        if ( event.key === 'Enter' )
            commitRename( input );

        if ( event.key === 'Escape' )
            changeLocal( { renaming: null } );
    }

    function onFocusOut( event )
    {
        const input = event.target.closest( '.rename-input' );

        if ( input !== null && input.dataset.edit !== undefined && local.editing !== null )
            commitProjectEdit( input );
        else if ( input !== null && local.renaming !== null )
            commitRename( input );
    }

    function onDocumentClick( event )
    {
        const outside = event.target.closest( '.row-menu, .row-menu-button' ) === null;

        if ( ( local.menuFor !== null || local.projectMenu ) && outside )
            changeLocal( { menuFor: null, movingFor: null, projectMenu: false } );
    }

    root.addEventListener( 'click', onClick );
    root.addEventListener( 'change', onChange );
    root.addEventListener( 'keydown', onKeyDown );
    root.addEventListener( 'focusout', onFocusOut );
    document.addEventListener( 'click', onDocumentClick );

    // ---- The workspace overview (4.3) -----------------------------------------------------------

    function noticeFor( row )
    {
        const notice = element( 'div', 'notice-card waiting' );
        const actionsRow = element( 'div', 'notice-actions' );
        const item = row.item;
        const finishing = row.kind === 'reply' && item.finish === 'length';
        const label = row.kind === 'plan' ? 'Answer and continue' : replyActions[ item.finish ];
        const act = button( 'button', label, finishing ? 'finish-reply' : 'open-conversation' );
        const project = projectOfChat( actions.getState(), item );

        act.dataset.identifier = item.id;
        actionsRow.append( act );
        notice.append( element( 'div', 'notice-kind', project === null ? 'Needs you' : 'Needs you · ' + project.name ),
            element( 'div', 'notice-title', item.title ), element( 'div', 'notice-text', attentionLine( row ) ), actionsRow );

        return notice;
    }

    /**
     * The overview's figures (4.3): with projects, *Projects*, *Waiting on you* (the projects, by name) and *Chats*
     * (those in no project); with none, *Chats* and *Waiting on you*.
     */
    function glanceOf( overview )
    {
        const glance = element( 'div', 'glance' );
        const waiting = overview.waiting;
        const toWaiting = waiting.length === 0 ? { action: null, data: {} } : { action: 'show-section', data: { section: 'needs-you' } };

        if ( overview.projects.length === 0 )
        {
            const newest = overview.newest === null ? 'None yet' : 'Newest: ' + overview.newest.title;
            const waitingNote = waiting.length === 0 ? 'Nothing right now' : waiting.map( ( row ) => row.item.title ).join( ', ' );

            glance.append( figure( overview.chatCount, 'Chats', newest, { action: 'open-place', data: { place: 'chats' } }, null ),
                figure( waiting.length, 'Waiting on you', waitingNote, toWaiting, waiting.length > 0 ? 'attention' : null ) );

            return glance;
        }

        const named = overview.waitingProjects.map( ( summary ) => summary.project.name );
        const looseWaiting = waiting.filter( ( row ) => projectOfChat( actions.getState(), row.item ) === null ).length;
        const waitingCount = named.length + looseWaiting;
        const waitingNote = waitingCount === 0 ? 'Nothing right now'
            : [ ...named, ...( looseWaiting > 0 ? [ looseWaiting + ( looseWaiting === 1 ? ' chat' : ' chats' ) ] : [] ) ].join( ', ' );
        const active = overview.activeCount;

        glance.append(
            figure( overview.projects.length, 'Projects', active === 0 ? 'None with work under way' : active + ' with work under way',
                { action: 'open-place', data: { place: 'projects' } }, null ),
            figure( waitingCount, 'Waiting on you', waitingNote, toWaiting, waitingCount > 0 ? 'attention' : null ),
            figure( overview.chatCount, 'Chats', overview.looseCount + ' not in a project', { action: 'open-place', data: { place: 'chats' } }, null ) );

        return glance;
    }

    function renderWorkspace( state, now )
    {
        const overview = workspaceOverview( state );
        const waiting = overview.waiting;

        const atGlance = section( 'At a glance', null, null );
        atGlance.append( glanceOf( overview ) );
        root.append( head( longDate( now ), 'Your Workspace' ), atGlance );

        if ( waiting.length > 0 )
        {
            const needs = section( 'Needs you', 'needs-you', null );
            const notices = element( 'div', 'notices' );

            notices.append( ...waiting.map( noticeFor ) );
            needs.append( notices );
            root.append( needs );
        }

        if ( overview.recentProjects.length > 0 )
        {
            const allProjects = button( 'link', 'All ' + overview.projects.length + ( overview.projects.length === 1 ? ' project' : ' projects' ), 'open-place' );
            const cards = element( 'div', 'project-cards' );

            allProjects.dataset.place = 'projects';
            cards.append( ...overview.recentProjects.map( ( summary ) => projectCard( summary, now ) ) );

            const recent = section( 'Recent projects', 'recent-projects', allProjects );
            recent.append( cards );
            root.append( recent );
        }

        const all = button( 'link', 'All ' + overview.chatCount + ( overview.chatCount === 1 ? ' chat' : ' chats' ), 'open-place' );
        all.dataset.place = 'chats';

        // With projects, the chats here are those in none (4.3).
        const chats = section( 'Chats', 'chats', overview.chatCount > 0 ? all : null );
        const rows = element( 'div', 'rows' );

        if ( overview.chats.length === 0 )
            rows.append( element( 'div', 'empty', overview.chatCount === 0 ? 'No chats yet.' : 'Every chat is in a project.' ) );
        else
            rows.append( ...overview.chats.map( ( item ) => chatRow( item, now, null ) ) );

        chats.append( rows );
        root.append( chats );
    }

    // ---- All chats (4.6) --------------------------------------------------------------------------

    /** The chat's menu's list of where it can move: each project but its own, and no project when it is in one. */
    function moveMenu( item )
    {
        const menu = element( 'div', 'row-menu menu move-menu' );
        const state = actions.getState();

        menu.append( element( 'div', 'menu-label', 'Move to' ) );

        for ( const project of state.projects )
        {
            if ( project.id === item.project )
                continue;

            const choice = button( 'menu-item', project.name, 'move-to' );
            choice.dataset.identifier = item.id;
            choice.dataset.project = project.id;
            menu.append( choice );
        }

        if ( typeof item.project === 'string' )
        {
            const none = button( 'menu-item', 'No project', 'move-to' );
            none.dataset.identifier = item.id;
            none.dataset.project = '';
            menu.append( none );
        }

        if ( menu.children.length === 1 )
            menu.append( element( 'div', 'menu-note', 'No project to move it to yet.' ) );

        return menu;
    }

    function managedRow( item, now, where )
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
        holder.append( chatRow( item, now, where ), menuButton );

        if ( local.menuFor === item.id && local.movingFor === item.id )
        {
            holder.classList.add( 'menu-open' );
            holder.append( moveMenu( item ) );
        }
        else if ( local.menuFor === item.id )
        {
            const menu = element( 'div', 'row-menu menu' );
            const rename = button( 'menu-item', 'Rename', 'rename' );
            const move = button( 'menu-item', 'Move to…', 'move' );
            const remove = button( 'menu-item danger', 'Delete', 'delete' );

            holder.classList.add( 'menu-open' );
            rename.dataset.identifier = item.id;
            move.dataset.identifier = item.id;
            remove.dataset.identifier = item.id;
            menu.append( rename, move, remove );
            holder.append( menu );
        }

        return holder;
    }

    function renderChats( state, now )
    {
        const shown = filteredChats( state, local.query, local.chatFilter );
        const tools = element( 'div', 'tools' );

        if ( search.value !== local.query )
            search.value = local.query;

        tools.append( search );

        // The filter is there once the workspace has projects (4.6).
        if ( state.projects.length > 0 )
        {
            for ( const filter of chatFilters )
            {
                const chip = button( 'filter', chatFilterWords[ filter ], 'chat-filter' );
                chip.dataset.filter = filter;
                chip.setAttribute( 'aria-pressed', local.chatFilter === filter ? 'true' : 'false' );
                tools.append( chip );
            }
        }

        clear( list );

        const where = ( item ) =>
        {
            if ( state.projects.length === 0 )
                return null;

            const project = projectOfChat( state, item );

            return project === null ? 'No project' : project.name;
        };

        if ( shown.length === 0 )
            list.append( element( 'div', 'empty', state.conversations.length === 0 ? 'No chats yet.' : 'No chat matches.' ) );
        else
            list.append( ...shown.map( ( item ) => managedRow( item, now, where( item ) ) ) );

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
                const where = eventWhere( event );

                text.append( eventOpener( state, event ), element( 'div', 'event-where', where ) );
                entry.append( element( 'span', 'event-time', clockLabel( event.time ) ), text );
                timeline.append( entry );
            }

            block.append( element( 'h2', 'section-title', dayLabel( day.day, now ) ), timeline );
            root.append( block );
        }

        // The session sends its latest events; older ones come a page at a time.
        if ( state.eventsBefore !== null )
            root.append( button( 'link activity-earlier', 'Show earlier project events', 'activity-earlier' ) );
    }

    /**
     * An entry opens what it names: its chat, or for one of the session's events the chat it moved or the project it
     * names while they are stored. An event about something deleted opens nothing.
     */
    function eventOpener( state, event )
    {
        if ( event.kind !== 'project' )
        {
            const open = button( 'link', activityText( event ), 'open-conversation' );
            open.dataset.identifier = event.item.id;

            return open;
        }

        if ( event.item !== null )
        {
            const open = button( 'link', activityText( event ), 'open-conversation' );
            open.dataset.identifier = event.item.id;

            return open;
        }

        if ( event.to !== null )
        {
            const open = button( 'link', activityText( event ), 'open-project' );
            open.dataset.project = event.project;

            return open;
        }

        return element( 'div', 'event-line', activityText( event ) );
    }

    function eventWhere( event )
    {
        if ( event.kind === 'project' )
            return event.who === 'mila' ? 'Mila' : 'You';

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

        if ( reach.folders.length === 0 && reach.projectFolders.length === 0 )
            folderRows.append( element( 'div', 'empty', 'No chat or project has a folder.' ) );

        // A project's folder is removed from the project, and so from every chat that holds it through the project.
        for ( const folder of reach.projectFolders )
        {
            const remove = button( 'button quiet', 'Remove', 'project-remove-folder' );
            const access = folder.changes ? 'Read and change' : 'Read';

            remove.dataset.project = folder.project.id;
            remove.dataset.path = folder.path;
            remove.setAttribute( 'aria-label', 'Remove ' + folder.path + ' from ' + folder.project.name );
            folderRows.append( reachRow( folder.path, access + ' · every chat in ' + folder.project.name, remove ) );
        }

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
        const key = JSON.stringify( [ place, state.projectId, state.projects, state.events, state.eventsBefore, state.conversations,
            state.conversationId, session === null ? null : [ session.busy, session.project ], reachKey, local, now.getMinutes() ] );

        if ( key === lastKey )
            return;

        lastKey = key;

        // Typing in a search keeps its place; anything else redrawn keeps where the person scrolled to.
        const typing = document.activeElement === search ? search : document.activeElement === projectSearch ? projectSearch : null;
        const scrolled = place === lastPlace ? scroller.scrollTop : 0;

        clear( root );

        if ( place === 'chats' )
            renderChats( state, now );
        else if ( place === 'projects' )
            renderProjects( root, state, local, now, projectSearch );
        else if ( place === 'project' )
            renderProject( root, state, local, now, managedRow );
        else if ( place === 'activity' )
            renderActivity( state, now );
        else if ( place === 'reach' )
            renderReach( state );
        else
            renderWorkspace( state, now );

        scroller.scrollTop = scrolled;
        lastPlace = place;

        if ( typing !== null )
            typing.focus();

        const editing = root.querySelector( '.project-edit' );

        if ( editing !== null && document.activeElement !== editing )
        {
            editing.focus();
            editing.select();
        }
    }

    /** A place shown again is read from the top. */
    function reset()
    {
        local.menuFor = null;
        local.movingFor = null;
        local.renaming = null;
        local.confirming = null;
        local.projectMenu = false;
        local.editing = null;
        local.confirmingProject = null;
        lastPlace = null;
        lastKey = '';
    }

    return { render, reset };
}
