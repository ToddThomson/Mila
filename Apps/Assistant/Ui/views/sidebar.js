// The sidebar (AssistantInterface.md 4.1): always the same places, with where the person is marked among them; the
// recent chats; and Settings at its foot. The model is the bar's (4.2).

import { element, button, setIcon, clear } from '../dom.js';
import { recentChats, recentProjects } from '../state.js';
import { stateDot, stateWords } from './place-parts.js';

export function createSidebarView( root, actions )
{
    const top = element( 'div', 'sidebar-top' );
    const foot = element( 'div', 'sidebar-foot' );
    const development = element( 'div', 'development' );
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

        if ( action === 'open-place' )
            actions.openPlace( target.dataset.place );

        if ( action === 'open-conversation' )
            actions.openConversation( target.dataset.identifier );

        if ( action === 'open-project' )
            actions.openProject( target.dataset.project );

        // Mila is asked for a project here; the sheet that makes one without asking is on All projects (4.1).
        if ( action === 'new-project' )
            actions.askForProject();

        if ( action === 'open-search' )
            actions.openSearch();

        if ( action === 'open-settings' )
            actions.openSettings( null );
    }

    root.addEventListener( 'click', onClick );

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
            placeItem( 'projects', 'Projects', 'projects', place === 'projects', state.projects.length ),
            placeItem( 'chats', 'Chats', 'chats', place === 'chats', state.conversations.length ),
            placeItem( 'activity', 'Activity', 'activity', place === 'activity', null )
        ];

        clear( top );
        top.append( search, element( 'div', 'side-rule' ), ...places );

        // The five most recently active projects, then New project; left out while there are none (4.1), when
        // New project is under Projects.
        const projects = recentProjects( state );
        const create = button( 'side-item side-new', null, 'new-project' );
        const createIcon = element( 'span', 'side-icon' );

        setIcon( createIcon, 'plus' );
        create.append( createIcon, element( 'span', 'side-label', 'New project' ) );

        if ( projects.length > 0 )
        {
            const title = element( 'div', 'side-title' );
            title.append( element( 'span', null, 'Projects' ), element( 'span', 'side-count', String( state.projects.length ) ) );
            top.append( title );

            for ( const summary of projects )
            {
                const row = button( 'side-item side-project', null, 'open-project' );
                const dot = stateDot( summary.state );

                row.dataset.project = summary.project.id;
                row.title = summary.project.name + ' · ' + stateWords[ summary.state ];
                row.append( dot, element( 'span', 'side-label', summary.project.name ) );

                if ( place === 'project' && summary.project.id === state.projectId )
                    row.setAttribute( 'aria-current', 'page' );

                top.append( row );
            }

            top.append( create );
        }
        else if ( place === 'projects' )
        {
            top.append( element( 'div', 'side-title', 'Projects' ), create );
        }

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

    // Settings and its shortcut; About Mila Assistant, the License and What Mila can reach are in Settings' About.
    function renderFoot()
    {
        const settings = button( 'side-item', null, 'open-settings' );
        const settingsIcon = element( 'span', 'side-icon' );
        const shortcut = element( 'kbd' );

        setIcon( settingsIcon, 'settings' );
        shortcut.textContent = actions.shortcutLabel( ',' );
        settings.append( settingsIcon, element( 'span', 'side-label', 'Settings' ), shortcut );

        clear( foot );
        foot.append( settings, development );
    }

    function render( state )
    {
        const chats = state.conversations.map( ( item ) => [ item.id, item.title, item.updated, item.project, item.progress, item.finish ] );
        const key = JSON.stringify( [ chats, state.projects, state.events.length, state.place, state.projectId, state.conversationId,
            state.session, state.progress, state.restore ] );

        if ( key === lastKey )
            return;

        lastKey = key;
        renderTop( state );

        if ( foot.childElementCount === 0 )
            renderFoot();
    }

    return { render, development };
}
