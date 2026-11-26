// The bar over the pane (AssistantInterface.md 4.2): where the person is, as a path they can step back along, and at
// its end the model as a ring filled by the open chat's share of the context, whose details open on a click. A
// narrow window's sidebar opens from its menu button.

import { element, button, iconButton, setIcon, clear } from '../dom.js';
import { currentChat, currentProject, projectOfChat, openChatProject, appStatus } from '../state.js';
import { contextShare, contextDetails, thinkingFact, namedTitle } from '../format.js';

const placeNames = { projects: 'Projects', chats: 'Chats', activity: 'Activity', reach: 'What Mila can reach' };

// The ring's arc: a circle of radius 7 drawn from twelve o'clock, filled by the share of the context used.
const ringRadius = 7;
const ringLength = 2 * Math.PI * ringRadius;
const ringMarkup = '<svg viewBox="0 0 20 20" aria-hidden="true">'
    + '<circle class="context-ring-track" cx="10" cy="10" r="7"></circle>'
    + '<circle class="context-ring-used" cx="10" cy="10" r="7" transform="rotate(-90 10 10)"></circle></svg>';

export function createHeaderView( root, actions )
{
    const toggle = iconButton( 'quiet-icon sidebar-toggle', 'menu', 'Show the sidebar', 'toggle-sidebar' );
    const crumbs = element( 'nav', 'crumbs' );
    const status = element( 'div', 'bar-status' );
    const local = { open: false };
    let lastKey = '';
    let lastStatusKey = '';

    crumbs.setAttribute( 'aria-label', 'Where you are' );
    root.append( toggle, crumbs, status );

    function onClick( event )
    {
        const target = event.target.closest( '[data-action]' );

        if ( target === null )
            return;

        const action = target.dataset.action;

        if ( action === 'toggle-model' )
        {
            setOpen( !local.open );

            return;
        }

        setOpen( false );

        if ( action === 'toggle-sidebar' )
            actions.toggleSidebar();

        if ( action === 'manage-models' )
            actions.openSettings( 'models' );

        if ( action === 'open-place' && target.dataset.project )
            actions.openProject( target.dataset.project );
        else if ( action === 'open-place' )
            actions.openPlace( target.dataset.place );
    }

    function setOpen( open )
    {
        if ( local.open === open )
            return;

        local.open = open;
        lastStatusKey = '';
        renderStatus( actions.getState() );
    }

    // By class, not by containment: the click that opened the popover drew the ring again, so its target is no longer
    // in the bar when the click reaches the document.
    function onDocumentClick( event )
    {
        if ( local.open && !event.target.closest( '.bar-popover, .model-status' ) )
            setOpen( false );
    }

    function onKey( event )
    {
        if ( event.key === 'Escape' && local.open )
            setOpen( false );
    }

    root.addEventListener( 'click', onClick );
    document.addEventListener( 'click', onDocumentClick );
    document.addEventListener( 'keydown', onKey );

    function step( place, label )
    {
        const made = button( 'crumb', label, 'open-place' );

        // A project's step names the project: `project:<id>`.
        if ( place.startsWith( 'project:' ) )
            made.dataset.project = place.slice( 'project:'.length );
        else
            made.dataset.place = place;

        return made;
    }

    /**
     * The place shown last, and the steps back to the workspace before it: *Workspace / Projects / Calculator app*,
     * or for a chat in a project *Workspace / Calculator app / its title* (4.2).
     */
    function path( state )
    {
        if ( state.place === 'workspace' )
            return [ [ null, 'Workspace' ] ];

        if ( state.place === 'project' )
        {
            const project = currentProject( state );

            return [ [ 'workspace', 'Workspace' ], [ 'projects', 'Projects' ], [ null, project === null ? 'Project' : project.name ] ];
        }

        if ( state.place === 'chat' )
        {
            const chat = currentChat( state );
            const project = chat === null ? openChatProject( state ) : projectOfChat( state, chat );
            const title = chat === null ? 'New chat' : chat.title;

            if ( project !== null )
                return [ [ 'workspace', 'Workspace' ], [ 'project:' + project.id, project.name ], [ null, title ] ];

            return [ [ 'workspace', 'Workspace' ], [ 'chats', 'Chats' ], [ null, title ] ];
        }

        return [ [ 'workspace', 'Workspace' ], [ null, placeNames[ state.place ] ] ];
    }

    function renderCrumbs( state )
    {
        const steps = path( state );
        const key = JSON.stringify( steps );

        if ( key === lastKey )
            return;

        lastKey = key;
        clear( crumbs );

        const icon = element( 'span', 'crumb-icon' );
        setIcon( icon, 'workspace' );
        crumbs.append( icon );

        steps.forEach( ( [ place, label ], index ) =>
        {
            if ( index > 0 )
                crumbs.append( element( 'span', 'crumb-separator', '/' ) );

            if ( place === null )
            {
                const here = element( 'span', 'crumb-here', label );
                here.setAttribute( 'aria-current', 'page' );
                here.title = label;
                crumbs.append( here );

                return;
            }

            crumbs.append( step( place, label ) );
        } );
    }

    function ring( share )
    {
        const made = element( 'span', 'context-ring' );

        made.innerHTML = ringMarkup;
        made.querySelector( '.context-ring-used' ).setAttribute( 'stroke-dasharray', share * ringLength + ' ' + ringLength );
        made.classList.toggle( 'full', share >= 0.9 );

        return made;
    }

    /**
     * The model's ring: what the app is doing to the model while it works, else the open chat's share of its context,
     * empty with no chat open. The model's name is in its details and its tooltip.
     */
    function modelButton( state )
    {
        const session = state.session;
        const model = button( 'model-status', null, 'toggle-model' );
        const working = appStatus( state );
        const named = session === null ? null : session.loading ? session.loading : session.model;
        const title = named ? namedTitle( state, named ) : 'No model loaded';

        model.setAttribute( 'aria-haspopup', 'dialog' );
        model.setAttribute( 'aria-expanded', local.open ? 'true' : 'false' );

        if ( working !== null )
        {
            const spinner = element( 'span', 'caret' );
            spinner.setAttribute( 'aria-hidden', 'true' );
            model.append( spinner, element( 'span', 'model-status-note', working ) );
            model.title = title;

            return model;
        }

        if ( !named )
        {
            model.append( element( 'span', 'model-status-note', title ) );

            return model;
        }

        const share = contextShare( session );

        model.append( ring( share === null ? 0 : share ) );
        model.title = share === null ? title : title + ' · Context: ' + contextDetails( session )[ 0 ][ 1 ];
        model.setAttribute( 'aria-label', model.title );

        return model;
    }

    /** What Mila is running on, and the open chat's share of its context; the model is changed in Settings. */
    function modelPopover( state )
    {
        const session = state.session;
        const popover = element( 'div', 'bar-popover menu' );
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

    function renderStatus( state )
    {
        const key = JSON.stringify( [ state.session, state.progress, state.restore, state.installedModels,
            state.publishedModels === null, local.open ] );

        if ( key === lastStatusKey )
            return;

        lastStatusKey = key;
        clear( status );
        status.append( modelButton( state ) );

        if ( local.open )
            status.append( modelPopover( state ) );
    }

    function render( state, surface )
    {
        toggle.hidden = !surface.narrow;
        renderCrumbs( state );
        renderStatus( state );
    }

    return { render };
}
