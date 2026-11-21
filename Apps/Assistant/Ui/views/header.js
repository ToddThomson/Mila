// The bar over the pane (AssistantInterface.md 4.2): where the person is, as a path they can step back along. A narrow
// window's sidebar opens from its menu button.

import { element, button, iconButton, setIcon, clear } from '../dom.js';
import { currentChat, currentProject, projectOfChat, openChatProject } from '../state.js';

const placeNames = { projects: 'Projects', chats: 'Chats', activity: 'Activity', reach: 'What Mila can reach' };

export function createHeaderView( root, actions )
{
    const toggle = iconButton( 'quiet-icon sidebar-toggle', 'menu', 'Show the sidebar', 'toggle-sidebar' );
    const crumbs = element( 'nav', 'crumbs' );
    let lastKey = '';

    crumbs.setAttribute( 'aria-label', 'Where you are' );
    root.append( toggle, crumbs );

    function onClick( event )
    {
        const target = event.target.closest( '[data-action]' );

        if ( target === null )
            return;

        if ( target.dataset.action === 'toggle-sidebar' )
            actions.toggleSidebar();

        if ( target.dataset.action === 'open-place' && target.dataset.project )
            actions.openProject( target.dataset.project );
        else if ( target.dataset.action === 'open-place' )
            actions.openPlace( target.dataset.place );
    }

    root.addEventListener( 'click', onClick );

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
     * The place shown last, and the steps back to the workspace before it: *Your Workspace / Projects / Calculator
     * app*, or for a chat in a project *Your Workspace / Calculator app / its title* (4.2).
     */
    function path( state )
    {
        if ( state.place === 'workspace' )
            return [ [ null, 'Your Workspace' ] ];

        if ( state.place === 'project' )
        {
            const project = currentProject( state );

            return [ [ 'workspace', 'Your Workspace' ], [ 'projects', 'Projects' ], [ null, project === null ? 'Project' : project.name ] ];
        }

        if ( state.place === 'chat' )
        {
            const chat = currentChat( state );
            const project = chat === null ? openChatProject( state ) : projectOfChat( state, chat );
            const title = chat === null ? 'New chat' : chat.title;

            if ( project !== null )
                return [ [ 'workspace', 'Your Workspace' ], [ 'project:' + project.id, project.name ], [ null, title ] ];

            return [ [ 'workspace', 'Your Workspace' ], [ 'chats', 'Chats' ], [ null, title ] ];
        }

        return [ [ 'workspace', 'Your Workspace' ], [ null, placeNames[ state.place ] ] ];
    }

    function render( state, surface )
    {
        toggle.hidden = !surface.narrow;

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

    return { render };
}
