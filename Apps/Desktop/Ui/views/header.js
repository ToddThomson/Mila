// The bar over the pane (AssistantInterface.md 4.2): where the person is, as a path they can step back along. A narrow
// window's sidebar opens from its menu button.

import { element, button, iconButton, setIcon, clear } from '../dom.js';
import { currentChat } from '../state.js';

const placeNames = { chats: 'Chats', activity: 'Activity' };

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

        if ( target.dataset.action === 'open-place' )
            actions.openPlace( target.dataset.place );
    }

    root.addEventListener( 'click', onClick );

    function step( place, label )
    {
        const made = button( 'crumb', label, 'open-place' );
        made.dataset.place = place;

        return made;
    }

    /** The place shown last, and the steps back to the workspace before it. */
    function path( state )
    {
        if ( state.place === 'workspace' )
            return [ [ null, 'Your Workspace' ] ];

        if ( state.place === 'chat' )
        {
            const chat = currentChat( state );

            return [ [ 'workspace', 'Your Workspace' ], [ 'chats', 'Chats' ], [ null, chat === null ? 'New chat' : chat.title ] ];
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
