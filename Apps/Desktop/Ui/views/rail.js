// The collapsed sidebar: a rail of three buttons, so the conversations, a new one and Settings stay one click away.
// A narrow window has none: its sidebar opens over the conversation from the header instead.

export function createRailView( root, actions )
{
    function onClick( event )
    {
        const target = event.target.closest( '[data-action]' );

        if ( target === null || !root.contains( target ) )
            return;

        const action = target.dataset.action;

        if ( action === 'show-sidebar' )
            actions.toggleSidebar();

        if ( action === 'new-conversation' )
            actions.newConversation();

        if ( action === 'open-settings' )
        {
            const opened = { settingsOpen: true };
            actions.changeSurface( opened );
        }
    }

    root.addEventListener( 'click', onClick );

    function render( state, surface )
    {
        root.hidden = surface.narrow || !surface.sidebarCollapsed;
    }

    return { render };
}
