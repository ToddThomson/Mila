// The log drawer: the session's log lines, below the conversation, while the detail setting is
// "Everything, with logs". Closed until the person opens it; opening it shows the newest line.

import { element, button, setIcon, clear } from '../dom.js';

export function createLogView( root, actions )
{
    const toggle = button( 'log-toggle', null, 'toggle-log' );
    const chevron = element( 'span', 'chevron' );
    const label = element( 'span', 'log-label', 'Log' );
    const count = element( 'span', 'log-count' );
    const body = element( 'div', 'log-body' );
    let shownLogs = null;

    setIcon( chevron, 'chevronRight' );
    toggle.append( chevron, label, count );
    root.append( toggle, body );

    function onClick()
    {
        const change = { logOpen: !root.classList.contains( 'open' ) };

        actions.changeSurface( change );
    }

    toggle.addEventListener( 'click', onClick );

    function fill( logs )
    {
        clear( body );

        for ( const entry of logs )
        {
            const line = element( 'div', 'log-entry', entry.text );
            line.classList.toggle( 'warning', entry.level === 'warning' || entry.level === 'error' );
            body.append( line );
        }

        body.scrollTop = body.scrollHeight;
    }

    function render( state, surface )
    {
        const shown = state.session !== null && state.session.detail === 'all';
        root.hidden = !shown;

        if ( !shown )
            return;

        const open = surface.logOpen;
        root.classList.toggle( 'open', open );
        toggle.setAttribute( 'aria-expanded', open ? 'true' : 'false' );
        count.textContent = state.logs.length === 0 ? 'nothing yet' : String( state.logs.length );
        body.hidden = !open;

        if ( !open || shownLogs === state.logs )
            return;

        shownLogs = state.logs;
        fill( state.logs );
    }

    return { render };
}
