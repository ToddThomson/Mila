// About Mila Assistant, from the person's menu: the version, and under Licences Mila's own and each the app carries,
// which the window names by an id; a text is shown here, and Chromium's opens in the browser (MilaDesktop.md 6.2).

import { element, button, clear, milaMark } from '../dom.js';
import { licenseParagraphs } from '../format.js';

const milaLicence = 'mila';

export function createAboutView( dialog, actions )
{
    // The licence the page asked the window for, whose text is shown when it comes back; null shows the list.
    let asked = null;

    function onClick( event )
    {
        if ( event.target === dialog || event.target.closest( '[data-action="close-about"]' ) !== null )
        {
            dialog.close();

            return;
        }

        if ( event.target.closest( '[data-action="about-back"]' ) !== null )
        {
            showList();

            return;
        }

        const chosen = event.target.closest( '[data-licence]' );

        if ( chosen !== null )
            choose( chosen.dataset.licence );
    }

    dialog.addEventListener( 'click', onClick );

    function about()
    {
        const known = actions.getState().about;

        return known === null ? { version: '', licences: [] } : known;
    }

    function foot( withBack )
    {
        const row = element( 'div', 'license-foot' );

        if ( withBack )
            row.append( button( 'setup-button', 'Back', 'about-back' ) );

        row.append( button( 'setup-button primary', 'Close', 'close-about' ) );

        return row;
    }

    function showList()
    {
        const known = about();
        const head = element( 'div', 'about-head' );
        const title = element( 'div' );
        const list = element( 'div', 'about-licences' );
        const names = [ { id: milaLicence, name: 'Mila' }, ...known.licences ];

        asked = null;
        title.append( element( 'h2', 'license-title', 'Mila Assistant' ),
            element( 'p', 'about-version', known.version ? 'Version ' + known.version : '' ) );
        head.append( milaMark( 'about-mark' ), title );
        list.setAttribute( 'role', 'list' );

        for ( const entry of names )
        {
            const row = button( 'about-licence', entry.name, null );

            row.dataset.licence = entry.id;
            row.setAttribute( 'role', 'listitem' );
            list.append( row );
        }

        clear( dialog );
        dialog.append( head, element( 'h3', 'about-heading', 'Licences' ), list, foot( false ) );
    }

    /** Mila's licence is reflowed as the setup shows it; another project's is kept in its own lines. */
    function showText( name, text, ownLines )
    {
        const box = element( 'div', ownLines ? 'license-text about-text' : 'license-text' );

        box.tabIndex = 0;
        box.setAttribute( 'role', 'document' );
        box.textContent = text;

        clear( dialog );
        dialog.append( element( 'h2', 'license-title', name ), box, foot( true ) );
        box.focus();
    }

    function nameOf( id )
    {
        const entry = about().licences.find( ( candidate ) => candidate.id === id );

        return entry === undefined ? null : entry.name;
    }

    /** Mila's own is the text the person accepted; every other is the window's to give. */
    function choose( id )
    {
        if ( id === milaLicence )
        {
            const license = actions.getState().license;
            const text = license === null || license.text === null
                ? ( license === null ? '' : license.problem )
                : licenseParagraphs( license.text );

            showText( 'Mila', text, false );

            return;
        }

        asked = id;
        actions.transmit( { type: 'licence.open', id } );
    }

    /** The window's answer to licence.open, shown if it is still the one asked for and the dialog is open. */
    function received( message )
    {
        if ( !dialog.open || message.id !== asked )
            return;

        const name = nameOf( message.id );

        if ( name === null )
            return;

        showText( name, typeof message.text === 'string' ? message.text : message.problem ?? '', true );
    }

    function open()
    {
        showList();
        dialog.showModal();
    }

    return { open, received, isOpen: () => dialog.open, close: () => dialog.close() };
}
