// Mila's licence, from the person's menu: the text the person accepted when Mila first started, to read again.

import { element, button, clear } from '../dom.js';
import { licenseParagraphs } from '../format.js';

export function createLicenseView( dialog, actions )
{
    function onClick( event )
    {
        if ( event.target === dialog || event.target.closest( '[data-action="close-license"]' ) !== null )
            dialog.close();
    }

    dialog.addEventListener( 'click', onClick );

    function open()
    {
        const license = actions.getState().license;
        const box = element( 'div', 'license-text' );
        const foot = element( 'div', 'license-foot' );

        box.tabIndex = 0;
        box.setAttribute( 'role', 'document' );
        box.textContent = license === null || license.text === null
            ? ( license === null ? '' : license.problem )
            : licenseParagraphs( license.text );
        foot.append( button( 'setup-button primary', 'Close', 'close-license' ) );

        clear( dialog );
        dialog.append( element( 'h2', 'license-title', 'Mila’s licence' ), box, foot );
        dialog.showModal();
        box.focus();
    }

    return { open, isOpen: () => dialog.open, close: () => dialog.close() };
}
