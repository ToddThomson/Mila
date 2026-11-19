// The sheet before a link in a reply opens (AssistantInterface.md 5.7): the whole address, every time. The words the
// link was shown under are the model's, so they are not repeated here.

import { element, button, clear } from '../dom.js';
import { linkDescription } from '../format.js';

export function createLinkView( dialog, actions )
{
    let shown = null;

    function onClick( event )
    {
        if ( event.target === dialog || event.target.closest( '[data-action="link-cancel"]' ) !== null )
        {
            dialog.close();

            return;
        }

        if ( event.target.closest( '[data-action="link-confirm"]' ) !== null )
        {
            const url = shown;

            dialog.close();
            actions.linkConfirmed( url );
        }
    }

    dialog.addEventListener( 'click', onClick );

    function open( url )
    {
        const description = linkDescription( url );

        if ( description === null )
            return;

        const facts = element( 'dl', 'link-facts' );
        const foot = element( 'div', 'license-foot' );

        for ( const [ label, value ] of description.facts )
            facts.append( element( 'dt', null, label ), element( 'dd', null, value ) );

        // Cancel comes first, so it is what the sheet focuses as it opens.
        foot.append( button( 'setup-button', 'Cancel', 'link-cancel' ), button( 'setup-button primary', description.action, 'link-confirm' ) );

        shown = url;
        clear( dialog );
        dialog.append( element( 'h2', 'license-title', description.title ), facts, element( 'p', 'link-note', description.note ), foot );
        dialog.showModal();
    }

    return { open, isOpen: () => dialog.open };
}
