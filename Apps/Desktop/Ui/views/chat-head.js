// A chat with the pane to itself (AssistantInterface.md 4.7): Chat, its title, where it is and when, above its
// conversation.

import { element, button, clear } from '../dom.js';
import { currentChat } from '../state.js';
import { whenLabel, folderName } from '../format.js';

export function createChatHeadView( root, actions )
{
    let lastKey = '';

    root.addEventListener( 'click', ( event ) =>
    {
        if ( event.target.closest( '[data-action="open-reach"]' ) !== null )
            actions.openPlace( 'reach' );
    } );

    function render( state )
    {
        const chat = currentChat( state );
        const now = new Date();
        const title = chat === null ? 'New chat' : chat.title;
        const parts = [ 'Not in a project' ];
        const folder = state.session === null ? null : state.session.folder;

        // A folder is read until the person allows a change in it.
        if ( folder )
            parts.push( ( state.session.folder_access === 'read_write' ? 'Working in ' : 'Reading ' ) + folderName( folder ) );

        if ( chat !== null )
            parts.push( whenLabel( chat.updated, now ) );

        const key = JSON.stringify( [ title, parts, folder ] );

        if ( key === lastKey )
            return;

        lastKey = key;
        clear( root );

        const placement = element( 'div', 'chat-placement', parts.join( ' · ' ) );

        if ( folder )
        {
            placement.title = folder;
            placement.append( ' · ', button( 'link', 'What Mila can reach', 'open-reach' ) );
        }

        root.append( element( 'div', 'eyebrow', 'Chat' ), element( 'h1', 'chat-title', title ), placement );
    }

    return { render };
}
