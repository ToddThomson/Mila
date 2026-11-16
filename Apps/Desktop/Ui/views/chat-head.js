// A chat with the pane to itself (AssistantInterface.md 4.7): Chat, its title, where it is and when, above its
// conversation.

import { element, clear } from '../dom.js';
import { currentChat } from '../state.js';
import { whenLabel, folderName } from '../format.js';

export function createChatHeadView( root )
{
    let lastKey = '';

    function render( state )
    {
        const chat = currentChat( state );
        const now = new Date();
        const title = chat === null ? 'New chat' : chat.title;
        const parts = [ 'Not in a project' ];
        const folder = state.session === null ? null : state.session.folder;

        // A chat begun in a folder before chats were general purpose still works there.
        if ( folder )
            parts.push( 'Working in ' + folderName( folder ) );

        if ( chat !== null )
            parts.push( whenLabel( chat.updated, now ) );

        const key = JSON.stringify( [ title, parts, folder ] );

        if ( key === lastKey )
            return;

        lastKey = key;
        clear( root );

        const placement = element( 'div', 'chat-placement', parts.join( ' · ' ) );

        if ( folder )
            placement.title = folder;

        root.append( element( 'div', 'eyebrow', 'Chat' ), element( 'h1', 'chat-title', title ), placement );
    }

    return { render };
}
