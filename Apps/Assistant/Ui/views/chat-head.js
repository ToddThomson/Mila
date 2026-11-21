// A chat with the pane to itself (AssistantInterface.md 4.7): Chat, its title, where it is and when, above its
// conversation.

import { element, button, clear } from '../dom.js';
import { currentChat, projectOfChat, openChatProject } from '../state.js';
import { whenLabel, folderName } from '../format.js';

export function createChatHeadView( root, actions )
{
    let lastKey = '';

    root.addEventListener( 'click', ( event ) =>
    {
        const target = event.target.closest( '[data-action]' );

        if ( target === null )
            return;

        if ( target.dataset.action === 'open-reach' )
            actions.openPlace( 'reach' );

        if ( target.dataset.action === 'open-project' )
            actions.openProject( target.dataset.project );
    } );

    function render( state )
    {
        const chat = currentChat( state );
        const project = chat === null ? openChatProject( state ) : projectOfChat( state, chat );
        const now = new Date();
        const title = chat === null ? 'New chat' : chat.title;
        const parts = [];
        const folder = state.session === null ? null : state.session.folder;

        // A folder is read until the person allows a change in it.
        if ( folder )
            parts.push( ( state.session.folder_access === 'read_write' ? 'Working in ' : 'Reading ' ) + folderName( folder ) );

        if ( chat !== null )
            parts.push( whenLabel( chat.updated, now ) );

        const key = JSON.stringify( [ title, project === null ? null : [ project.id, project.name ], parts, folder ] );

        if ( key === lastKey )
            return;

        lastKey = key;
        clear( root );

        const placement = element( 'div', 'chat-placement' );

        // Where it is: *In Kyoto in November*, which opens the project, or *Not in a project*.
        if ( project === null )
        {
            placement.append( 'Not in a project' );
        }
        else
        {
            const opener = button( 'link', project.name, 'open-project' );
            opener.dataset.project = project.id;
            placement.append( 'In ', opener );
        }

        for ( const part of parts )
            placement.append( ' · ' + part );

        if ( folder )
        {
            placement.title = folder;
            placement.append( ' · ', button( 'link', 'What Mila can reach', 'open-reach' ) );
        }

        root.append( element( 'div', 'eyebrow', 'Chat' ), element( 'h1', 'chat-title', title ), placement );
    }

    return { render };
}
