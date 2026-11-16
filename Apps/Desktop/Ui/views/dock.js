// The conversation risen over the pane above the Mila bar (AssistantInterface.md 5.1): its head says what it is,
// with Open as a whole chat, New conversation and lower. The conversation itself is the transcript, which the page
// moves here while it is risen.

import { element, iconButton, clear } from '../dom.js';
import { currentChat } from '../state.js';

export function createDockView( head, actions )
{
    const context = element( 'div', 'conversation-context' );
    const title = element( 'div', 'conversation-title' );
    const placement = element( 'div', 'conversation-placement', 'Not in a project' );
    const expand = iconButton( 'quiet-icon', 'expand', 'Open as a whole chat', 'expand-conversation' );
    const fresh = iconButton( 'quiet-icon', 'plus', 'New conversation', 'new-conversation' );
    const lower = iconButton( 'quiet-icon', 'chevronDown', 'Lower the conversation', 'lower-conversation' );

    context.append( title, placement );
    clear( head );
    head.append( context, expand, fresh, lower );

    function onClick( event )
    {
        const target = event.target.closest( '[data-action]' );

        if ( target === null )
            return;

        const action = target.dataset.action;

        if ( action === 'expand-conversation' )
            actions.openPlace( 'chat' );

        if ( action === 'new-conversation' )
            actions.newConversation();

        if ( action === 'lower-conversation' )
            actions.dispatch( { type: 'conversation.raise', risen: false } );
    }

    head.addEventListener( 'click', onClick );

    function render( state )
    {
        const chat = currentChat( state );
        const named = chat === null ? 'New chat' : chat.title;

        if ( title.textContent !== named )
        {
            title.textContent = named;
            title.title = named;
        }

        expand.hidden = state.transcript.length === 0;
    }

    return { render };
}
