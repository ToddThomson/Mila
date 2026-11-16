// Search (AssistantInterface.md 4.11): one box over the pane, the chats first and then the files Mila wrote, each
// with where it is; arrow keys and Enter, Esc to close.

import { element, button, clear } from '../dom.js';
import { searchResults } from '../state.js';
import { folderName } from '../format.js';

export function createSearchView( dialog, actions )
{
    const input = element( 'input', 'search-window-input' );
    const list = element( 'div', 'search-results' );
    let selected = 0;
    let results = [];

    input.type = 'search';
    input.placeholder = 'Jump to a chat or a file';
    input.autocomplete = 'off';
    input.setAttribute( 'aria-label', 'Search' );
    input.setAttribute( 'aria-controls', 'search-results' );
    list.id = 'search-results';
    list.setAttribute( 'role', 'listbox' );
    dialog.append( input, list );

    function fill()
    {
        results = searchResults( actions.getState(), input.value );
        selected = Math.min( selected, Math.max( 0, results.length - 1 ) );
        clear( list );

        if ( results.length === 0 )
        {
            list.append( element( 'div', 'search-empty', input.value.trim() === '' ? 'No chats yet.' : 'Nothing matches.' ) );

            return;
        }

        results.forEach( ( result, index ) =>
        {
            const option = button( 'search-result', null, 'choose' );
            const label = result.kind === 'chat' ? result.item.title : folderName( result.path );
            const kind = result.kind === 'chat' ? 'Chat' : 'File · in ' + result.item.title;

            option.dataset.index = String( index );
            option.setAttribute( 'role', 'option' );
            option.setAttribute( 'aria-selected', index === selected ? 'true' : 'false' );
            option.title = result.kind === 'chat' ? result.item.title : result.path;
            option.append( element( 'span', 'search-result-label', label ), element( 'span', 'search-result-kind', kind ) );
            list.append( option );
        } );

        list.querySelector( '[aria-selected="true"]' )?.scrollIntoView( { block: 'nearest' } );
    }

    function choose( index )
    {
        const result = results[ index ];

        if ( result === undefined )
            return;

        close();
        actions.openConversation( result.item.id );
    }

    function onInput()
    {
        selected = 0;
        fill();
    }

    function onKeyDown( event )
    {
        if ( event.key === 'ArrowDown' || event.key === 'ArrowUp' )
        {
            event.preventDefault();
            selected = event.key === 'ArrowDown' ? Math.min( selected + 1, results.length - 1 ) : Math.max( selected - 1, 0 );
            fill();

            return;
        }

        if ( event.key === 'Enter' )
        {
            event.preventDefault();
            choose( selected );
        }
    }

    function onClick( event )
    {
        // A click on the backdrop lands on the dialog itself.
        if ( event.target === dialog )
        {
            close();

            return;
        }

        const option = event.target.closest( '[data-action="choose"]' );

        if ( option !== null )
            choose( Number( option.dataset.index ) );
    }

    input.addEventListener( 'input', onInput );
    input.addEventListener( 'keydown', onKeyDown );
    dialog.addEventListener( 'click', onClick );

    function open()
    {
        if ( dialog.open )
            return;

        input.value = '';
        selected = 0;
        fill();
        dialog.showModal();
        input.focus();
    }

    function close()
    {
        if ( dialog.open )
            dialog.close();
    }

    return { open, close, isOpen: () => dialog.open };
}
