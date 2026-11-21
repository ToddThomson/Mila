// The Made drawer (AssistantInterface.md 4.9): a file Mila wrote or a reply the person kept, its contents where the page
// can show them, whether the file changed on disk after Mila last wrote it, and where it came from. Over the pane from
// the right, closed by Esc or outside it; everything in it is the session's `creation.loaded`.

import { element, button, iconButton, clear } from '../dom.js';
import { renderMarkdown } from '../markdown.js';
import { folderName, clockLabel, dayLabel } from '../format.js';
import { projectName } from '../state.js';

const diskWords = {
    same: null,
    changed: 'Changed on disk after Mila last wrote it',
    missing: 'No longer on disk where Mila wrote it'
};

export function createMadeView( dialog, actions )
{
    let shown = null;

    dialog.addEventListener( 'click', ( event ) =>
    {
        if ( event.target === dialog || event.target.closest( '[data-action="close-made"]' ) !== null )
        {
            dialog.close();

            return;
        }

        const chat = event.target.closest( '[data-action="made-open-chat"]' );

        if ( chat !== null )
        {
            dialog.close();
            actions.openConversation( chat.dataset.identifier );
        }
    } );

    // However it closes, the page forgets what it showed.
    dialog.addEventListener( 'close', () =>
    {
        shown = null;
        actions.dispatch( { type: 'creation.close' } );
    } );

    function when( time, now )
    {
        if ( typeof time !== 'string' )
            return 'Not recorded';

        return dayLabel( new Date( time ).setHours( 0, 0, 0, 0 ), now ) + ', ' + clockLabel( time );
    }

    function fact( label, value )
    {
        const row = element( 'div', 'made-fact' );
        row.append( element( 'dt', null, label ), element( 'dd', null, value ) );

        return row;
    }

    function headOf( title, sub )
    {
        const top = element( 'div', 'made-head' );
        const words = element( 'div', 'made-head-words' );

        words.append( element( 'div', 'eyebrow', 'Made' ), element( 'h2', 'made-title', title ) );

        if ( sub !== null )
            words.append( element( 'div', 'made-path', sub ) );

        top.append( words, iconButton( 'quiet-icon', 'close', 'Close', 'close-made' ) );

        return top;
    }

    function contentsOf( loaded, creation )
    {
        const box = element( 'div', 'made-contents' );
        const text = creation.kind === 'reply' ? creation.text : loaded.content;
        const markdown = creation.kind === 'reply' || loaded.content_kind === 'markdown';

        if ( typeof text !== 'string' )
        {
            box.classList.add( 'empty' );
            box.textContent = loaded.disk === 'missing' ? 'The file is not there now.' : 'This file is not shown here.';

            return box;
        }

        if ( markdown )
        {
            const body = element( 'div', 'markdown' );
            body.innerHTML = renderMarkdown( text );
            box.append( body );
        }
        else
        {
            box.append( element( 'pre', 'made-text', text ) );
        }

        return box;
    }

    /** Where it came from (4.9): its project and chat, the plan step, the model, what Mila read first, every write. */
    function originOf( state, creation, now )
    {
        const origin = element( 'section', 'made-origin' );
        const facts = element( 'dl', 'made-facts' );
        const project = projectName( state, creation.project );
        const chat = button( 'link', creation.title || 'A chat', 'made-open-chat' );

        chat.dataset.identifier = creation.conversation_id;

        const chatFact = element( 'div', 'made-fact' );
        const chatValue = element( 'dd' );
        chatValue.append( chat );
        chatFact.append( element( 'dt', null, 'In the chat' ), chatValue );

        facts.append( fact( 'For', project ?? 'No project' ), chatFact );

        if ( creation.kind === 'file' )
        {
            const made = creation.made;

            facts.append( fact( 'During', made.step ?? 'No plan step' ), fact( 'By', 'Mila, with ' + ( made.model ?? 'a model not recorded' ) ),
                fact( 'Made', when( made.time, now ) ) );
        }
        else
        {
            facts.append( fact( 'By', 'Mila, with ' + ( creation.model ?? 'a model not recorded' ) ), fact( 'Kept', when( creation.kept, now ) ) );
        }

        origin.append( element( 'h3', 'made-heading', 'Where it came from' ), facts );
        origin.append( element( 'h3', 'made-heading', creation.kind === 'file' ? 'What Mila read first' : 'What Mila read' ) );

        const reads = element( 'ul', 'made-list' );

        if ( creation.read_before.length === 0 )
            reads.append( element( 'li', 'empty', 'Nothing' ) );

        for ( const read of creation.read_before )
        {
            const line = element( 'li' );
            line.append( element( 'span', null, read.what ), element( 'span', 'made-when', when( read.time, now ) ) );
            reads.append( line );
        }

        origin.append( reads );

        const earlier = creation.read_before_total - creation.read_before.length;

        if ( earlier > 0 )
            origin.append( element( 'p', 'made-note', earlier + ( earlier === 1 ? ' read before these' : ' reads before these' ) ) );

        if ( creation.exposed )
            origin.append( element( 'p', 'made-note', 'You did not write what Mila read here, and it may have shaped what Mila wrote.' ) );

        if ( creation.kind !== 'file' )
            return origin;

        origin.append( element( 'h3', 'made-heading', creation.writes.length === 1 ? 'Written once' : 'Every write' ) );

        const writes = element( 'ul', 'made-list' );

        for ( const write of creation.writes )
        {
            const line = element( 'li' );
            const what = ( write.tool === 'edit_file' ? 'Changed' : 'Written' ) + ( write.step ? ' · ' + write.step : '' );

            line.append( element( 'span', null, what ), element( 'span', 'made-when', when( write.time, now ) ) );
            writes.append( line );
        }

        origin.append( writes );

        return origin;
    }

    function draw( state )
    {
        const creation = state.creation;

        clear( dialog );

        if ( creation.problem !== null )
        {
            dialog.append( headOf( 'Not found', null ), element( 'p', 'made-note', creation.problem ) );

            return;
        }

        if ( creation.loaded === null )
        {
            const request = creation.request;
            const title = typeof request.path === 'string' ? folderName( request.path ) : 'A reply you kept';

            dialog.append( headOf( title, null ), element( 'p', 'made-note', 'Reading…' ) );

            return;
        }

        const now = new Date();
        const loaded = creation.loaded;
        const made = loaded.creation;
        const title = made.kind === 'file' ? folderName( made.path ) : 'A reply you kept';
        const sub = made.kind === 'file' ? made.location ?? made.path : null;

        dialog.append( headOf( title, sub ) );

        const disk = made.kind === 'file' ? diskWords[ loaded.disk ] ?? null : null;

        if ( disk !== null )
            dialog.append( element( 'p', 'made-disk', disk ) );

        dialog.append( contentsOf( loaded, made ), originOf( state, made, now ) );
    }

    function render( state )
    {
        if ( state.creation === null )
        {
            if ( dialog.open )
                dialog.close();

            return;
        }

        if ( shown === state.creation )
            return;

        shown = state.creation;
        draw( state );

        if ( !dialog.open )
            dialog.showModal();
    }

    return { render, isOpen: () => dialog.open };
}
