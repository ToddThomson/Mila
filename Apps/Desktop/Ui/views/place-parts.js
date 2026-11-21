// The pieces every place of the workspace is built from (AssistantInterface.md 4): a head, a section, a chat's row, a
// figure, and a project's dot of its state.

import { element, button } from '../dom.js';
import { whenLabel, plainLine } from '../format.js';

export const stateWords = { waiting: 'Waiting on you', active: 'Active', quiet: 'Quiet' };

export function head( eyebrow, title )
{
    const made = element( 'div', 'place-head' );
    made.append( element( 'div', 'eyebrow', eyebrow ), element( 'h1', 'place-title', title ) );

    return made;
}

export function section( title, identifier, aside )
{
    const made = element( 'section', 'section' );
    const top = element( 'div', 'section-head' );

    if ( identifier )
        made.id = identifier;

    top.append( element( 'h2', 'section-title', title ) );

    if ( aside )
        top.append( aside );

    made.append( top );

    return made;
}

/**
 * A chat as a row: its title, where it is when given, the start of Mila's last reply, and when; marked when Mila sees it
 * as belonging with others (4.3).
 */
export function chatRow( item, now, where, together = false )
{
    const row = button( together ? 'row together' : 'row', null, 'open-conversation' );
    const main = element( 'span', 'row-main' );
    const preview = plainLine( item.reply_start );
    const sub = preview.length > 0 ? preview : 'No reply yet';

    row.dataset.identifier = item.id;
    main.append( element( 'span', 'row-title', item.title ) );
    main.append( element( 'span', 'row-sub', where ? where + ' · ' + sub : sub ) );
    row.append( main, element( 'span', 'row-when', whenLabel( item.updated, now ) ) );

    return row;
}

export function figure( value, label, note, action, modifier )
{
    const made = button( modifier ? 'figure ' + modifier : 'figure', null, action.action );

    Object.assign( made.dataset, action.data );
    made.append( element( 'span', 'figure-value', String( value ) ), element( 'span', 'figure-label', label ),
        element( 'span', 'figure-note', note ) );

    return made;
}

/** A project's state as a dot: amber waiting on the person, teal active, grey quiet (4.1). */
export function stateDot( state )
{
    const dot = element( 'span', 'state-dot ' + state );
    dot.setAttribute( 'aria-hidden', 'true' );

    return dot;
}

/** A project's last fourteen days of work, as a strip of bars. */
export function pulseStrip( pulse )
{
    const strip = element( 'span', 'pulse' );
    const most = Math.max( 1, ...pulse );

    strip.setAttribute( 'aria-hidden', 'true' );

    for ( const count of pulse )
    {
        const bar = element( 'span', count === 0 ? 'pulse-bar none' : 'pulse-bar' );
        bar.style.height = ( count === 0 ? 2 : 4 + Math.round( 12 * count / most ) ) + 'px';
        strip.append( bar );
    }

    return strip;
}

/** How many chats and files: *2 chats · 3 made*. */
export function countsLine( summary )
{
    const chats = summary.chats.length;

    return chats + ( chats === 1 ? ' chat' : ' chats' ) + ' · ' + summary.made + ' made';
}
