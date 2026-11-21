// The mock's offers, as the session keeps them (AssistantInterface.md 5.4 and 6): an offer on the call that made it,
// the person's next message read as an answer to it by the same rules as Desktop.OfferAnswer, and every offer a chat
// holds, for its list item.

const acceptances = [ 'yes', 'y', 'yeah', 'yep', 'yup', 'ok', 'okay', 'sure', 'please', 'please do', 'do it', 'go ahead',
    'yes please', 'yes do it', 'ok do it', 'okay do it', 'sounds good', 'make it', 'alright', 'all right', 'good idea', 'great',
    'perfect', 'fine' ];

const refusals = [ 'no', 'n', 'nope', 'nah', 'no thanks', 'no thank you', 'not now', 'not yet', 'maybe later', 'later', 'dont',
    'no dont', 'leave it', 'no leave it', 'not this time', 'no not now' ];

const keepSeparate = [ 'keep it separate', 'keep separate', 'keep this separate', 'keep it apart', 'keep this chat separate', 'undo',
    'undo it', 'undo that', 'move it back', 'put it back' ];

/** The words alone, case folded and one space apart; an apostrophe joins its word. */
function normalized( text )
{
    return text.replace( /['’]/g, '' ).toLowerCase().split( /[^a-z0-9\u0080-￿]+/ ).filter( ( word ) => word.length > 0 ).join( ' ' );
}

/** `accept`, `decline`, `keep_separate` or `other`. */
export function readOfferAnswer( text )
{
    const words = normalized( typeof text === 'string' ? text : '' );

    if ( acceptances.includes( words ) )
        return 'accept';

    if ( refusals.includes( words ) )
        return 'decline';

    return keepSeparate.includes( words ) ? 'keep_separate' : 'other';
}

/** The call whose offer the next message answers: one open in Mila's last reply, or a move made and said there. */
export function pendingOffer( messages )
{
    for ( let index = messages.length - 1; index >= 0; index = index - 1 )
    {
        const message = messages[ index ];

        if ( message.role !== 'assistant' )
            continue;

        const calls = Array.isArray( message.tool_calls ) ? message.tool_calls : [];

        for ( let call = calls.length - 1; call >= 0; call = call - 1 )
        {
            const offer = calls[ call ].offer ?? null;

            if ( offer !== null && ( offer.state === 'open' || ( offer.kind === 'move' && offer.state === 'accepted' && offer.said === true ) ) )
                return calls[ call ];
        }

        return null;
    }

    return null;
}

/** The call @p callId in a chat's messages, when it holds an offer. */
export function callWithOffer( messages, callId )
{
    for ( const message of messages )
    {
        for ( const call of Array.isArray( message.tool_calls ) ? message.tool_calls : [] )
        {
            if ( call.call_id === callId && call.offer )
                return call;
        }
    }

    return null;
}

/** Every offer in a chat, with the message and call it is kept on, oldest first. */
export function offersIn( messages )
{
    const offers = [];

    for ( const [ index, message ] of messages.entries() )
    {
        for ( const call of Array.isArray( message.tool_calls ) ? message.tool_calls : [] )
        {
            if ( call.offer )
                offers.push( { ...call.offer, message: index, call_id: call.call_id } );
        }
    }

    return offers;
}
