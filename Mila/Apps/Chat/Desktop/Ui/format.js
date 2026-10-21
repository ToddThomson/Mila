// The words and numbers the page shows for session facts. No DOM, so they are tested with the
// reducer. Wording follows the console's (Chat.ixx, printSessionStatus and showTurnStats).

const numberFormat = new Intl.NumberFormat( 'en-US' );

export function formatCount( value )
{
    return numberFormat.format( value );
}

export function formatBytes( bytes )
{
    const gigabytes = bytes / 1e9;

    if ( gigabytes >= 1 )
        return gigabytes.toFixed( 1 ) + ' GB';

    const megabytes = bytes / 1e6;

    if ( megabytes >= 1 )
        return Math.round( megabytes ) + ' MB';

    const rounded = Math.round( bytes / 1e3 );
    const kilobytes = Math.max( 1, rounded );

    return kilobytes + ' KB';
}

export function contextFact( session )
{
    const length = formatCount( session.context_length );
    const automatic = session.context_automatic ? ' (auto)' : '';

    return 'Context window ' + length + automatic;
}

/**
 * The thinking clause is left out for a model without the channel: "off" would read as a
 * setting the user could turn on.
 */
export function thinkingFact( session )
{
    if ( !session.thinking_capable )
        return null;

    return 'Thinking ' + session.thinking;
}

export function thoughtLabel( block, streaming )
{
    if ( block.tokens === null )
        return streaming ? 'Thinking' : 'Thought';

    const count = formatCount( block.tokens );
    const unit = block.tokens === 1 ? ' token' : ' tokens';

    return 'Thought for ' + count + unit;
}

export function statsLine( summary )
{
    if ( summary === null )
        return '';

    const parts = [];
    const prefill = 'Prefill ' + Math.round( summary.prefillMs ) + ' ms';
    parts.push( prefill );

    if ( summary.tokensPerSecond > 0 )
    {
        const rate = summary.tokensPerSecond.toFixed( 1 ) + ' tok/s';
        parts.push( rate );
    }

    const tokens = formatCount( summary.tokens ) + ( summary.tokens === 1 ? ' token' : ' tokens' );
    parts.push( tokens );

    const rounds = summary.rounds + ( summary.rounds === 1 ? ' round' : ' rounds' );
    parts.push( rounds );

    return parts.join( ' · ' );
}

/**
 * What a reply that did not finish on its own says about it, in the reply itself.
 */
export function finishNotice( finish )
{
    switch ( finish )
    {
        case 'length':
            return 'The reply reached its length limit before it finished.';

        case 'context':
            return 'The reply stopped at the end of the context window.';

        case 'cancelled':
            return 'Stopped.';

        case 'error':
            return 'The reply ended with an error.';

        default:
            return null;
    }
}

/**
 * A tool's arguments or result for reading: indented when it is JSON, as written when not.
 */
export function readableJson( text )
{
    if ( text === null || text === undefined )
        return '';

    try
    {
        const parsed = JSON.parse( text );

        return JSON.stringify( parsed, null, 2 );
    }
    catch
    {
        return text;
    }
}
