// What every scripted reply is built from: text cut into token-sized pieces, one step per piece at
// a decode rate plausible for a 12B model on a 12 GB card.

import { splitIntoTokens } from './fixtures.js';

export const tokenMilliseconds = 26;

export function streamSteps( text, round, channel )
{
    const pieces = splitIntoTokens( text );

    return pieces.map( ( piece ) => ( { delay: tokenMilliseconds, message: { type: 'turn.text', round, channel, text: piece } } ) );
}

export function tokensPerSecond()
{
    return Math.round( 10000 / tokenMilliseconds ) / 10;
}

/**
 * A reply of one round: thinking, then the answer, streamed.
 */
export function singleRoundReply( thought, answer, prefillMs )
{
    const thoughtSteps = streamSteps( thought, 1, 'thought' );
    const answerSteps = streamSteps( answer, 1, 'answer' );
    const rate = tokensPerSecond();

    const roundEnd = {
        type: 'turn.round_end',
        round: 1,
        prefill_ms: prefillMs,
        tokens_per_second: rate,
        tokens: thoughtSteps.length + answerSteps.length,
        thought_tokens: thoughtSteps.length,
        finish: 'stop'
    };

    const end = { type: 'turn.end', finish: 'stop', answer };

    return [
        { delay: prefillMs, message: { type: 'turn.begin' } },
        ...thoughtSteps,
        ...answerSteps,
        { delay: tokenMilliseconds, message: roundEnd },
        { delay: 0, message: end }
    ];
}
