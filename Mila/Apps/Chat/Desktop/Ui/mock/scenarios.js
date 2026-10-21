// The mock's scripted scenarios, by the numbers ChatDesktop.md section 5 gives them.

import { streamedReply } from './scenario-streamed-reply.js';
import { imageAttachment } from './scenario-image-attachment.js';

export const scenarios = [
    streamedReply,
    imageAttachment
];

export function findScenario( number )
{
    const wanted = Number( number );
    const found = scenarios.find( ( scenario ) => scenario.number === wanted );

    return found === undefined ? scenarios[ 0 ] : found;
}
