// The mock's scripted scenarios, by the numbers MilaApp.md section 5 gives them.

import { streamedReply } from './scenario-streamed-reply.js';
import { imageAttachment } from './scenario-image-attachment.js';
import { toolApproval } from './scenario-tool-approval.js';

export const scenarios = [
    streamedReply,
    imageAttachment,
    toolApproval
];

export function findScenario( number )
{
    const wanted = Number( number );
    const found = scenarios.find( ( scenario ) => scenario.number === wanted );

    return found === undefined ? scenarios[ 0 ] : found;
}
