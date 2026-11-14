// The mock's scripted scenarios, by the numbers MilaDesktop.md section 5 gives them.

import { streamedReply } from './scenario-streamed-reply.js';
import { imageAttachment } from './scenario-image-attachment.js';
import { toolApproval } from './scenario-tool-approval.js';
import { noModelLoaded } from './scenario-no-model.js';
import { firstRun, firstRunOffline } from './scenario-first-run.js';

export const scenarios = [
    streamedReply,
    noModelLoaded,
    imageAttachment,
    firstRun,
    firstRunOffline,
    toolApproval
];

export function findScenario( number )
{
    const wanted = Number( number );
    const found = scenarios.find( ( scenario ) => scenario.number === wanted );

    return found === undefined ? scenarios[ 0 ] : found;
}
