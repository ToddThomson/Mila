// The mock's scripted scenarios, by the numbers MilaAssistant.md section 5 gives them.

import { streamedReply } from './scenario-streamed-reply.js';
import { imageAttachment } from './scenario-image-attachment.js';
import { toolApproval } from './scenario-tool-approval.js';
import { noModelLoaded } from './scenario-no-model.js';
import { firstRun, firstRunOffline, licenseNotAccepted } from './scenario-first-run.js';
import { folderRequest } from './scenario-folder-request.js';
import { exposedChat } from './scenario-exposed-chat.js';
import { projectsWorkspace } from './scenario-projects.js';
import { madeWorkspace } from './scenario-made.js';
import { goingSomewhere } from './scenario-going-somewhere.js';
import { milaProposes } from './scenario-mila-proposes.js';
import { whatMilaUnderstands } from './scenario-understanding.js';
import { onboardingFirstRun, onboardingReturning, onboardingDownload } from './scenario-onboarding.js';

export const scenarios = [
    streamedReply,
    noModelLoaded,
    imageAttachment,
    firstRun,
    firstRunOffline,
    toolApproval,
    licenseNotAccepted,
    folderRequest,
    exposedChat,
    projectsWorkspace,
    madeWorkspace,
    goingSomewhere,
    milaProposes,
    whatMilaUnderstands,
    onboardingFirstRun,
    onboardingReturning,
    onboardingDownload
];

export function findScenario( number )
{
    const wanted = Number( number );
    const found = scenarios.find( ( scenario ) => scenario.number === wanted );

    return found === undefined ? scenarios[ 0 ] : found;
}
