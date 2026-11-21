// Scenario 8: a first run, nothing installed. The welcome, the licence accepted, the published list planned for
// this computer, the suggestion installed with its progress and loaded, and the conclusion. Turns, once a model is
// loaded, are scenario 1's.
// Scenario 12: the same first run with the publisher unreachable.
// Scenario 13: models installed and the licence not yet accepted: the welcome and the licence, then the remembered
// model opens.

import { streamedReply } from './scenario-streamed-reply.js';

export const firstRun = {
    ...streamedReply,
    number: 8,
    title: 'First run: licence, the suggestion installed and loaded',
    model: null,
    installed: [],
    licenseAccepted: false
};

export const firstRunOffline = {
    ...firstRun,
    number: 12,
    title: 'First run: the publisher unreachable',
    publisher: 'unreachable'
};

export const licenseNotAccepted = {
    ...streamedReply,
    number: 13,
    title: 'Models installed, the licence not yet accepted',
    licenseAccepted: false
};
