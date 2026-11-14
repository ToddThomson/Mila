// Scenario 8: a first run, nothing installed. The published list planned for this computer, the suggestion
// installed with its progress and loaded, and a remove. Turns, once a model is loaded, are scenario 1's.
// Scenario 12: the same first run with the publisher unreachable.

import { streamedReply } from './scenario-streamed-reply.js';

export const firstRun = {
    ...streamedReply,
    number: 8,
    title: 'First run: the suggestion installed and loaded',
    model: null,
    installed: []
};

export const firstRunOffline = {
    ...firstRun,
    number: 12,
    title: 'First run: the publisher unreachable',
    publisher: 'unreachable'
};
