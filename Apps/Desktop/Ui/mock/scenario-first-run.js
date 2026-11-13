// Scenario 8: a first run, nothing installed. The published list, an install with its progress, a load, and a
// remove. Turns, once a model is loaded, are scenario 1's.

import { streamedReply } from './scenario-streamed-reply.js';

export const firstRun = {
    ...streamedReply,
    number: 8,
    title: 'First run: install, load, remove',
    model: null,
    installed: []
};
