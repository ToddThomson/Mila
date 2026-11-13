// Scenario 5: models installed and none loaded. A load shows its progress; Qwen 3.8 27B is refused with the
// refusal's text, and nothing is left loaded. Turns, once a model is loaded, are scenario 1's.

import { streamedReply } from './scenario-streamed-reply.js';

const refusals = {
    'qwen3.8-27b-fp4': 'it needs 17.9 GB at the shortest context it runs at, and this card has 15.6 GB free.'
};

export const noModelLoaded = {
    ...streamedReply,
    number: 5,
    title: 'No model loaded, then a load and a refusal',
    model: null,
    refusals
};
