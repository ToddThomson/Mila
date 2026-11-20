// Scenario 18: going somewhere, and finding things, by asking (AssistantInterface.md 5.2). Scenario 16's workspace.
// Asked to open the calculator project, Mila calls open_place and the pane goes there; asked for all projects, it
// opens them and says which waits on the person; asked what an earlier chat said, it calls find_records and answers
// from the passage it found, naming the chat.

import { projectsWorkspace } from './scenario-projects.js';
import { streamSteps, tokenMilliseconds, tokensPerSecond } from './stream.js';

const toolCallTokens = 14;
const callId = 'call-1';

const prompt = 'Open the calculator project';

const routes = [
    {
        heard: /\bfind\b|\bsaid\b|\bdecide|\bremember|\bearlier\b/i,
        thought: 'They are asking what an earlier chat said about Fushimi Inari, so I should look through their chats.',
        name: 'find_records',
        summary: 'Looking through your chats for Fushimi Inari',
        arguments: { words: 'Fushimi Inari' },
        went: null,
        result: '1 message in the user\'s other chats holding fushimi inari, the most recently changed chat first:\n'
            + '1. In the chat "Plan a three-day trip to Kyoto", in the project Kyoto in November, Mila replied: '
            + '"...**Day 3: Fushimi Inari.** Start before eight; the upper trail is quiet and the maples are at their peak."',
        answer: 'In **Plan a three-day trip to Kyoto** I put Fushimi Inari on day 3, starting before eight, when the upper '
            + 'trail is quiet. No other chat mentions it.'
    },
    {
        heard: /\bprojects\b|\bwaiting\b/i,
        thought: 'They want to see their projects. Opening All projects will show them, and the result says which waits.',
        name: 'open_place',
        summary: 'Going to projects',
        arguments: { place: 'projects' },
        went: { place: { view: 'projects', id: null }, text: 'Opened All projects' },
        result: 'Opened All projects: 3 projects.\n- Calculator app: waiting on the user, 1 chat\n'
            + '- Kyoto in November: nothing waiting, 1 chat\n- Local LLMs on 16 GB: nothing waiting, 1 chat',
        answer: 'Here are your three projects. **Calculator app** is waiting on you: its plan stopped partway.'
    },
    {
        heard: /./,
        thought: 'They want the calculator project opened.',
        name: 'open_place',
        summary: 'Going to calculator',
        arguments: { place: 'calculator' },
        went: { place: { view: 'project', id: 'project-mock-calculator' }, text: 'Opened Calculator app' },
        result: 'Opened the project Calculator app (A small calculator with a parser and its tests.): 1 chat, waiting on the user.',
        answer: 'Opened **Calculator app**. Its plan for tidying the tests stopped partway and is waiting on you.'
    }
];

function createReply( request )
{
    const route = routes.find( ( candidate ) => candidate.heard.test( request.text ) );
    const firstThoughtSteps = streamSteps( route.thought, 1, 'thought' );
    const answerSteps = streamSteps( route.answer, 2, 'answer' );

    const toolCall = { type: 'turn.tool_call', round: 1, call_id: callId, name: route.name, summary: route.summary,
        arguments: JSON.stringify( route.arguments ) };

    const firstRoundEnd = { type: 'turn.round_end', round: 1, prefill_ms: 180, tokens_per_second: tokensPerSecond(),
        tokens: firstThoughtSteps.length + toolCallTokens, thought_tokens: firstThoughtSteps.length, finish: 'tool_call' };

    const secondRoundEnd = { type: 'turn.round_end', round: 2, prefill_ms: 40, tokens_per_second: tokensPerSecond(),
        tokens: answerSteps.length, thought_tokens: 0, finish: 'stop' };

    // In a chat not exposed both run unasked (5.6), so nothing waits for the person.
    const went = route.went === null ? [] : [ { delay: 30, message: { type: 'turn.went', call_id: callId, ...route.went } } ];

    return [
        { delay: 180, message: { type: 'turn.begin' } },
        ...firstThoughtSteps,
        { delay: toolCallTokens * tokenMilliseconds, message: toolCall },
        { delay: 0, message: firstRoundEnd },
        ...went,
        { delay: 40, message: { type: 'turn.tool_result', call_id: callId, result: route.result, is_error: false } },
        ...answerSteps,
        { delay: tokenMilliseconds, message: secondRoundEnd },
        { delay: 0, message: { type: 'turn.end', finish: 'stop', answer: route.answer } }
    ];
}

export const goingSomewhere = {
    ...projectsWorkspace,
    number: 18,
    title: 'Going somewhere by asking',
    prompt,
    createReply
};
