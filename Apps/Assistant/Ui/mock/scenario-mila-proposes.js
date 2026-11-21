// Scenario 19: Mila proposes structure (AssistantInterface.md 6). Scenario 16's workspace. Asked to load CSV files into
// DuckDB, Mila sees that this chat and the earlier comparison of SQLite and DuckDB are one piece of work and offers a
// project gathering both; *yes* makes it, *not now* declines it, and anything else leaves it waiting on the overview's
// card, where its name can be changed before it is made. Asked to file this chat with the calculator, Mila offers the
// move, since the person has not yet said yes to one into that project.

import { projectsWorkspace } from './scenario-projects.js';
import { streamSteps, tokenMilliseconds, tokensPerSecond } from './stream.js';

const toolCallTokens = 18;

const prompt = 'Help me load my CSV files into DuckDB';

function routesFor( context )
{
    return [
        {
            heard: /\bquery\b|\bhow\b/i,
            thought: 'A question about DuckDB itself; nothing to open or offer.',
            answer: 'Query them as one table: `SELECT region, sum(total) FROM read_csv(\'data/*.csv\') GROUP BY region`.'
        },
        {
            heard: /\bcalculator\b|\bbelongs?\b|\bmove\b/i,
            thought: 'They say this chat belongs with the calculator project, so I should file it there.',
            name: 'move_to_project',
            summary: 'Moving this chat to Calculator app',
            arguments: { project: 'Calculator app' },
            offer: { kind: 'move', state: 'open', project: 'project-mock-calculator', name: 'Calculator app', in_window: false, said: false },
            result: 'Offered to move this chat into the project Calculator app. The user answers in their next message; do not '
                + 'move it yourself or offer it again. In a sentence, say what you suggested.',
            answer: 'Shall I move this chat into **Calculator app**?'
        },
        {
            heard: /./,
            thought: 'Loading CSV into DuckDB follows on from their comparison of SQLite and DuckDB. Those two chats are one '
                + 'piece of work, so a project would keep them together.',
            name: 'propose_project',
            summary: 'Suggesting a project: CSV analytics',
            arguments: { name: 'CSV analytics', purpose: 'Running analytics over a few GB of CSV files with DuckDB.',
                chats: 'Compare SQLite and DuckDB for analytics' },
            offer: {
                kind: 'project',
                state: 'open',
                name: 'CSV analytics',
                purpose: 'Running analytics over a few GB of CSV files with DuckDB.',
                chats: [ { id: context.conversationId, title: context.title },
                    { id: 'conversation-7', title: 'Compare SQLite and DuckDB for analytics' } ],
                in_window: false
            },
            result: 'Offered the user a project called CSV analytics that gathers this chat and 1 other: Compare SQLite and DuckDB '
                + 'for analytics. The user answers in their next message. Do not make the project yourself or offer it again; '
                + 'in a sentence, say what you suggested.',
            answer: 'DuckDB reads CSV directly: `SELECT * FROM read_csv(\'data/*.csv\')`. This follows on from your comparison of '
                + 'SQLite and DuckDB, so shall I gather both chats into a project called **CSV analytics**?'
        }
    ];
}

function createReply( request, context )
{
    const route = routesFor( context ).find( ( candidate ) => candidate.heard.test( request.text ) );

    // As the session numbers them: a chat's call ids are its own, since an offer is answered by its call's id.
    const callId = 'call-' + context.turnNumber;
    const firstThoughtSteps = streamSteps( route.thought, 1, 'thought' );

    if ( route.name === undefined )
    {
        const steps = streamSteps( route.answer, 1, 'answer' );
        const end = { type: 'turn.round_end', round: 1, prefill_ms: 180, tokens_per_second: tokensPerSecond(),
            tokens: firstThoughtSteps.length + steps.length, thought_tokens: firstThoughtSteps.length, finish: 'stop' };

        return [ { delay: 180, message: { type: 'turn.begin' } }, ...firstThoughtSteps, ...steps, { delay: tokenMilliseconds, message: end },
            { delay: 0, message: { type: 'turn.end', finish: 'stop', answer: route.answer } } ];
    }

    const answerSteps = streamSteps( route.answer, 2, 'answer' );

    const toolCall = { type: 'turn.tool_call', round: 1, call_id: callId, name: route.name, summary: route.summary,
        arguments: JSON.stringify( route.arguments ) };

    const firstRoundEnd = { type: 'turn.round_end', round: 1, prefill_ms: 180, tokens_per_second: tokensPerSecond(),
        tokens: firstThoughtSteps.length + toolCallTokens, thought_tokens: firstThoughtSteps.length, finish: 'tool_call' };

    const secondRoundEnd = { type: 'turn.round_end', round: 2, prefill_ms: 40, tokens_per_second: tokensPerSecond(),
        tokens: answerSteps.length, thought_tokens: 0, finish: 'stop' };

    // An offer is never an approval: the call runs unasked and the person answers it after the reply (5.4).
    return [
        { delay: 180, message: { type: 'turn.begin' } },
        ...firstThoughtSteps,
        { delay: toolCallTokens * tokenMilliseconds, message: toolCall },
        { delay: 0, message: firstRoundEnd },
        { delay: 30, message: { type: 'turn.offer', call_id: callId, offer: route.offer } },
        { delay: 40, message: { type: 'turn.tool_result', call_id: callId, result: route.result, is_error: false } },
        ...answerSteps,
        { delay: tokenMilliseconds, message: secondRoundEnd },
        { delay: 0, message: { type: 'turn.end', finish: 'stop', answer: route.answer } }
    ];
}

export const milaProposes = {
    ...projectsWorkspace,
    number: 19,
    title: 'Mila proposes a project',
    prompt,
    createReply
};
