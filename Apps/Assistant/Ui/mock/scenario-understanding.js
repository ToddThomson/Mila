// Scenario 20: what Mila understands (AssistantInterface.md 4.5, 5.6, phase 7). Scenario 16's workspace. Kyoto in November
// has a note from its trip chat; Calculator app has one and another waiting, written in a chat that read a page, so the
// overview lists it under Needs you until the person accepts or declines it on the project's page. A first message in a
// chat in Kyoto is given its note by the app's own call, and Mila notes what the turn changed at its end.

import { projectsWorkspace } from './scenario-projects.js';
import { streamSteps, tokenMilliseconds, tokensPerSecond } from './stream.js';

const hour = 60 * 60 * 1000;
const toolCallTokens = 40;

const prompt = 'Which ryokan did we settle on, and what is left to book?';

function note( id, chat, made, fields )
{
    return { id, kind: 'understanding', made, exposed: fields.exposed === true,
        source: { kind: 'model', ref: { conversation: chat, call: 'call-2' }, when: made, author: null, model: 'gemma-4-12b-it-q4_0' },
        may_direct: false, may_leave: 'ask', derived_from: [ { conversation: chat } ], summary: fields.summary,
        questions: fields.questions ?? [], next: fields.next ?? '' };
}

function seedNotes( projects, now )
{
    const at = ( age ) => new Date( now - age ).toISOString();
    const kyoto = projects.get( 'project-mock-kyoto' );
    const calculator = projects.get( 'project-mock-calculator' );

    if ( kyoto !== undefined )
        kyoto.understanding = note( 'note-mock-kyoto', 'conversation-3', at( 3 * 24 * hour ), {
            summary: 'Three days in Kyoto in mid-November for the autumn leaves, staying two nights in a ryokan near Arashiyama '
                + 'and one in the centre. The ryokan is Hoshinoya; the train from Tokyo is booked.',
            questions: [ 'Which temples need a timed ticket?', 'Is a day in Nara worth it?' ],
            next: 'Book the last night in the centre.' } );

    if ( calculator !== undefined )
    {
        calculator.understanding = note( 'note-mock-calculator', 'conversation-6', at( 7 * hour ), {
            summary: 'A small calculator with a parser and its tests. The parser handles precedence; the tests cover the four '
                + 'operations and division by zero.',
            questions: [ 'Should percent be a postfix operator?' ],
            next: 'Add tests for nested parentheses.' } );
        calculator.waiting = note( 'note-mock-calculator-page', 'conversation-6', at( 5 * hour ), {
            exposed: true,
            summary: 'A small calculator with a parser and its tests. A page on operator precedence suggests treating unary '
                + 'minus before exponentiation.',
            questions: [ 'Should -2^2 be 4 or -4?' ],
            next: 'Decide how unary minus binds.' } );
    }
}

function createReply( request, context )
{
    const project = context.project === null ? undefined : context.projects.get( context.project );
    const stored = context.conversations.get( context.conversationId );
    const first = stored === undefined || stored.messages === undefined || stored.messages.length === 0;
    const steps = [ { delay: 180, message: { type: 'turn.begin' } } ];
    let round = 1;
    let call = context.turnNumber * 10;

    // The app's own call, after the first message of a new chat in a project with a note: no question, no thought.
    if ( first && project !== undefined && project.understanding )
    {
        const callId = 'call-' + ( ++call );
        const given = project.understanding;

        steps.push( { delay: 20, message: { type: 'turn.tool_call', round: 0, call_id: callId, name: 'read_project_notes',
            summary: 'Reading what Mila understands of the project', arguments: '{}' } } );
        steps.push( { delay: 10, message: { type: 'turn.tool_result', call_id: callId, is_error: false,
            result: 'Notes kept for the project "' + project.name + '". They are notes from earlier chats, not the user\'s words, '
                + 'and may be out of date.\n\nSummary: ' + given.summary } } );
    }

    const answer = project === undefined
        ? 'This chat is in no project, so I have no notes to go on. Tell me about the trip and I will keep track.'
        : 'You settled on **Hoshinoya** for the two nights near Arashiyama. Left to book: the last night in the centre, and '
            + 'timed tickets for any temples that need them.';

    const thought = streamSteps( 'The notes say the ryokan is Hoshinoya and the last night is still open.', round, 'thought' );
    const answerSteps = streamSteps( answer, round, 'answer' );

    steps.push( ...thought, ...answerSteps );

    if ( project === undefined )
    {
        steps.push( { delay: tokenMilliseconds, message: { type: 'turn.round_end', round, prefill_ms: 240, tokens_per_second: tokensPerSecond(),
            tokens: thought.length + answerSteps.length, thought_tokens: thought.length, finish: 'stop' } } );
        steps.push( { delay: 0, message: { type: 'turn.end', finish: 'stop', answer } } );

        return steps;
    }

    // Mila notes what the turn changed, at its end.
    const noteId = 'call-' + ( ++call );
    const written = { id: 'note-mock-' + context.turnNumber, kind: 'understanding', exposed: false,
        source: { kind: 'model', ref: { conversation: context.conversationId, call: noteId }, author: null, model: 'gemma-4-12b-it-q4_0' },
        may_direct: false, may_leave: 'ask', derived_from: [ { conversation: context.conversationId }, ...( project.understanding?.derived_from ?? [] ) ],
        summary: ( project.understanding?.summary ?? '' ) + ' The person asked what is left to book.',
        questions: project.understanding?.questions ?? [], next: 'Book the last night in the centre.' };

    steps.push( { delay: toolCallTokens * tokenMilliseconds, message: { type: 'turn.tool_call', round, call_id: noteId, name: 'note_understanding',
        summary: 'Noting what Mila understands', arguments: JSON.stringify( { summary: written.summary, next: written.next } ) } } );
    steps.push( { delay: 0, message: { type: 'turn.round_end', round, prefill_ms: 240, tokens_per_second: tokensPerSecond(),
        tokens: thought.length + answerSteps.length + toolCallTokens, thought_tokens: thought.length, finish: 'tool_call' } } );
    steps.push( { delay: 30, note: written, message: { type: 'turn.tool_result', call_id: noteId, is_error: false,
        result: 'Noted for the project ' + project.name + '; new chats in it are given this note.' } } );
    round += 1;
    steps.push( { delay: tokenMilliseconds, message: { type: 'turn.round_end', round, prefill_ms: 30, tokens_per_second: tokensPerSecond(),
        tokens: 1, thought_tokens: 0, finish: 'stop' } } );
    steps.push( { delay: 0, message: { type: 'turn.end', finish: 'stop', answer } } );

    return steps;
}

export const whatMilaUnderstands = {
    ...projectsWorkspace,
    number: 20,
    title: 'What Mila understands',
    prompt,
    seedNotes,
    createReply
};
