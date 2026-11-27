// Scenario 21: a first run (AssistantOnboarding.md 4.4, round 13). The model that suits this computer is picked at
// launch and downloads once the person agrees, while the welcome and the Terms and Conditions show; once they are
// accepted and the model is loaded, the workspace opens on the first chat, where Mila begins the interview. The
// interview is scripted: each reply is the next of its steps, taking the person's words as their answer. Once both
// questions are answered the setup ends, and the chat goes on as the person's first.

import { streamSteps, tokensPerSecond } from './stream.js';

const opener = 'Hello, I\'m Mila, and I run here, on this computer.\n\n'
    + 'Before we start, I\'d like to set myself up for you. A few questions, and you can skip any of them.\n\n'
    + 'First: what should I call you?';

const capabilities = 'I can answer questions, plan a piece of work and carry it out once you approve it, keep your chats '
    + 'about one piece of work together in a project, find what was said in earlier chats, read web pages you point me '
    + 'to, and work with the files in a folder you choose, asking you first.';

/** The person's answers, as the scripted model reads them: their name, what they will use Mila for, then a first request. */
const replies = [
    ( said ) => ( {
        answer: 'It\'s nice to meet you, ' + said + '!\n\n' + capabilities + '\n\nWhat do you expect to use me for most?',
        calls: [ { name: 'remember_name', summary: 'Remembering what to call you', arguments: { name: said },
            result: 'Saved. Mila will call the user ' + said + '. Greet them back by that name. Then go on with the next question.' } ]
    } ),
    ( said ) => ( {
        answer: 'Thank you, that helps: I\'ll keep in mind that you\'ll mostly use me for ' + said + '. What would you like to start with?',
        calls: [ { name: 'note_uses', summary: 'Noting what you will use Mila for', arguments: { uses: said },
            result: 'Kept. Every question is answered or skipped, so the setup has ended, and this chat goes on as any chat does.' } ]
    } ),
    () => ( {
        answer: 'Happy to. Tell me a little more about it, and we\'ll take it a step at a time.',
        calls: []
    } )
];

function roundEnd( round, tokens, finish )
{
    return { type: 'turn.round_end', round, prefill_ms: 120, tokens_per_second: tokensPerSecond(), tokens, thought_tokens: 0, finish };
}

function answerSteps( text, round )
{
    const steps = streamSteps( text, round, 'answer' );

    return [ ...steps, { delay: 0, message: roundEnd( round, steps.length, 'stop' ) }, { delay: 0, message: { type: 'turn.end', finish: 'stop', answer: text } } ];
}

/** Mila's first message, with no message from the person before it. */
function createInterview()
{
    return [ { delay: 400, message: { type: 'turn.begin' } }, ...answerSteps( opener, 1 ) ];
}

function scriptedReply( request, context )
{
    // The opener was turn 1, so the person's first answer is turn 2.
    const index = Math.min( Math.max( context.turnNumber - 2, 0 ), replies.length - 1 );
    const step = replies[ index ]( request.text.trim() );
    const steps = [ { delay: 200, message: { type: 'turn.begin' } } ];

    step.calls.forEach( ( call, position ) =>
    {
        const callId = 'call-setup-' + context.turnNumber + '-' + position;
        const argumentsJson = JSON.stringify( call.arguments );
        steps.push( { delay: 300, message: { type: 'turn.tool_call', round: 1, call_id: callId, name: call.name, summary: call.summary, arguments: argumentsJson } } );
        steps.push( { delay: 0, message: roundEnd( 1, 12, 'tool_call' ) } );
        steps.push( { delay: 150, message: { type: 'turn.tool_result', call_id: callId, result: call.result, is_error: false } } );
    } );

    steps.push( ...answerSteps( step.answer, step.calls.length === 0 ? 1 : 2 ) );

    return steps;
}

export const onboardingFirstRun = {
    number: 21,
    title: 'First run: one download for this computer, terms, the interview, then the first chat',
    model: null,
    installed: [],
    licenseAccepted: false,
    onboarding: 'gemma-4-12b-it-q4_0',
    // Slow enough to see the download run behind the welcome and the terms: about 20 seconds.
    installStepMilliseconds: 500,
    prompt: 'Alex',
    createInterview,
    createReply: scriptedReply
};

// Scenario 22: the same person launching Mila again part way through the interview. The terms are accepted, the model
// is installed and the panes were gone through, so Almost Ready shows while the model loads, then the interview's chat;
// the session reopens the chat it began, where the mock begins it again.
export const onboardingReturning = {
    ...onboardingFirstRun,
    number: 22,
    title: 'Launching again part way through the interview: Almost Ready, then its chat, no panes',
    installed: [ 'gemma-4-12b-it-q4_0' ],
    licenseAccepted: true,
    panesGoneThrough: true
};
