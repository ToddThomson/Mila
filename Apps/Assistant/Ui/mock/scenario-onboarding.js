// Scenario 21: a first run with the onboarding model (AssistantOnboarding.md 4.4). Its download begins at launch,
// while the welcome and the Terms and Conditions show; once they are accepted and the model is loaded, the workspace
// opens on the first chat, where Mila begins the interview. The interview is scripted: each reply is the next of its
// steps, whatever the person says.

import { streamSteps, tokensPerSecond } from './stream.js';

const opener = 'Hello, I\'m Mila, and I run here, on this computer.\n\n'
    + 'Before we start, I\'d like to set myself up for you. A few questions, and you can skip any of them.\n\n'
    + 'First: what should I call you?';

const replies = [
    {
        answer: 'Thank you. What do you expect to use me for, mostly? For example writing, working through documents, '
            + 'planning a trip, keeping track of a project, or code.',
        calls: [ { name: 'set_person_name', summary: 'Remembering what to call you', arguments: { name: 'from your reply' } } ]
    },
    {
        answer: 'That helps. Are there folders on this computer you would like me to be able to read, such as Documents? '
            + 'I will only read from the folders you name, and I ask before I change anything.',
        calls: [ { name: 'note_interests', summary: 'Noting what you will use Mila for', arguments: { interests: 'from your reply' } } ]
    },
    {
        answer: 'You\'re set up. I can read the folders you named, and I will ask before writing to any of them.\n\n'
            + 'The model I\'m running on suits this computer and what you have described, so there is nothing more to '
            + 'install. You can change any of this later: just ask me.\n\nWhat would you like to start with?',
        calls: [ { name: 'allow_folders', summary: 'Letting Mila read the folders you named', arguments: { folders: 'from your reply' } } ]
    }
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

function createReply( request, context )
{
    // The opener was turn 1, so the person's first answer is turn 2.
    const index = Math.min( Math.max( context.turnNumber - 2, 0 ), replies.length - 1 );
    const step = replies[ index ];
    const steps = [ { delay: 200, message: { type: 'turn.begin' } } ];

    step.calls.forEach( ( call, position ) =>
    {
        const callId = 'call-setup-' + context.turnNumber + '-' + position;
        const argumentsJson = JSON.stringify( call.arguments );
        steps.push( { delay: 300, message: { type: 'turn.tool_call', round: 1, call_id: callId, name: call.name, summary: call.summary, arguments: argumentsJson } } );
        steps.push( { delay: 0, message: roundEnd( 1, 12, 'tool_call' ) } );
        steps.push( { delay: 150, message: { type: 'turn.tool_result', call_id: callId, result: '{"ok": true}', is_error: false } } );
    } );

    steps.push( ...answerSteps( step.answer, 2 ) );

    return steps;
}

export const onboardingFirstRun = {
    number: 21,
    title: 'First run with the onboarding model: download at launch, terms, the interview',
    model: null,
    installed: [],
    licenseAccepted: false,
    onboarding: 'gemma-4-12b-it-q4_0',
    // Slow enough to see the download run behind the welcome and the terms: about 20 seconds.
    installStepMilliseconds: 500,
    prompt: 'Alex',
    createInterview,
    createReply
};
