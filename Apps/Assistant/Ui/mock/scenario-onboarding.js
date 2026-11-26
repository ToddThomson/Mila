// Scenario 21: a first run with the onboarding model (AssistantOnboarding.md 4.4). Its download begins at launch,
// while the welcome and the Terms and Conditions show; once they are accepted and the model is loaded, the workspace
// opens on the first chat, where Mila begins the interview. The interview is scripted: each reply is the next of its
// steps, taking the person's words as their answer. Then Mila assesses what it needs (round 12): in scenario 21, nothing,
// so the setup ends with no question; in scenario 23, a download for coding work, which the person answers in words or
// with its buttons. The setup's chat then closes.

import { streamSteps, tokensPerSecond } from './stream.js';

const opener = 'Hello, I\'m Mila, and I run here, on this computer.\n\n'
    + 'Before we start, I\'d like to set myself up for you. A few questions, and you can skip any of them.\n\n'
    + 'First: what should I call you?';

const capabilities = 'I can answer questions, plan a piece of work and carry it out once you approve it, keep your chats '
    + 'about one piece of work together in a project, find what was said in earlier chats, read web pages you point me '
    + 'to, and work with the files in a folder you choose, asking you first.';

// The download the session's assess_needs would ask for on the 16 GB card for coding work: Gemma 4 26B-A4B Q4_0, measured
// agentic there with the FP8 cache. Its name goes to the session alone; the person reads what it gives and its size.
const codingDownload = { kind: 'download', state: 'open', in_window: false,
    download: { name: 'gemma-4-26b-a4b-it-q4_0', size_bytes: 14420000000, gains: [ 'work of several steps with your files and tools' ] } };

/** The person's answers, as the scripted model reads them: their words, whatever they are, and then Mila's assessment. */
function repliesFor( download )
{
    const assessed = download === undefined
        ? { answer: 'Thank you, that helps. You\'re set up: start a new chat whenever you\'re ready.', needs: [ 'conversation' ],
            result: 'Mila has what this work needs on this computer, so nothing is downloaded. The setup has ended.' }
        : { answer: 'Thank you, that helps. To be better at work of several steps with your files and tools, I need to '
            + 'download about 14.4 GB. It downloads while you go on. Shall I download it?', needs: [ 'files_and_tools' ],
            result: 'To do more of this work, Mila needs a download.' };

    return [
        ( said ) => ( {
            answer: 'It\'s nice to meet you, ' + said + '!\n\n' + capabilities + '\n\nWhat do you expect to use me for most?',
            calls: [ { name: 'remember_name', summary: 'Remembering what to call you', arguments: { name: said },
                result: 'Saved. Mila will call the user ' + said + '.' } ]
        } ),
        ( said ) => ( {
            answer: assessed.answer,
            calls: [
                { name: 'note_uses', summary: 'Noting what you will use Mila for', arguments: { uses: said, needs: assessed.needs },
                    result: 'Kept. Both questions are answered or skipped: call assess_needs.' },
                { name: 'assess_needs', summary: 'Working out what Mila needs for your work', arguments: {}, offer: download,
                    result: assessed.result } ]
        } )
    ];
}

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

/** The interview's replies, with Mila's assessment asking for this download, or for nothing when there is none. */
function replyFor( download )
{
    const replies = repliesFor( download );

    return ( request, context ) => scriptedReply( replies, request, context );
}

function scriptedReply( replies, request, context )
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

        // An offer is kept on the call that made it, as the session sends it while the call runs.
        if ( call.offer !== undefined )
            steps.push( { delay: 0, message: { type: 'turn.offer', call_id: callId, offer: { ...call.offer } } } );

        steps.push( { delay: 0, message: roundEnd( 1, 12, 'tool_call' ) } );
        steps.push( { delay: 150, message: { type: 'turn.tool_result', call_id: callId, result: call.result, is_error: false } } );
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
    createReply: replyFor( undefined )
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

// Scenario 23: the interview of scenario 22 for a person whose work is coding, on the 16 GB card: Mila assesses that it
// needs a download for it and asks, no model named, with Download and Not now (AssistantOnboarding.md 4.4, round 12).
export const onboardingDownload = {
    ...onboardingReturning,
    number: 23,
    title: 'The interview for coding work: Mila asks to download what it needs',
    prompt: 'Alex',
    downloadOffer: codingDownload,
    createReply: replyFor( codingDownload )
};
