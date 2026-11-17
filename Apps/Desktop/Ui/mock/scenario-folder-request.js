// Scenario 14: Mila asks for a folder (AssistantInterface.md 5.5). The person chooses one in the picker, and the
// reply goes on in it; or declines, and Mila goes on without files.

import { streamSteps, tokenMilliseconds, tokensPerSecond } from './stream.js';

const toolCallTokens = 18;
const callId = 'call-1';

const prompt = 'Can you tidy up the README in my calculator project?';

const why = 'To read and edit the README in your calculator project';

const firstThought = 'The README is a file on their computer, and this chat has no folder yet, so I need to ask for one.';

const allowedThought = 'They gave me the calc folder. I would read README.md next and propose the edits.';

const allowedAnswer = 'Thanks, I have the **calc** folder now. I\'ll read `README.md` and suggest how to tidy it before changing anything.';

const deniedThought = 'No folder was given, so I cannot read the README. I should say so and offer another way.';

const deniedAnswer = 'I can\'t see your files without a folder. If you paste the README here, I\'ll tidy it up and give it back to you.';

const declined = 'The person declined to run this tool.';

function secondRound( thought, answer, toolResult )
{
    const thoughtSteps = streamSteps( thought, 2, 'thought' );
    const answerSteps = streamSteps( answer, 2, 'answer' );

    const roundEnd = {
        type: 'turn.round_end',
        round: 2,
        prefill_ms: 44,
        tokens_per_second: tokensPerSecond(),
        tokens: thoughtSteps.length + answerSteps.length,
        thought_tokens: thoughtSteps.length,
        finish: 'stop'
    };

    const end = { type: 'turn.end', finish: 'stop', answer };

    return [
        { delay: 60, message: toolResult },
        { delay: 44, message: { type: 'turn.thought_break', round: 2 } },
        ...thoughtSteps,
        ...answerSteps,
        { delay: tokenMilliseconds, message: roundEnd },
        { delay: 0, message: end }
    ];
}

function createReply()
{
    const firstThoughtSteps = streamSteps( firstThought, 1, 'thought' );

    const toolCall = {
        type: 'turn.tool_call',
        round: 1,
        call_id: callId,
        name: 'request_folder',
        summary: 'Asking for a folder: ' + why,
        arguments: JSON.stringify( { why } )
    };

    const firstRoundEnd = {
        type: 'turn.round_end',
        round: 1,
        prefill_ms: 205,
        tokens_per_second: tokensPerSecond(),
        tokens: firstThoughtSteps.length + toolCallTokens,
        thought_tokens: firstThoughtSteps.length,
        finish: 'tool_call'
    };

    // A folder is chosen each time it is asked for, so nothing covers the conversation.
    const approval = { type: 'turn.tool_approval', call_id: callId, choices: [ 'allow_once', 'deny' ],
        reaches: { folders: [], program_folders: [], network: false, runs_programs: false } };
    const given = { type: 'turn.tool_result', call_id: callId, is_error: false,
        result: 'The user gave you the folder C:\\Users\\you\\Projects\\calc. Paths are relative to it, as list_folder shows them.' };
    const refusal = { type: 'turn.tool_result', call_id: callId, result: declined, is_error: true };

    const branches = {
        allow: secondRound( allowedThought, allowedAnswer, given ),
        deny: secondRound( deniedThought, deniedAnswer, refusal )
    };

    return [
        { delay: 205, message: { type: 'turn.begin' } },
        ...firstThoughtSteps,
        { delay: toolCallTokens * tokenMilliseconds, message: toolCall },
        { delay: 0, message: firstRoundEnd },
        { delay: 0, message: approval, tool: toolCall.name, branches }
    ];
}

export const folderRequest = {
    number: 14,
    title: 'Mila asks for a folder',
    model: 'gemma-4-12b-it-fp4',
    prompt,
    createReply
};
