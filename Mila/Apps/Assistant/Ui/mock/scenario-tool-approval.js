// Scenario 11: a get_weather call that waits for the person. Allowed, the reply continues with the
// reading; denied, the model says it could not check and answers without it.

import { streamSteps, tokenMilliseconds, tokensPerSecond } from './stream.js';

const toolCallTokens = 15;
const callId = 'call-1';

const prompt = 'Should I bring an umbrella in Vancouver this afternoon?';

const firstThought = 'This depends on the weather in Vancouver right now, which I cannot know without looking. '
    + 'The get_weather tool takes a location; "Vancouver, Canada" is the one meant.';

const allowedThought = 'The reading is 11 °C with light rain. That settles it: an umbrella, and a layer for the cool.';

const allowedAnswer = 'Yes, bring one. It\'s **11 °C with light rain** in Vancouver right now, '
    + 'so an umbrella and a light jacket will keep you comfortable this afternoon.';

const deniedThought = 'The weather lookup was declined, so I have no reading. I should say that plainly, '
    + 'and give only general guidance without inventing conditions.';

const deniedAnswer = 'I couldn\'t check the current weather, because the lookup wasn\'t allowed. '
    + 'Vancouver afternoons this time of year are often wet, so an umbrella is a safe choice. '
    + 'Allow the weather check if you\'d like me to look at the actual conditions.';

const declined = 'The person declined to run this tool.';

function secondRound( thought, answer, toolResult )
{
    const thoughtSteps = streamSteps( thought, 2, 'thought' );
    const answerSteps = streamSteps( answer, 2, 'answer' );

    const roundEnd = {
        type: 'turn.round_end',
        round: 2,
        prefill_ms: 41,
        tokens_per_second: tokensPerSecond(),
        tokens: thoughtSteps.length + answerSteps.length,
        thought_tokens: thoughtSteps.length,
        finish: 'stop'
    };

    const end = { type: 'turn.end', finish: 'stop', answer };

    return [
        { delay: 140, message: toolResult },
        { delay: 41, message: { type: 'turn.thought_break', round: 2 } },
        ...thoughtSteps,
        ...answerSteps,
        { delay: tokenMilliseconds, message: roundEnd },
        { delay: 0, message: end }
    ];
}

function createReply()
{
    const firstThoughtSteps = streamSteps( firstThought, 1, 'thought' );
    const location = { location: 'Vancouver, Canada' };

    const toolCall = {
        type: 'turn.tool_call',
        round: 1,
        call_id: callId,
        name: 'get_weather',
        summary: 'Getting weather for Vancouver, Canada',
        arguments: JSON.stringify( location )
    };

    const firstRoundEnd = {
        type: 'turn.round_end',
        round: 1,
        prefill_ms: 198,
        tokens_per_second: tokensPerSecond(),
        tokens: firstThoughtSteps.length + toolCallTokens,
        thought_tokens: firstThoughtSteps.length,
        finish: 'tool_call'
    };

    const approval = { type: 'turn.tool_approval', call_id: callId, choices: [ 'allow_once', 'allow_conversation', 'deny' ] };
    const reading = { type: 'turn.tool_result', call_id: callId, result: '{"temperature_c": 11, "condition": "light rain"}', is_error: false };
    const refusal = { type: 'turn.tool_result', call_id: callId, result: declined, is_error: true };

    const branches = {
        allow: secondRound( allowedThought, allowedAnswer, reading ),
        deny: secondRound( deniedThought, deniedAnswer, refusal )
    };

    return [
        { delay: 198, message: { type: 'turn.begin' } },
        ...firstThoughtSteps,
        { delay: toolCallTokens * tokenMilliseconds, message: toolCall },
        { delay: 0, message: firstRoundEnd },
        { delay: 0, message: approval, tool: toolCall.name, branches }
    ];
}

export const toolApproval = {
    number: 11,
    title: 'Tool call that asks first',
    model: 'gemma-4-12b-it-fp4',
    prompt,
    createReply
};
