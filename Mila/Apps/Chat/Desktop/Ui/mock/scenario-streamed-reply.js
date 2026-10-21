// Scenario 1: a Gemma reply streamed token by token -- thinking, a get_weather call and its
// result, more thinking, then an answer with a table and a code block.

import { splitIntoTokens } from './fixtures.js';

const tokenMilliseconds = 26;
const toolCallTokens = 17;

const prompt = 'What\'s the weather in Toronto right now? Is it good for cycling to work, and can you give me a quick Python function to convert it to Fahrenheit?';

const firstThought = 'The user is asking about current conditions in Toronto, so I need live data rather than a guess. '
    + 'I have a get_weather tool that takes a location, and "Toronto, Canada" is unambiguous. '
    + 'Once I have the reading I can judge whether it suits cycling and write the conversion.';

const secondThought = 'The tool reports 18 °C and cloudy. That is comfortable for a commute: warm enough for a light layer, and no glare. '
    + 'It gave no wind or rain figures, so I should not claim anything about either; I will say so.\n\n'
    + 'For the conversion, F = C × 9/5 + 32, so 18 °C is 64.4 °F. A small typed function with an example call will do. '
    + 'Conditions in a table first, then the code.';

const answer = 'It\'s **18 °C and cloudy** in Toronto right now, which is good weather for riding to work.\n\n'
    + '| | Now | For the ride |\n'
    + '|---|---|---|\n'
    + '| Temperature | 18 °C (64.4 °F) | A light layer is enough; you will warm up within minutes |\n'
    + '| Sky | Cloudy | No glare, and no sun to overheat in |\n'
    + '| Wind and rain | Not reported | Check a forecast before you leave if either matters |\n\n'
    + 'Here is the conversion as a function:\n\n'
    + '```python\n'
    + 'def celsius_to_fahrenheit(celsius: float) -> float:\n'
    + '    """Convert a temperature from Celsius to Fahrenheit."""\n'
    + '    return celsius * 9 / 5 + 32\n'
    + '\n'
    + '\n'
    + 'print(celsius_to_fahrenheit(18))  # 64.4\n'
    + '```\n\n'
    + 'Multiply by `9 / 5` and add 32. To go the other way, subtract 32 and multiply by `5 / 9`.';

function streamSteps( text, round, channel )
{
    const pieces = splitIntoTokens( text );

    return pieces.map( ( piece ) => ( { delay: tokenMilliseconds, message: { type: 'turn.text', round, channel, text: piece } } ) );
}

function tokensPerSecond()
{
    return Math.round( 10000 / tokenMilliseconds ) / 10;
}

function createReply()
{
    const firstThoughtSteps = streamSteps( firstThought, 1, 'thought' );
    const secondThoughtSteps = streamSteps( secondThought, 2, 'thought' );
    const answerSteps = streamSteps( answer, 2, 'answer' );
    const rate = tokensPerSecond();
    const location = { location: 'Toronto, Canada' };
    const argumentsJson = JSON.stringify( location );
    const result = '{"temperature_c": 18, "condition": "cloudy"}';

    const firstRoundEnd = {
        type: 'turn.round_end',
        round: 1,
        prefill_ms: 212,
        tokens_per_second: rate,
        tokens: firstThoughtSteps.length + toolCallTokens,
        thought_tokens: firstThoughtSteps.length,
        finish: 'tool_call'
    };

    const toolCall = {
        type: 'turn.tool_call',
        round: 1,
        call_id: 'call-1',
        name: 'get_weather',
        summary: 'Getting weather for Toronto, Canada',
        arguments: argumentsJson
    };

    const toolResult = { type: 'turn.tool_result', call_id: 'call-1', result, is_error: false };

    const secondRoundEnd = {
        type: 'turn.round_end',
        round: 2,
        prefill_ms: 47,
        tokens_per_second: rate,
        tokens: secondThoughtSteps.length + answerSteps.length,
        thought_tokens: secondThoughtSteps.length,
        finish: 'stop'
    };

    const end = { type: 'turn.end', finish: 'stop', answer };

    return [
        { delay: 212, message: { type: 'turn.begin' } },
        ...firstThoughtSteps,
        { delay: toolCallTokens * tokenMilliseconds, message: toolCall },
        { delay: 0, message: firstRoundEnd },
        { delay: 140, message: toolResult },
        { delay: 47, message: { type: 'turn.thought_break', round: 2 } },
        ...secondThoughtSteps,
        ...answerSteps,
        { delay: tokenMilliseconds, message: secondRoundEnd },
        { delay: 0, message: end }
    ];
}

export const streamedReply = {
    number: 1,
    title: 'Gemma reply, streamed',
    model: 'gemma-4-12b-it-fp4',
    prompt,
    createReply
};
