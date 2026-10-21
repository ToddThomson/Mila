// What the mock session holds before any scenario runs: the installed models, their facts, and a
// conversation store with something in every date group. Sizes and rates are plausible for a
// 12 GB card, not measurements.

export const weatherTool = {
    name: 'get_weather',
    summary: 'Getting weather for {location}'
};

export const models = [
    {
        name: 'gemma-4-12b-it-fp4',
        size_bytes: 6800000000,
        quantization: 'fp4',
        family: 'gemma',
        context_length: 56320,
        thinking_capable: true,
        effort_levels: false,
        modalities: [ 'image', 'audio' ],
        sampling: { temperature: 1.0, top_k: 64, top_p: 0.95, presence_penalty: 0 }
    },
    {
        name: 'Llama-3.2-3B-Instruct-fp4',
        size_bytes: 3100000000,
        quantization: 'fp4',
        family: 'llama',
        context_length: 31744,
        thinking_capable: false,
        effort_levels: false,
        modalities: [],
        sampling: { temperature: 0.6, top_k: 0, top_p: 0.9, presence_penalty: 0 }
    },
    {
        name: 'Llama-3.1-8B-Instruct-fp4',
        size_bytes: 5400000000,
        quantization: 'fp4',
        family: 'llama',
        context_length: 40960,
        thinking_capable: false,
        effort_levels: false,
        modalities: [],
        sampling: { temperature: 0.6, top_k: 0, top_p: 0.9, presence_penalty: 0 }
    },
    {
        name: 'qwen3.8-27b-fp4',
        size_bytes: 14600000000,
        quantization: 'fp4',
        family: 'qwen',
        context_length: 16384,
        thinking_capable: true,
        effort_levels: true,
        modalities: [],
        sampling: { temperature: 0.6, top_k: 20, top_p: 0.95, presence_penalty: 0 }
    }
];

export function findModel( name )
{
    const folded = name.toLowerCase();

    return models.find( ( model ) => model.name.toLowerCase() === folded ) || null;
}

const hourMilliseconds = 60 * 60 * 1000;
const dayMilliseconds = 24 * hourMilliseconds;

function userMessage( content )
{
    return { role: 'user', content, thought: null, tool_calls: null, attachments: null, stats: null };
}

function assistantMessage( content, thought, stats )
{
    return { role: 'assistant', content, thought, tool_calls: null, attachments: null, stats };
}

function stats( prefillMs, tokensPerSecond, tokens )
{
    return { prefill_ms: prefillMs, tokens_per_second: tokensPerSecond, tokens, rounds: 1, finish: 'stop' };
}

/**
 * Five stored conversations, one or two per date group, dated back from now so the groups hold
 * whatever day the page is opened on.
 */
export function seedConversations( now )
{
    const kvStats = stats( 204, 38.6, 312 );
    const lifetimeStats = stats( 118, 61.2, 268 );
    const kyotoStats = stats( 233, 37.9, 541 );
    const regexStats = stats( 590, 14.8, 402 );
    const attentionStats = stats( 61, 118.4, 236 );

    const seeds = [
        {
            id: 'conversation-1',
            title: 'Why does a KV cache make decoding faster?',
            model: 'gemma-4-12b-it-fp4',
            age: 2 * hourMilliseconds,
            messages: [
                userMessage( 'Why does a KV cache make decoding faster?' ),
                assistantMessage(
                    'Each new token attends to every earlier one. Without a cache, the model recomputes the keys and values for the whole sequence at every step, so the work per token grows with the length.\n\nA **KV cache** keeps the keys and values from earlier steps. Each decode step then computes them for one token and reads the rest from memory, which turns a quadratic cost into a linear one.',
                    'The user wants the intuition. Keys and values of past tokens do not change, so they can be stored.',
                    kvStats )
            ]
        },
        {
            id: 'conversation-2',
            title: 'Rust lifetimes in a struct that holds a slice',
            model: 'Llama-3.1-8B-Instruct-fp4',
            age: dayMilliseconds + 3 * hourMilliseconds,
            messages: [
                userMessage( 'How do I write a Rust struct that holds a slice of another struct\'s data?' ),
                assistantMessage(
                    'Give the struct a lifetime parameter and use it on the slice:\n\n```rust\nstruct Window<\'a> {\n    samples: &\'a [f32],\n}\n```\n\nThe compiler then checks that no `Window` outlives the data it borrows.',
                    null,
                    lifetimeStats )
            ]
        },
        {
            id: 'conversation-3',
            title: 'Plan a three-day trip to Kyoto',
            model: 'gemma-4-12b-it-fp4',
            age: 3 * dayMilliseconds,
            messages: [
                userMessage( 'Plan a three-day trip to Kyoto in November.' ),
                assistantMessage(
                    '**Day 1: Higashiyama.** Kiyomizu-dera early, then down through Sannenzaka to Gion.\n\n**Day 2: Arashiyama.** The bamboo grove at opening, Tenryu-ji, and the river in the afternoon.\n\n**Day 3: Fushimi Inari.** Start before eight; the upper trail is quiet and the maples are at their peak.',
                    'November means autumn leaves, so crowds are heavy. Early starts matter.',
                    kyotoStats )
            ]
        },
        {
            id: 'conversation-4',
            title: 'A regular expression for ISO 8601 dates',
            model: 'qwen3.8-27b-fp4',
            age: 5 * dayMilliseconds,
            messages: [
                userMessage( 'Give me a regular expression for ISO 8601 dates.' ),
                assistantMessage(
                    'For calendar dates alone:\n\n```\n^\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])$\n```\n\nIt does not check month lengths, so `2026-02-31` passes. Validate with a date library when that matters.',
                    'A full ISO 8601 grammar is large; the user probably wants YYYY-MM-DD.',
                    regexStats )
            ]
        },
        {
            id: 'conversation-5',
            title: 'Summarize "Attention Is All You Need"',
            model: 'Llama-3.2-3B-Instruct-fp4',
            age: 19 * dayMilliseconds,
            messages: [
                userMessage( 'Summarize "Attention Is All You Need" in three sentences.' ),
                assistantMessage(
                    'The paper introduces the Transformer, a sequence model built only from attention and feed-forward layers, with no recurrence. Multi-head self-attention lets every position look at every other in one step, which trains far faster in parallel. It set new results in machine translation and became the basis of today\'s language models.',
                    null,
                    attentionStats )
            ]
        }
    ];

    return seeds.map( ( seed ) => storedConversation( seed, now ) );
}

function storedConversation( seed, now )
{
    const updated = new Date( now - seed.age );
    const stamp = updated.toISOString();

    return { id: seed.id, title: seed.title, model: seed.model, created: stamp, updated: stamp, messages: seed.messages };
}

/**
 * Text cut into token-sized pieces, each carrying its leading whitespace, so a stream of them
 * arrives as a model's would: words in parts, punctuation on its own.
 */
export function splitIntoTokens( text )
{
    const pieces = text.match( /\s*[\p{L}\p{N}]{1,5}|\s*[^\s\p{L}\p{N}]|\s+/gu );

    return pieces === null ? [] : pieces;
}

/**
 * The title a new conversation takes: its first message, cut at a word under 60 characters
 * (ChatDesktop.md 6.3).
 */
export function titleFor( text )
{
    const flattened = text.replace( /\s+/g, ' ' ).trim();

    if ( flattened.length < 60 )
        return flattened;

    const cut = flattened.slice( 0, 59 );
    const space = cut.lastIndexOf( ' ' );

    return space > 0 ? cut.slice( 0, space ) : cut;
}
