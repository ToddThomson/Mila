// What the mock session holds before any scenario runs: the installed models, their facts, and a
// conversation store with something in every date group. Sizes and rates are plausible for a
// 12 GB card, not measurements. Only Gemma declares a display name, as a manifest written before
// the field leaves it out.

// The licence the session carries beside it (License.md), and a digest standing for its own.
export const mockLicense = {
    sha256: 'mock-license-sha256',
    text: 'The MIT License (MIT)\n\n'
        + 'Copyright (c) 2021..2026 Todd J. Thomson\n\n'
        + 'Permission is hereby granted, free of charge, to any person obtaining a copy\n'
        + 'of this software and associated documentation files (the "Software"), to deal\n'
        + 'in the Software without restriction, including without limitation the rights\n'
        + 'to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n'
        + 'copies of the Software, and to permit persons to whom the Software is\n'
        + 'furnished to do so, subject to the following conditions:\n\n'
        + 'The above copyright notice and this permission notice shall be included in all\n'
        + 'copies or substantial portions of the Software.\n\n'
        + 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n'
        + 'IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n'
        + 'FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n'
        + 'AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n'
        + 'LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n'
        + 'OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n'
        + 'SOFTWARE.\n'
};

export const weatherTool = {
    name: 'get_weather',
    summary: 'Getting weather for {location}'
};

export const models = [
    {
        name: 'gemma-4-12b-it-fp4',
        display_name: 'Gemma 4 12B',
        size_bytes: 6800000000,
        quantization: 'fp4',
        family: 'gemma',
        context_length: 56320,
        thinking_capable: true,
        effort_levels: null,
        modalities: [ 'image', 'audio' ],
        published: { license: 'gemma', agentic: true, tools: true, planned_context: 131072 },
        sampling: { temperature: 1.0, top_k: 64, top_p: 0.95, presence_penalty: 0 }
    },
    {
        name: 'Llama-3.2-3B-Instruct-fp4',
        size_bytes: 3100000000,
        quantization: 'fp4',
        family: 'llama',
        context_length: 31744,
        thinking_capable: false,
        effort_levels: null,
        modalities: [],
        published: { license: 'llama3.2', agentic: false, tools: true, planned_context: 131072 },
        sampling: { temperature: 0.6, top_k: 0, top_p: 0.9, presence_penalty: 0 }
    },
    {
        name: 'Llama-3.1-8B-Instruct-fp4',
        size_bytes: 5400000000,
        quantization: 'fp4',
        family: 'llama',
        context_length: 40960,
        thinking_capable: false,
        effort_levels: null,
        modalities: [],
        published: { license: 'llama3.1', agentic: false, tools: true, planned_context: 98304 },
        sampling: { temperature: 0.6, top_k: 0, top_p: 0.9, presence_penalty: 0 }
    },
    {
        name: 'qwen3.8-27b-fp4',
        size_bytes: 14600000000,
        quantization: 'fp4',
        family: 'qwen',
        context_length: 16384,
        thinking_capable: true,
        effort_levels: [ 'xhigh', 'medium', 'low' ],
        effort_default: 'xhigh',
        modalities: [],
        published: { license: 'apache-2.0', agentic: true, tools: true, planned_context: null,
            why_not: 'Its weights need 14.6 GB and this GPU has 11.2 GB free.' },
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

function stats( prefillMs, tokensPerSecond, tokens, finish = 'stop' )
{
    return { prefill_ms: prefillMs, tokens_per_second: tokensPerSecond, tokens, rounds: 1, finish };
}

/** A file tool's call as the store keeps it: what its card showed, and its result. */
function fileCall( name, path, result )
{
    const verb = name === 'write_file' ? 'Writing ' : 'Editing ';

    return { call_id: 'call-' + path, name, summary: verb + path, arguments: JSON.stringify( { path } ),
        decision: 'allow_once', result, is_error: false };
}

function assistantWithCalls( content, calls, replyStats )
{
    return { role: 'assistant', content, thought: null, tool_calls: calls, attachments: null, stats: replyStats };
}

/**
 * What a stored conversation reports of itself in `conversations.list`, as the session's summary does
 * (Desktop.ConversationSummary): how far its plan came, how its last reply began and ended, and the files its calls
 * wrote.
 */
export function summarizeConversation( conversation )
{
    const messages = Array.isArray( conversation.messages ) ? conversation.messages : [];
    const plan = Array.isArray( conversation.plan ) && conversation.plan.length > 0 ? conversation.plan : null;
    let progress = null;

    if ( plan !== null )
    {
        const finished = plan.filter( ( step ) => step.status === 'done' || step.status === 'skipped' ).length;
        const working = plan.find( ( step ) => step.status === 'in_progress' ) ?? plan.find( ( step ) => step.status === 'pending' );

        progress = { steps: plan.length, finished, current: working === undefined ? null : working.text };
    }

    const replies = messages.filter( ( message ) => message.role === 'assistant' );
    const last = replies.length === 0 ? null : replies[ replies.length - 1 ];
    const finish = last !== null && last.stats ? last.stats.finish ?? null : null;
    const line = last !== null && typeof last.content === 'string' ? last.content.replace( /\s+/g, ' ' ).trim().slice( 0, 160 ) : '';
    const replyStart = line.length > 0 ? line : null;
    let wrote = [];

    for ( const message of messages )
    {
        for ( const call of message.tool_calls ?? [] )
        {
            if ( ( call.name !== 'write_file' && call.name !== 'edit_file' ) || call.is_error || call.result === null )
                continue;

            const path = JSON.parse( call.arguments ).path;
            wrote = wrote.filter( ( file ) => file.path !== path );
            wrote.push( { path, tool: call.name } );
        }
    }

    return { progress, finish, reply_start: replyStart, wrote };
}

/**
 * Seven stored conversations, one to three per date group, dated back from now so the groups hold whatever day the
 * page is opened on: questions, a build that edited a file, a plan stopped partway with files written, and research
 * whose reply was cut short, so the home screen has every section.
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
            activity: 'build',
            folder: 'C:\\Users\\you\\Projects\\window',
            age: dayMilliseconds + 3 * hourMilliseconds,
            messages: [
                userMessage( 'How do I write a Rust struct that holds a slice of another struct\'s data?' ),
                assistantWithCalls(
                    'Give the struct a lifetime parameter and use it on the slice:\n\n```rust\nstruct Window<\'a> {\n    samples: &\'a [f32],\n}\n```\n\nThe compiler then checks that no `Window` outlives the data it borrows. I changed `src/window.rs` to match.',
                    [ fileCall( 'edit_file', 'src/window.rs', 'Edited src/window.rs.' ) ],
                    lifetimeStats )
            ]
        },
        {
            id: 'conversation-6',
            title: 'Tidy the calculator project\'s tests',
            model: 'gemma-4-12b-it-fp4',
            activity: 'tasks',
            folder: 'C:\\Users\\you\\Projects\\calc',
            age: 5 * hourMilliseconds,
            plan: [
                { text: 'Read the tests and list what each covers', status: 'done' },
                { text: 'Move the parser cases into their own file', status: 'done' },
                { text: 'Add cases for division by zero', status: 'in_progress' },
                { text: 'Run the suite and fix what fails', status: 'pending' },
                { text: 'Summarize the changes', status: 'pending' }
            ],
            messages: [
                userMessage( 'Tidy the calculator project\'s tests: group them, fill the gaps, make them pass.' ),
                assistantWithCalls(
                    'The parser cases are in `tests/test_parser.py` now, and the suite\'s notes are in `TESTING.md`. Next is division by zero.',
                    [ fileCall( 'write_file', 'tests/test_parser.py', 'Wrote tests/test_parser.py.' ),
                        fileCall( 'write_file', 'TESTING.md', 'Wrote TESTING.md.' ) ],
                    stats( 410, 36.2, 1820, 'stop' ) )
            ]
        },
        {
            id: 'conversation-7',
            title: 'Compare SQLite and DuckDB for analytics',
            model: 'gemma-4-12b-it-fp4',
            activity: 'research',
            age: dayMilliseconds + 6 * hourMilliseconds,
            messages: [
                userMessage( 'Compare SQLite and DuckDB for running analytics over a few GB of CSV files.' ),
                assistantMessage(
                    'Both are embedded and need no server. **DuckDB** is columnar and vectorized, so scans and aggregates over wide tables run far faster; **SQLite** stores rows, which suits many small reads and writes. For a few GB of CSV',
                    null,
                    stats( 380, 37.0, 2048, 'length' ) )
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
                    'The paper introduces the Transformer, a sequence model built only from attention and feed-forward layers, with no recurrence. Multi-head self-attention lets every position look at every other in one step, which trains far faster in parallel. It set new results in machine translation and became the basis of today\'s language models.\n\nThe paper is at [arXiv 1706.03762](https://arxiv.org/abs/1706.03762).',
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

    const activity = seed.activity === undefined ? 'ask' : seed.activity;
    const folder = seed.folder === undefined ? null : seed.folder;
    const plan = seed.plan === undefined ? null : seed.plan;

    return { id: seed.id, title: seed.title, model: seed.model, activity, folder, plan, created: stamp, updated: stamp, messages: seed.messages };
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
 * (MilaDesktop.md 6.3).
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
