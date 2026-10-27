import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createInitialState, reduce, summarizeTurn, groupConversations } from '../state.js';
import { statsLine, thoughtLabel, contextFact, thinkingFact, finishNotice, activityCard, folderName, planProgress } from '../format.js';

function apply( messages )
{
    let state = createInitialState();

    for ( const message of messages )
        state = reduce( state, message );

    return state;
}

function sessionState( model, conversationId )
{
    const stream = { stream_id: 's', parent_stream_id: null, conversation_id: conversationId, busy: false };

    return {
        v: 1,
        type: 'session.state',
        model,
        no_model_reason: null,
        context_length: 56320,
        context_automatic: true,
        thinking_capable: true,
        thinking: 'on',
        effort: null,
        effort_levels: null,
        detail: 'thoughts',
        sampling: {},
        modalities: [],
        busy: false,
        streams: [ stream ]
    };
}

const begin = { type: 'turn.begin', turn_id: 't1', conversation_id: 'c1' };

function text( channel, value, round )
{
    return { type: 'turn.text', turn_id: 't1', round, channel, text: value };
}

function bufferedTextOneMessagePerRoundRendersAsStreamedTextDoes()
{
    const streamed = apply( [ begin, text( 'answer', 'Hel', 1 ), text( 'answer', 'lo', 1 ) ] );
    const buffered = apply( [ begin, text( 'answer', 'Hello', 1 ) ] );

    assert.equal( streamed.transcript[ 0 ].blocks[ 0 ].text, 'Hello' );
    assert.equal( buffered.transcript[ 0 ].blocks[ 0 ].text, 'Hello' );
}

test( 'buffered text, one message per round, renders as streamed text does', bufferedTextOneMessagePerRoundRendersAsStreamedTextDoes );

function aThoughtBreakStartsANewThoughtBlockEvenWithNoToolCallBetween()
{
    const breakMessage = { type: 'turn.thought_break', turn_id: 't1', round: 2 };
    const state = apply( [ begin, text( 'thought', 'a', 1 ), breakMessage, text( 'thought', 'b', 2 ) ] );
    const blocks = state.transcript[ 0 ].blocks;

    assert.equal( blocks.length, 2 );
    assert.equal( blocks[ 1 ].text, 'b' );
    assert.equal( blocks[ 1 ].round, 2 );
}

test( 'a thought break starts a new thought block even with no tool call between', aThoughtBreakStartsANewThoughtBlockEvenWithNoToolCallBetween );

function aRoundEndLabelsThatRoundsThoughtWithItsTokenCount()
{
    const roundEnd = { type: 'turn.round_end', turn_id: 't1', round: 1, prefill_ms: 200, tokens_per_second: 40, tokens: 80, thought_tokens: 30, finish: 'stop' };
    const state = apply( [ begin, text( 'thought', 'x', 1 ), text( 'answer', 'y', 1 ), roundEnd ] );
    const thought = state.transcript[ 0 ].blocks[ 0 ];

    assert.equal( thought.tokens, 30 );
    assert.equal( thoughtLabel( thought, false ), 'Thought for 30 tokens' );
}

test( 'a round end labels that round\'s thought with its token count', aRoundEndLabelsThatRoundsThoughtWithItsTokenCount );

function aToolResultFindsItsCallById()
{
    const call = { type: 'turn.tool_call', turn_id: 't1', round: 1, call_id: 'k', name: 'get_weather', summary: 'Getting weather for Paris, France', arguments: '{}' };
    const result = { type: 'turn.tool_result', turn_id: 't1', call_id: 'k', result: 'sunny', is_error: false };
    const state = apply( [ begin, call, result ] );

    assert.equal( state.transcript[ 0 ].blocks[ 0 ].result, 'sunny' );
}

test( 'a tool result finds its call by id', aToolResultFindsItsCallById );

function onlyTheChangedTurnAndBlockBecomeNewObjects()
{
    const before = apply( [ begin, text( 'thought', 'a', 1 ), text( 'answer', 'b', 1 ) ] );
    const after = reduce( before, text( 'answer', 'c', 1 ) );

    assert.notEqual( after.transcript[ 0 ], before.transcript[ 0 ] );
    assert.equal( after.transcript[ 0 ].blocks[ 0 ], before.transcript[ 0 ].blocks[ 0 ] );
    assert.notEqual( after.transcript[ 0 ].blocks[ 1 ], before.transcript[ 0 ].blocks[ 1 ] );
}

test( 'only the changed turn and block become new objects', onlyTheChangedTurnAndBlockBecomeNewObjects );

function textForAnUnknownTurnChangesNothing()
{
    const before = apply( [ begin ] );
    const stray = { type: 'turn.text', turn_id: 'other', round: 1, channel: 'answer', text: 'x' };

    assert.equal( reduce( before, stray ), before );
}

test( 'text for an unknown turn changes nothing', textForAnUnknownTurnChangesNothing );

function unknownMessageTypesAreIgnored()
{
    const before = apply( [ begin ] );
    const unknown = { type: 'something.new', value: 1 };

    assert.equal( reduce( before, unknown ), before );
}

test( 'unknown message types are ignored', unknownMessageTypesAreIgnored );

function sendingAddsTheUserTurnAndTakesThePendingAttachmentsWithIt()
{
    const add = { type: 'attachment.add', attachment_id: 'a1', kind: 'image', name: 'cat.png', path: '/cat.png' };
    const send = { type: 'turn.send', text: 'Look', attachment_ids: [ 'a1' ] };
    const state = apply( [ add, send ] );

    assert.equal( state.transcript[ 0 ].kind, 'user' );
    assert.equal( state.transcript[ 0 ].attachments[ 0 ].name, 'cat.png' );
    assert.equal( state.attachments.length, 0 );
}

test( 'sending adds the user turn and takes the pending attachments with it', sendingAddsTheUserTurnAndTakesThePendingAttachmentsWithIt );

function anImagePreviewFollowsTheAttachmentIntoTheUserTurn()
{
    const add = { type: 'attachment.add', attachment_id: 'a1', kind: 'image', name: 'cat.png', path: '/cat.png', previewUrl: 'blob:mila://app/1' };
    const send = { type: 'turn.send', text: 'Look', attachment_ids: [ 'a1' ] };
    const pending = apply( [ add ] );
    const sent = apply( [ add, send ] );

    assert.equal( pending.attachments[ 0 ].previewUrl, 'blob:mila://app/1' );
    assert.equal( sent.transcript[ 0 ].attachments[ 0 ].previewUrl, 'blob:mila://app/1' );
}

test( 'an image preview follows the attachment into the user turn', anImagePreviewFollowsTheAttachmentIntoTheUserTurn );

function loadingAnotherModelClearsTheTranscript()
{
    const first = sessionState( 'gemma-4-12b-it-fp4', 'c1' );
    const second = sessionState( 'Llama-3.2-3B-Instruct-fp4', null );
    const state = apply( [ first, begin, text( 'answer', 'x', 1 ), second ] );

    assert.equal( state.transcript.length, 0 );
    assert.equal( state.conversationId, null );
}

test( 'loading another model clears the transcript', loadingAnotherModelClearsTheTranscript );

function aLoadedConversationBecomesTheTranscript()
{
    const stats = { prefill_ms: 100, tokens_per_second: 40, tokens: 120, rounds: 2, finish: 'stop' };
    const toolCalls = [
        { call_id: 'k', name: 'get_weather', summary: 'Getting weather for Oslo', arguments: '{}', decision: 'allow_once', result: 'cloudy', is_error: false },
        { call_id: 'm', name: 'get_weather', summary: 'Getting weather for Bergen', arguments: '{}', decision: null, result: null, is_error: false },
        { call_id: 'n', name: 'get_weather', summary: 'Getting weather for Tromso', arguments: '{}', decision: 'deny', result: 'declined', is_error: true }
    ];
    const user = { role: 'user', content: 'Hi', thought: null, tool_calls: null, attachments: null, stats: null };
    const assistant = { role: 'assistant', content: 'Hello', thought: 'greet', tool_calls: toolCalls, attachments: null, stats };
    const loaded = { type: 'conversation.loaded', id: 'c9', title: 'Hi', model: 'm', messages: [ user, assistant ] };
    const state = apply( [ loaded ] );
    const turn = state.transcript[ 1 ];
    const tools = turn.blocks.filter( ( block ) => block.kind === 'tool' );

    assert.equal( state.conversationId, 'c9' );
    assert.equal( state.transcript[ 0 ].text, 'Hi' );
    assert.deepEqual( turn.blocks.map( ( block ) => block.kind ), [ 'thought', 'tool', 'tool', 'tool', 'answer' ] );
    assert.deepEqual( tools.map( ( tool ) => tool.summary ), [ 'Getting weather for Oslo', 'Getting weather for Bergen', 'Getting weather for Tromso' ] );
    assert.deepEqual( tools.map( ( tool ) => tool.notRun ), [ false, true, false ] );
    assert.deepEqual( tools.map( ( tool ) => tool.decision ), [ 'allow_once', null, 'deny' ] );
    assert.deepEqual( tools.map( ( tool ) => tool.isError ), [ false, false, true ] );
    assert.equal( summarizeTurn( turn ).rounds, 2 );
}

test( 'a loaded conversation becomes the transcript', aLoadedConversationBecomesTheTranscript );

function aRefusedInstallIsKeptForTheModelsViewUntilTheNextModelRequest()
{
    const refused = { type: 'error', text: 'Installed, but loading it failed: it is a base model.', during: 'install' };
    const turnError = { type: 'error', text: 'A reply is still being written.', during: 'turn' };
    const installing = { type: 'model.install', name: 'gpt2-small' };
    const afterRefusal = apply( [ refused ] );

    assert.equal( afterRefusal.modelsError.text, refused.text );
    assert.equal( afterRefusal.transcript.at( -1 ).text, refused.text, 'the transcript keeps it too' );
    assert.equal( apply( [ refused, turnError ] ).modelsError.text, refused.text, 'an error elsewhere leaves it' );
    assert.equal( apply( [ turnError ] ).modelsError, null );
    assert.equal( apply( [ refused, installing ] ).modelsError, null );
}

test( 'a refused install is kept for the Models view until the next model request', aRefusedInstallIsKeptForTheModelsViewUntilTheNextModelRequest );

function aReplysFiguresSumItsRoundsAndWeighTheRateByTokens()
{
    const first = { type: 'turn.round_end', turn_id: 't1', round: 1, prefill_ms: 200, tokens_per_second: 40, tokens: 40, thought_tokens: 0, finish: 'tool_call' };
    const second = { type: 'turn.round_end', turn_id: 't1', round: 2, prefill_ms: 50, tokens_per_second: 20, tokens: 40, thought_tokens: 0, finish: 'stop' };
    const state = apply( [ begin, first, second ] );
    const summary = summarizeTurn( state.transcript[ 0 ] );

    assert.equal( summary.prefillMs, 250 );
    assert.equal( summary.tokens, 80 );
    assert.ok( Math.abs( summary.tokensPerSecond - 80 / 3 ) < 1e-9 );
    assert.equal( statsLine( summary ), 'Prefill 250 ms · 26.7 tok/s · 80 tokens · 2 rounds' );
}

test( 'a reply\'s figures sum its rounds and weigh the rate by tokens', aReplysFiguresSumItsRoundsAndWeighTheRateByTokens );

function conversationsGroupByTheViewersDaysNewestFirstEmptyGroupsLeft()
{
    const now = new Date( 2026, 9, 21, 15, 0, 0 );
    const at = ( days, hours ) => new Date( 2026, 9, 21 - days, hours, 0, 0 ).toISOString();
    const items = [
        { id: 'old', title: 'o', updated: at( 30, 9 ), model: 'm' },
        { id: 'today-early', title: 'a', updated: at( 0, 1 ), model: 'm' },
        { id: 'today-late', title: 'b', updated: at( 0, 14 ), model: 'm' },
        { id: 'week', title: 'w', updated: at( 6, 12 ), model: 'm' }
    ];
    const groups = groupConversations( items, now );

    assert.deepEqual( groups.map( ( group ) => group.label ), [ 'Today', 'Previous 7 days', 'Older' ] );
    assert.deepEqual( groups[ 0 ].items.map( ( item ) => item.id ), [ 'today-late', 'today-early' ] );
}

test( 'conversations group by the viewer\'s days, newest first, empty groups left out', conversationsGroupByTheViewersDaysNewestFirstEmptyGroupsLeft );

function theHeaderFactsReadAsTheConsolePrintsThem()
{
    const session = sessionState( 'gemma-4-12b-it-fp4', null );

    assert.equal( contextFact( session ), 'Context window 56,320 (auto)' );
    assert.equal( thinkingFact( session ), 'Thinking on' );

    const plain = { ...session, thinking_capable: false, context_automatic: false };
    assert.equal( thinkingFact( plain ), null );
    assert.equal( contextFact( plain ), 'Context window 56,320' );
}

test( 'the header facts read as the console prints them', theHeaderFactsReadAsTheConsolePrintsThem );

function aReplyThatDidNotFinishOnItsOwnSaysWhy()
{
    assert.equal( finishNotice( 'stop' ), null );
    assert.match( finishNotice( 'length' ), /length limit/ );
    assert.match( finishNotice( 'context' ), /context window/ );
}

test( 'a reply that did not finish on its own says why', aReplyThatDidNotFinishOnItsOwnSaysWhy );

function anActivityIsNamedForItsCardAndAFolderByItsLastPart()
{
    assert.equal( activityCard( 'build' ).name, 'Build & Code' );
    assert.equal( activityCard( 'research' ).name, 'Research & Analyze' );
    assert.equal( activityCard( 'tasks' ).name, 'Get Things Done' );
    assert.equal( activityCard( 'plan' ).name, 'plan' );
    assert.equal( planProgress( [ { text: 'a', status: 'done' }, { text: 'b', status: 'skipped' }, { text: 'c', status: 'in_progress' } ] ), '2 of 3 done' );
    assert.equal( folderName( 'C:\\Users\\you\\Projects\\calc\\' ), 'calc' );
    assert.equal( folderName( '/home/you/calc' ), 'calc' );
}

test( 'an activity is named for its card, and a folder by its last part', anActivityIsNamedForItsCardAndAFolderByItsLastPart );
