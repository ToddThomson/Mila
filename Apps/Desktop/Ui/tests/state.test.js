import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createInitialState, reduce, summarizeTurn, appStatus, emptyScreen, isAskingForModel, isWelcoming, firstRunStep,
    suggestedModel, setupPane, workspaceOverview, recentChats, matchingChats, searchResults, activityDays, barMode,
    isRisen } from '../state.js';
import { statsLine, thoughtLabel, contextShare, contextDetails, thinkingFact, finishNotice, folderName, planProgress, barPrompt,
    modelAbilities, fitLine, modelTitle, namedTitle, whenLabel, attentionLine, progressLine, plainLine, activityText, dayLabel,
    initials } from '../format.js';

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

function anApprovalKeepsWhatTheCallWouldReachBesideOneThatSaysNothing()
{
    const call = { type: 'turn.tool_call', turn_id: 't1', round: 1, call_id: 'k', name: 'run_command', summary: 'Running git status', arguments: '{}' };
    const reaches = { folders: [ { path: 'D:\\project', access: 'read_write' } ], program_folders: [ 'C:\\Git\\cmd' ], network: false, runs_programs: true };
    const asked = { type: 'turn.tool_approval', turn_id: 't1', call_id: 'k', choices: [ 'allow_once', 'deny' ], reaches };
    const silent = { type: 'turn.tool_approval', turn_id: 't1', call_id: 'k', choices: [ 'allow_once', 'deny' ] };

    assert.deepEqual( apply( [ begin, call, asked ] ).transcript[ 0 ].blocks[ 0 ].reaches, reaches );
    assert.equal( apply( [ begin, call, silent ] ).transcript[ 0 ].blocks[ 0 ].reaches, null );
}

test( 'an approval keeps what the call would reach beside one that says nothing', anApprovalKeepsWhatTheCallWouldReachBesideOneThatSaysNothing );

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
    assert.equal( afterRefusal.transcript.length, 0, 'before a conversation, only where the models are' );
    assert.equal( apply( [ begin, refused ] ).transcript.at( -1 ).text, refused.text, 'in a conversation, the transcript keeps it too' );
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

// Local hours, so every one falls on the viewer's same day wherever the tests run.
const at = ( hours ) => new Date( 2026, 10, 16, hours, 0, 0 ).toISOString();

const workspaceItems = [
    { id: 'question', title: 'Why does a KV cache grow?', created: at( 9 ), updated: at( 9 ), folder: null, progress: null, finish: 'stop',
        wrote: [], reply_start: 'Because **every** token adds a key and a value.' },
    { id: 'cut', title: 'Long answer', created: at( 10 ), updated: at( 10 ), folder: null, progress: null, finish: 'length', wrote: [],
        reply_start: 'The short version' },
    { id: 'plan', title: 'Tidy tests', created: at( 6 ), updated: at( 11 ), folder: 'C:\\calc',
        progress: { steps: 4, finished: 1, current: 'Fix the parser' }, finish: 'stop',
        wrote: [ { path: 'tests/a.py', tool: 'write_file' }, { path: 'notes.md', tool: 'write_file' } ], reply_start: null },
    { id: 'build', title: 'Fix the calculator', created: at( 8 ), updated: at( 8 ), folder: 'C:\\calc', progress: null, finish: 'stop',
        wrote: [ { path: 'tests/a.py', tool: 'edit_file' } ], reply_start: 'Fixed.' },
    { id: 'finished', title: 'Done plan', created: at( 7 ), updated: at( 7 ), folder: 'C:\\calc',
        progress: { steps: 2, finished: 2, current: null }, finish: 'stop', wrote: [], reply_start: 'All done.' }
];

function workspace( messages )
{
    return apply( [ { v: 1, type: 'conversations.list', items: workspaceItems }, ...messages ] );
}

function theOverviewCountsTheChatsAndSaysWhatIsWaiting()
{
    const overview = workspaceOverview( workspace( [] ) );

    assert.equal( overview.chatCount, 5 );
    assert.equal( overview.newest.id, 'plan' );
    assert.deepEqual( overview.waiting.map( ( row ) => row.item.id ), [ 'plan', 'cut' ] );
    assert.equal( attentionLine( overview.waiting[ 0 ] ), 'Stopped at step 2 of 4: Fix the parser' );
    assert.equal( attentionLine( overview.waiting[ 1 ] ), 'The last reply was cut short' );
    assert.equal( progressLine( workspaceItems[ 4 ].progress ), 'Finished, 2 steps' );
    assert.deepEqual( overview.chats.map( ( item ) => item.id ), [ 'plan', 'cut', 'question', 'build', 'finished' ] );
    assert.deepEqual( recentChats( workspace( [] ) ).map( ( item ) => item.id ), [ 'plan', 'cut', 'question', 'build' ] );
}

test( 'the overview counts the chats and says what is waiting', theOverviewCountsTheChatsAndSaysWhatIsWaiting );

function aConversationWithAReplyUnderWayNeedsNothingYet()
{
    const items = [ { id: 'c', title: 't', updated: new Date().toISOString(), folder: null, progress: null, finish: 'length', wrote: [] } ];
    const busy = { ...sessionState( 'gemma-4-12b-it-fp4', 'c' ), busy: true };
    const state = apply( [ busy, { v: 1, type: 'conversations.list', items }, { v: 1, type: 'conversation.loaded', id: 'c', title: 't',
        model: 'gemma-4-12b-it-fp4', activity: 'research', folder: null, messages: [] } ] );

    assert.equal( workspaceOverview( state ).waiting.length, 0 );
}

test( 'a conversation with a reply under way needs nothing yet', aConversationWithAReplyUnderWayNeedsNothingYet );

function allChatsAndSearchFindEveryWordTyped()
{
    const state = workspace( [] );

    assert.equal( matchingChats( state, '' ).length, 5 );
    assert.deepEqual( matchingChats( state, 'kv GROW' ).map( ( item ) => item.id ), [ 'question' ] );
    assert.deepEqual( matchingChats( state, 'every token' ).map( ( item ) => item.id ), [ 'question' ], 'the start of the reply counts' );

    // Search lists chats by title, then each file once, at the latest chat that wrote it.
    const found = searchResults( state, 'a.py' );
    assert.deepEqual( found.map( ( result ) => result.kind + ':' + ( result.path ?? result.item.id ) ), [ 'file:tests/a.py' ] );
    assert.equal( found[ 0 ].item.id, 'plan' );
    assert.deepEqual( searchResults( state, 'calculator' ).map( ( result ) => result.item.id ), [ 'build' ] );
    assert.equal( searchResults( state, '' ).length, 5, 'nothing typed lists the recent chats' );
}

test( 'all chats and search find every word typed', allChatsAndSearchFindEveryWordTyped );

function activityIsByDayNewestFirstFromWhatTheStoreSays()
{
    const state = workspace( [] );
    const days = activityDays( state, 'all' );
    const events = days.flatMap( ( day ) => day.events );

    assert.equal( days.length, 1 );
    assert.equal( events.length, 5 + 2 + 2, 'every chat started, two that wrote files, two with a plan' );
    assert.deepEqual( events.slice( 0, 3 ).map( ( event ) => event.kind + ':' + event.item.id ), [ 'plan:plan', 'file:plan', 'chat:cut' ] );
    assert.equal( activityText( events[ 0 ] ), 'Stopped at step 2 of 4: Fix the parser' );
    assert.equal( activityText( events[ 1 ] ), 'Wrote notes.md and a.py' );
    assert.equal( activityText( events[ 2 ] ), 'Started a chat: Long answer' );
    assert.deepEqual( activityDays( state, 'plan' ).flatMap( ( day ) => day.events ).map( ( event ) => event.item.id ), [ 'plan', 'finished' ] );

    const now = new Date( 2026, 10, 16, 15, 0, 0 );
    assert.equal( dayLabel( new Date( 2026, 10, 16 ).getTime(), now ), 'Today' );
    assert.equal( dayLabel( new Date( 2026, 10, 15 ).getTime(), now ), 'Yesterday' );
    assert.equal( dayLabel( new Date( 2026, 10, 12 ).getTime(), now ), 'Thursday, November 12' );
}

test( 'activity is by day, newest first, from what the store says', activityIsByDayNewestFirstFromWhatTheStoreSays );

function theBarStartsAChatGoesOnWithItAndRepliesInAnOpenOne()
{
    const sent = { type: 'turn.send', text: 'Hello', attachment_ids: [] };
    const fresh = workspace( [ sessionState( 'gemma-4-12b-it-fp4', null ) ] );

    assert.equal( barMode( fresh ), 'continue', 'the session begins with a conversation of the bar' );
    assert.equal( isRisen( fresh ), false, 'an empty conversation does not rise' );

    const talking = reduce( fresh, sent );
    assert.equal( isRisen( talking ), true, 'sending from a place raises the conversation over it' );
    assert.equal( isRisen( reduce( talking, { type: 'conversation.raise', risen: false } ) ), false );

    const elsewhere = reduce( talking, { type: 'place.open', place: 'activity' } );
    assert.equal( elsewhere.place, 'activity' );
    assert.equal( isRisen( elsewhere ), true, 'the conversation stays risen over the next place' );

    const opened = reduce( talking, { type: 'conversation.open', id: 'question' } );
    assert.equal( opened.place, 'chat' );
    assert.equal( barMode( opened ), 'reply' );
    assert.equal( isRisen( reduce( opened, sent ) ), false, 'in a chat the reply is the pane’s' );

    const back = reduce( opened, { type: 'place.open', place: 'workspace' } );
    assert.equal( barMode( back ), 'start', 'from a chat opened from a list, the bar begins a new one' );
    assert.equal( reduce( back, { type: 'place.open', place: 'nowhere' } ), back );

    assert.equal( barPrompt( 'chat' ), 'Reply to Mila' );
    assert.equal( barPrompt( 'workspace' ), 'Ask Mila anything, or open one of your chats' );
}

test( 'the bar starts a chat, goes on with it, and replies in an open one', theBarStartsAChatGoesOnWithItAndRepliesInAnOpenOne );

function aChatLeftWhileItsReplyRunsRisesOverTheNextPlace()
{
    const busy = { ...sessionState( 'gemma-4-12b-it-fp4', 'question' ), busy: true };
    const opened = workspace( [ { type: 'conversation.open', id: 'question' }, busy, begin ] );
    const left = reduce( opened, { type: 'place.open', place: 'chats' } );

    assert.equal( isRisen( left ), true );
}

test( 'a chat left while its reply runs rises over the next place', aChatLeftWhileItsReplyRunsRisesOverTheNextPlace );

function aRowSaysAReplyInOneLineAndAPersonByTheirInitials()
{
    assert.equal( plainLine( '## Plan\n\n**Day 1:** see [Gion](https://example.com) and `run`' ), 'Plan Day 1: see Gion and run' );
    assert.equal( plainLine( 'Cloudy.\n\n| | Now |\n|---|:---:|\n| Sky | Grey |' ), 'Cloudy. Now Sky Grey', 'a table keeps its words' );
    assert.equal( plainLine( null ), '' );
    assert.equal( initials( 'Todd Thomson' ), 'TT' );
    assert.equal( initials( 'ToddT' ), 'TT' );
    assert.equal( initials( 'todd' ), 'T' );
    assert.equal( initials( '' ), '' );
}

test( 'a row says a reply in one line, and a person by their initials', aRowSaysAReplyInOneLineAndAPersonByTheirInitials );

function theWindowSaysWhoIsSignedInAndTheVersion()
{
    const state = apply( [ { type: 'about', user_name: 'ToddT', version: '0.21.0' } ] );

    assert.deepEqual( state.about, { userName: 'ToddT', version: '0.21.0' } );
}

test( 'the window says who is signed in, and the version', theWindowSaysWhoIsSignedInAndTheVersion );

function whenSaysTheTimeAsAPersonWould()
{
    const now = new Date( 2026, 10, 16, 15, 0, 0 );
    const before = ( minutes ) => new Date( now.getTime() - minutes * 60 * 1000 ).toISOString();

    assert.equal( whenLabel( before( 0.5 ), now ), 'Just now' );
    assert.equal( whenLabel( before( 1 ), now ), '1 minute ago' );
    assert.equal( whenLabel( before( 45 ), now ), '45 minutes ago' );
    assert.equal( whenLabel( before( 5 * 60 ), now ), '5 hours ago' );
    assert.equal( whenLabel( new Date( 2026, 10, 15, 20, 0, 0 ).toISOString(), now ), 'Yesterday' );
    assert.equal( whenLabel( new Date( 2026, 10, 12, 9, 0, 0 ).toISOString(), now ), 'Thursday' );
    assert.equal( whenLabel( new Date( 2026, 9, 30, 9, 0, 0 ).toISOString(), now ), 'Oct 30' );
}

test( 'when says the time as a person would', whenSaysTheTimeAsAPersonWould );

function theHeaderFactsReadAsTheConsolePrintsThem()
{
    const session = sessionState( 'gemma-4-12b-it-fp4', null );

    assert.equal( thinkingFact( session ), 'Thinking on' );

    const plain = { ...session, thinking_capable: false, context_automatic: false };
    assert.equal( thinkingFact( plain ), null );
}

test( 'the header facts read as the console prints them', theHeaderFactsReadAsTheConsolePrintsThem );

function theContextRingShowsWhatTheConversationTakes()
{
    const counted = { context_length: 131072, context_automatic: true, context_used: 32768, compacts_at: 126976 };

    assert.equal( contextShare( counted ), 0.25 );
    assert.deepEqual( contextDetails( counted ), [
        [ 'Used', '32,768 tokens (25%)' ],
        [ 'Context window', '131,072 tokens, chosen for this computer' ],
        [ 'Compacts at', '126,976 tokens' ],
        [ 'Left', '98,304 tokens' ] ] );

    const uncounted = { ...counted, context_used: null, compacts_at: null, context_automatic: false };

    assert.equal( contextShare( uncounted ), null );
    assert.deepEqual( contextDetails( uncounted ), [
        [ 'Used', 'Counted after the next reply' ],
        [ 'Context window', '131,072 tokens, set in Settings' ] ] );

    assert.equal( contextShare( { ...counted, context_used: 200000 } ), 1, 'a share never passes the whole' );
}

test( 'the context ring shows what the conversation takes', theContextRingShowsWhatTheConversationTakes );

function aReplyThatDidNotFinishOnItsOwnSaysWhy()
{
    assert.equal( finishNotice( 'stop' ), null );
    assert.match( finishNotice( 'length' ), /length limit/ );
    assert.match( finishNotice( 'context' ), /context window/ );
}

test( 'a reply that did not finish on its own says why', aReplyThatDidNotFinishOnItsOwnSaysWhy );

function aPlanCountsItsStepsAndAFolderIsNamedByItsLastPart()
{
    assert.equal( planProgress( [ { text: 'a', status: 'done' }, { text: 'b', status: 'skipped' }, { text: 'c', status: 'in_progress' } ] ), '2 of 3 done' );
    assert.equal( folderName( 'C:\\Users\\you\\Projects\\calc\\' ), 'calc' );
    assert.equal( folderName( '/home/you/calc' ), 'calc' );
}

test( 'a plan counts its steps, and a folder is named by its last part', aPlanCountsItsStepsAndAFolderIsNamedByItsLastPart );

function memoryComingBackIsTheTranscriptsUntilItIsDone()
{
    const restoring = { type: 'progress', task: 'restore', label: 'Getting ready', done: 0, total: 4096 };
    const loading = { type: 'progress', task: 'load', label: 'Loading', done: null, total: null };

    const during = apply( [ loading, restoring ] );
    assert.deepEqual( during.restore, restoring );
    assert.deepEqual( during.progress, loading );

    const after = apply( [ loading, restoring, { ...restoring, done: 4096 } ] );
    assert.equal( after.restore, null );
    assert.deepEqual( after.progress, loading );
}

test( 'memory coming back is the transcript\'s to show, until it is done', memoryComingBackIsTheTranscriptsUntilItIsDone );

function aThoughtBeingWrittenCountsItsTokens()
{
    const thought = ( tokens ) => ( { type: 'turn.text', turn_id: 't1', round: 1, channel: 'thought', text: 'x', tokens } );
    const state = apply( [ begin, thought( 3 ), thought( 412 ) ] );
    const block = state.transcript[ state.transcript.length - 1 ].blocks[ 0 ];

    assert.equal( thoughtLabel( block, true ), 'Thinking · 412 tokens' );
    assert.equal( thoughtLabel( { ...block, liveTokens: null }, true ), 'Thinking' );
    assert.equal( thoughtLabel( block, false ), 'Thought' );
}

test( 'a thought being written counts its tokens', aThoughtBeingWrittenCountsItsTokens );

function theHeaderSaysWhatTheAppIsDoingAndNothingWhenItIsDone()
{
    const loading = { type: 'progress', task: 'load', label: 'Loading gemma-4-12b-it-q4_0', done: null, total: null };
    const restoring = { type: 'progress', task: 'restore', label: 'Getting ready', done: 0, total: 4096 };
    const installing = { type: 'progress', task: 'install', label: 'Installing llama-3.2-3b', done: 10, total: 100 };

    assert.equal( appStatus( apply( [] ) ), null );
    assert.equal( appStatus( apply( [ loading ] ) ), 'Loading' );
    assert.equal( appStatus( apply( [ installing ] ) ), 'Installing llama-3.2-3b' );
    assert.equal( appStatus( apply( [ loading, { ...loading, done: 1, total: 1 } ] ) ), null );
    assert.equal( appStatus( apply( [ restoring ] ) ), 'Getting ready' );
    assert.equal( appStatus( apply( [ restoring, { ...restoring, done: 4096 } ] ) ), null );
}

test( 'the header says what the app is doing, and nothing when it is done', theHeaderSaysWhatTheAppIsDoingAndNothingWhenItIsDone );

function screenFor( session, installed )
{
    const known = installed !== null;
    const state = { session, modelsKnown: known, installedModels: known ? installed : [] };

    return emptyScreen( state );
}

function anEmptyConversationShowsTheWayToAModel()
{
    const missing = { model: null, loading: null };
    const loading = { model: null, loading: 'gemma-4-12b-it-q4_0' };
    const one = [ { name: 'gemma-4-12b-it-q4_0', size_bytes: 7e9, quantization: 'q4_0' } ];

    assert.equal( screenFor( null, null ), null );
    assert.equal( screenFor( missing, null ), null, 'nothing until the session has listed its models' );
    assert.equal( screenFor( missing, [] ), 'first-run' );
    assert.equal( screenFor( missing, one ), 'choose-model' );
    assert.equal( screenFor( loading, [] ), 'home', 'a model being loaded is not said to be missing' );
}

test( 'an empty conversation shows the way to a model', anEmptyConversationShowsTheWayToAModel );

function theModelsListSaysTheInstalledModelsAreKnown()
{
    const listed = { type: 'models.list', installed: [], published: null };

    assert.equal( apply( [] ).modelsKnown, false );
    assert.equal( apply( [ listed ] ).modelsKnown, true );
}

test( 'the models list says the installed models are known', theModelsListSaysTheInstalledModelsAreKnown );

function aRefusedLoadBeforeAConversationIsSaidWhereTheModelsAre()
{
    const refused = { type: 'error', text: 'Could not load qwen3.8-27b-fp4: too large.', during: 'load' };
    const before = apply( [ refused ] );
    const during = apply( [ begin, refused ] );

    assert.equal( before.transcript.length, 0 );
    assert.equal( before.modelsError.text, refused.text );
    assert.equal( during.transcript[ during.transcript.length - 1 ].kind, 'error' );
    assert.equal( during.modelsError.text, refused.text );
}

test( 'a refused load before a conversation is said where the models are', aRefusedLoadBeforeAConversationIsSaidWhereTheModelsAre );

function aFirstRunAsksThePublisherAndShowsWhatItAnswered()
{
    const asked = { type: 'models.refresh', online: true };
    const gemma = { name: 'gemma-4-12b-it-fp4', fits: true };
    const qwen = { name: 'qwen3.8-27b-fp4', fits: false };
    const suggesting = { type: 'models.list', installed: [], published: [ gemma, qwen ], suggested: gemma.name };
    const nothing = { type: 'models.list', installed: [], published: [ qwen ], suggested: null };
    const failed = { type: 'error', text: 'Could not list the models that can be installed: offline.', during: 'models' };
    const empty = { type: 'models.list', installed: [], published: null, suggested: null };

    assert.equal( firstRunStep( apply( [] ) ), 'not-asked' );
    assert.equal( firstRunStep( apply( [ asked ] ) ), 'asking' );
    assert.equal( firstRunStep( apply( [ asked, suggesting ] ) ), 'suggested' );
    assert.deepEqual( suggestedModel( apply( [ asked, suggesting ] ) ), gemma );
    assert.equal( firstRunStep( apply( [ asked, nothing ] ) ), 'none-fits' );

    const unreachable = apply( [ asked, failed, empty ] );
    assert.equal( firstRunStep( unreachable ), 'unreachable' );
    assert.equal( unreachable.transcript.length, 0, 'the failure is said on the screen that asked' );
    assert.equal( firstRunStep( reduce( unreachable, asked ) ), 'asking', 'asking again clears it' );
}

test( 'a first run asks the publisher and shows what it answered', aFirstRunAsksThePublisherAndShowsWhatItAnswered );

function whatAnInstallOrLoadSaysKeepsAnEmptyConversationEmpty()
{
    const installed = { type: 'info', text: 'Installed gemma-4-12b-it-fp4 (6.8 GB).', during: 'install' };
    const loaded = { type: 'info', text: 'Model loaded.', during: 'install' };
    const other = { type: 'info', text: 'Sampler reseeded (7).' };

    assert.equal( apply( [ installed, loaded ] ).transcript.length, 0 );
    assert.equal( apply( [ other ] ).transcript.length, 1 );
    assert.equal( apply( [ begin, installed ] ).transcript.length, 2, 'once a conversation has begun it is a line of it' );
}

test( 'what an install or a load says keeps an empty conversation empty', whatAnInstallOrLoadSaysKeepsAnEmptyConversationEmpty );

function aSuggestedModelIsDescribedByWhatItCanDoHere()
{
    const agent = { agentic_here: true, thinking: true, modalities: [ 'image', 'audio' ], size_bytes: 6.8e9, context_length: 131072 };
    const plain = { agentic_here: false, thinking: false, modalities: [], size_bytes: 3.1e9, context_length: null };

    assert.equal( modelAbilities( agent ), 'Works through tasks with tools, thinks before it answers, reads images and audio.' );
    assert.equal( modelAbilities( plain ), 'Answers questions and holds a conversation.' );
    assert.equal( fitLine( agent ), 'Runs on this computer with room for long tasks.' );
    assert.equal( fitLine( plain ), null );
    assert.equal( modelTitle( { name: 'gemma-4-12b-it-q4_0', display_name: 'Gemma 4 12B' } ), 'Gemma 4 12B' );
    assert.equal( modelTitle( { name: 'Llama-3.2-3B-Instruct-fp4', display_name: '' } ), 'Llama-3.2-3B-Instruct-fp4',
        'a manifest written before the field shows its name' );

    const gemma = { name: 'gemma-4-12b-it-q4_0', display_name: 'Gemma 4 12B' };
    assert.equal( namedTitle( { installedModels: [ gemma ], publishedModels: null }, gemma.name ), 'Gemma 4 12B' );
    assert.equal( namedTitle( { installedModels: [], publishedModels: [ gemma ] }, gemma.name ), 'Gemma 4 12B',
        'a model just installed is named before the installed list arrives' );
    assert.equal( namedTitle( { installedModels: [], publishedModels: null }, 'qwen3.8-27b-fp4' ), 'qwen3.8-27b-fp4' );
}

test( 'a suggested model is described by what it can do here', aSuggestedModelIsDescribedByWhatItCanDoHere );

function aScreenAskingForAModelIsTheOneWayForward()
{
    const missing = { model: null, loading: null };
    const loaded = { model: 'gemma-4-12b-it-q4_0', loading: null };
    const firstRun = { session: missing, modelsKnown: true, installedModels: [], transcript: [] };
    const refusedMidConversation = { ...firstRun, transcript: [ { kind: 'user', text: 'Hello' } ] };
    const home = { ...firstRun, session: loaded };

    assert.equal( isAskingForModel( firstRun ), true );
    assert.equal( isAskingForModel( refusedMidConversation ), false, 'a conversation keeps its menu and box' );
    assert.equal( isAskingForModel( home ), false );
}

test( 'a screen asking for a model is the one way forward', aScreenAskingForAModelIsTheOneWayForward );

const licenseText = { type: 'license', text: 'THE SOFTWARE IS PROVIDED "AS IS"', sha256: 'digest', accepted: false, problem: null };
const licenseAccepted = { ...licenseText, accepted: true };

function setupState( messages )
{
    const missing = { ...sessionState( null, null ), loading: null };
    const listed = { type: 'models.list', installed: [], published: null, suggested: null };

    return apply( [ missing, listed, ...messages ] );
}

function aFirstRunSetupHasTheWholeWindow()
{
    const firstRun = setupState( [ licenseAccepted ] );
    const installed = reduce( firstRun, { type: 'models.list', installed: [ { name: 'gemma-4-12b-it-q4_0' } ], published: null } );
    const loading = reduce( installed, { ...sessionState( null, null ), loading: 'gemma-4-12b-it-q4_0' } );
    const midConversation = reduce( firstRun, { type: 'turn.send', text: 'Hello', attachment_ids: [] } );

    assert.equal( isWelcoming( firstRun ), true );
    assert.equal( isAskingForModel( firstRun ), true, 'no message box beside the setup' );
    assert.equal( isWelcoming( installed ), false, 'choosing among installed models keeps the sidebar' );
    assert.equal( isWelcoming( loading ), false, 'the installed model loading brings the window back' );
    assert.equal( isWelcoming( midConversation ), false );
    assert.equal( isWelcoming( setupState( [] ) ), false, 'nothing until the session says where the licence stands' );
}

test( 'a first-run setup has the whole window', aFirstRunSetupHasTheWholeWindow );

function theLicenceComesBeforeAnyModel()
{
    const welcome = setupState( [ licenseText ] );
    const license = reduce( welcome, { type: 'setup.continue' } );

    assert.equal( setupPane( welcome ), 'welcome' );
    assert.equal( setupPane( license ), 'license' );
    assert.equal( setupPane( reduce( license, { type: 'setup.continue' } ) ), 'license', 'Continue does not pass an unaccepted licence' );
    assert.equal( setupPane( reduce( license, { type: 'setup.back' } ) ), 'welcome' );

    const read = reduce( license, { type: 'license.read' } );
    assert.equal( read.licenseRead, true );

    const accepting = reduce( read, { type: 'license.accept', sha256: 'digest' } );
    assert.equal( setupPane( accepting ), 'license', 'the pane waits for the session to say it is accepted' );

    const accepted = reduce( accepting, licenseAccepted );
    assert.equal( setupPane( accepted ), 'model' );
    assert.equal( setupPane( reduce( accepted, { type: 'setup.back' } ) ), 'license', 'Go Back returns to the licence read' );

    const refused = reduce( accepting, { type: 'error', text: 'The licence shown is not the one Mila carries.', during: 'license' } );
    assert.equal( setupPane( refused ), 'license' );
    assert.equal( refused.licenseError, 'The licence shown is not the one Mila carries.' );
    assert.equal( refused.transcript.length, 0 );
}

test( 'the licence comes before any model', theLicenceComesBeforeAnyModel );

function aLicenceAcceptedBeforeSkipsItsPane()
{
    const welcome = setupState( [ licenseAccepted ] );
    const model = reduce( welcome, { type: 'setup.continue' } );

    assert.equal( setupPane( model ), 'model' );
    assert.equal( setupPane( reduce( model, { type: 'setup.back' } ) ), 'welcome', 'a licence not read on this run is not shown again' );

    const installedBefore = reduce( setupState( [ licenseText ] ), { type: 'models.list', installed: [ { name: 'gemma-4-12b-it-q4_0' } ], published: null } );
    assert.equal( setupPane( installedBefore ), 'welcome', 'a licence not yet accepted is asked for with models installed' );
    assert.equal( setupPane( reduce( installedBefore, licenseAccepted ) ), null, 'then the installed models are chosen among' );
}

test( 'a licence accepted before skips its pane', aLicenceAcceptedBeforeSkipsItsPane );

function aModelInstalledFromTheSetupEndsOnItsConclusion()
{
    const model = reduce( setupState( [ licenseAccepted ] ), { type: 'setup.continue' } );
    const installing = reduce( model, { type: 'model.install', name: 'gemma-4-12b-it-fp4' } );
    const loading = reduce( installing, { ...sessionState( null, null ), loading: 'gemma-4-12b-it-fp4' } );
    const loaded = reduce( loading, { ...sessionState( 'gemma-4-12b-it-fp4', null ), loading: null } );

    assert.equal( installing.setupInstalling, true );
    assert.equal( setupPane( loading ), 'model', 'the load after the install is the model pane’s' );
    assert.equal( setupPane( loaded ), 'conclusion' );
    assert.equal( isAskingForModel( loaded ), true, 'no message box beside the conclusion' );
    assert.equal( setupPane( reduce( loaded, { type: 'setup.done' } ) ), null );

    const refused = reduce( installing, { type: 'error', text: 'Could not load gemma-4-12b-it-fp4: too large.', during: 'load' } );
    assert.equal( setupPane( refused ), null, 'a model refused at its load is chosen among the installed' );
}

test( 'a model installed from the setup ends on its conclusion', aModelInstalledFromTheSetupEndsOnItsConclusion );
