// Mila proposes structure (AssistantInterface.md 5.4 and 6): an offer is kept on the call that made it and drawn from
// it, the person's words answer it without a reply from Mila, the overview's card answers it with the name as edited,
// a project made from one is undone while nothing has changed it, and the mock's scenario 19 answers as the session would.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createInitialState, reduce, suggestions, chatsTogether, workspaceOverview } from '../state.js';
import { barPrompt, gatheredLine } from '../format.js';
import { readOfferAnswer } from '../mock/offers.js';
import { createMockSession } from '../mock/session.js';
import { createClock } from './clock.js';

const now = Date.UTC( 2026, 11, 21, 15, 0, 0 );

function apply( messages, state = createInitialState() )
{
    for ( const message of messages )
        state = reduce( state, message );

    return state;
}

const offer = { kind: 'project', state: 'open', name: 'Garden', purpose: 'Growing vegetables', in_window: false,
    chats: [ { id: 'c-beds', title: 'Raised beds' }, { id: 'c-soil', title: 'Soil' } ], time: '2026-12-21T14:00:00.000Z' };

function chatItem( id, title, offers = [], project = null )
{
    return { id, title, created: '2026-12-21T10:00:00.000Z', updated: '2026-12-21T14:00:00.000Z', project, offers };
}

function turnWithAnOffer()
{
    return apply( [
        { type: 'session.state', model: 'm', busy: true, streams: [ { stream_id: 'stream-1', conversation_id: 'c-beds' } ] },
        { type: 'turn.begin', turn_id: 'turn-1', conversation_id: 'c-beds' },
        { type: 'turn.tool_call', turn_id: 'turn-1', round: 1, call_id: 'call-1', name: 'propose_project',
            summary: 'Suggesting a project: Garden', arguments: '{}' },
        { type: 'turn.offer', turn_id: 'turn-1', call_id: 'call-1', offer },
        { type: 'turn.end', turn_id: 'turn-1', finish: 'stop', answer: 'Shall I?', message: 1 } ] );
}

function anOfferIsKeptOnItsCallAndAnsweredWordsLeaveTheAppsLine()
{
    const offered = turnWithAnOffer();
    const call = offered.transcript[ 0 ].blocks.find( ( block ) => block.kind === 'tool' );

    assert.deepEqual( call.offer, offer );

    const answered = apply( [
        { type: 'turn.send', text: 'yes', attachment_ids: [] },
        { type: 'turn.begin', turn_id: 'turn-2', conversation_id: 'c-beds' },
        { type: 'turn.answered', turn_id: 'turn-2', call_id: 'call-1', offer: { ...offer, state: 'accepted', project: 'p-garden' },
            text: 'Made the project Garden and moved this chat and 1 other into it.' },
        { type: 'turn.end', turn_id: 'turn-2', finish: 'stop', answer: '', message: 3 } ], offered );

    const line = answered.transcript.at( -1 );

    assert.equal( line.app, true );
    assert.deepEqual( line.blocks, [ { kind: 'app', id: 'turn-2:app', text: 'Made the project Garden and moved this chat and 1 other into it.' } ] );
    assert.equal( answered.transcript[ 0 ].blocks.find( ( block ) => block.kind === 'tool' ).offer.state, 'accepted' );
}

test( 'an offer is kept on its call, and words that answer it leave the app\'s line', anOfferIsKeptOnItsCallAndAnsweredWordsLeaveTheAppsLine );

function aStoredChatOpensWithItsOffersAndTheAppsLine()
{
    const state = reduce( createInitialState(), { type: 'conversation.loaded', id: 'c-beds', messages: [
        { role: 'user', content: 'Raised beds?', attachments: null },
        { role: 'assistant', content: 'Shall I?', thought: null, stats: null, tool_calls: [ { call_id: 'call-1', name: 'propose_project',
            summary: 'Suggesting a project: Garden', arguments: '{}', result: 'Offered', is_error: false, decision: null, offer } ] },
        { role: 'user', content: 'yes', attachments: null },
        { role: 'assistant', content: 'Made the project Garden.', thought: null, stats: null, tool_calls: null, app: true } ] } );

    assert.deepEqual( state.transcript[ 1 ].blocks[ 0 ].offer, offer );
    assert.equal( state.transcript[ 3 ].app, true );
    assert.equal( state.transcript[ 3 ].answer, '', 'the app\'s line is no reply to keep' );
    assert.deepEqual( state.transcript[ 3 ].blocks.map( ( block ) => block.kind ), [ 'app' ] );
}

test( 'a stored chat opens with its offers and the app\'s line', aStoredChatOpensWithItsOffersAndTheAppsLine );

function anOfferAnsweredOnTheCardReachesTheOpenChat()
{
    const offered = turnWithAnOffer();
    const listed = reduce( offered, { type: 'conversations.list', items: [
        chatItem( 'c-beds', 'Raised beds', [ { ...offer, state: 'declined', message: 1, call_id: 'call-1' } ] ) ] } );

    assert.equal( listed.transcript[ 0 ].blocks.find( ( block ) => block.kind === 'tool' ).offer.state, 'declined' );

    const again = reduce( listed, { type: 'conversations.list', items: listed.conversations } );

    assert.equal( again.transcript, listed.transcript, 'an offer unchanged leaves the transcript as it was' );
}

test( 'an offer answered on the overview\'s card reaches the open chat', anOfferAnsweredOnTheCardReachesTheOpenChat );

function theOverviewShowsEachSuggestionWaitingAndEachProjectMadeThatCanBeUndone()
{
    const state = apply( [
        { type: 'projects.list', items: [ { id: 'p-garden', name: 'Garden', folders: [], created: '2026-12-21T13:00:00.000Z',
            updated: '2026-12-21T13:00:00.000Z', undo: true }, { id: 'p-old', name: 'Old', folders: [], undo: false } ] },
        { type: 'conversations.list', items: [
            chatItem( 'c-beds', 'Raised beds', [ { ...offer, message: 1, call_id: 'call-1', state: 'undecided' } ] ),
            chatItem( 'c-kept', 'Made', [ { ...offer, message: 1, call_id: 'call-1', state: 'accepted', project: 'p-garden',
                time: '2026-12-21T12:00:00.000Z' } ], 'p-garden' ),
            chatItem( 'c-old', 'Old', [ { ...offer, message: 1, call_id: 'call-1', state: 'accepted', project: 'p-old' } ], 'p-old' ),
            chatItem( 'c-no', 'No', [ { ...offer, message: 1, call_id: 'call-1', state: 'declined' } ] ),
            chatItem( 'c-move', 'Move', [ { kind: 'move', state: 'open', name: 'Garden', project: 'p-garden', message: 1, call_id: 'call-1' } ] ) ] } ] );

    const rows = suggestions( state );

    assert.deepEqual( rows.map( ( row ) => [ row.kind, row.item.id ] ), [ [ 'suggestion', 'c-beds' ], [ 'made', 'c-kept' ] ] );
    assert.equal( rows[ 1 ].project.id, 'p-garden' );
    assert.deepEqual( [ ...chatsTogether( state ) ].sort(), [ 'c-beds', 'c-soil' ] );
    assert.equal( workspaceOverview( state ).suggested.length, 2 );
    assert.equal( workspaceOverview( state ).waiting.length, 0, 'a suggestion waits on nobody' );
}

test( 'the overview shows each suggestion waiting, and each project made from one that can be undone', theOverviewShowsEachSuggestionWaitingAndEachProjectMadeThatCanBeUndone );

function newProjectInTheSidebarAsksMilaUntilSomethingIsSent()
{
    const asking = apply( [ { type: 'conversation.new' }, { type: 'project.ask' } ] );

    assert.equal( asking.askingForProject, true );
    assert.equal( barPrompt( 'workspace', null, asking.askingForProject ), 'Tell Mila what the project is for' );
    assert.equal( barPrompt( 'chat', null, true ), 'Reply to Mila' );
    assert.equal( reduce( asking, { type: 'turn.send', text: 'A garden', attachment_ids: [] } ).askingForProject, false );
    assert.equal( reduce( asking, { type: 'place.open', place: 'activity' } ).askingForProject, false );
}

test( 'New project in the sidebar asks Mila, until something is sent', newProjectInTheSidebarAsksMilaUntilSomethingIsSent );

function aSuggestionNamesTheChatsItGathersAndThisChatInTheOneThatMadeIt()
{
    const gathering = { chats: [ { id: 'c-here', title: 'Load CSV' }, { id: 'c-b', title: 'Compare' }, { id: 'c-c', title: 'Plot' } ] };

    assert.equal( gatheredLine( gathering ), 'Gathers “Load CSV”, “Compare” and “Plot”' );
    assert.equal( gatheredLine( gathering, 'c-here' ), 'Gathers this chat, “Compare” and “Plot”' );
    assert.equal( gatheredLine( { chats: [ gathering.chats[ 0 ], gathering.chats[ 1 ] ] }, 'c-here' ), 'Gathers this chat and “Compare”' );
    assert.equal( gatheredLine( { chats: [ { id: 'c-here', title: 'Load CSV' } ] }, 'c-here' ), 'Gathers this chat' );
}

test( 'a suggestion names the chats it gathers, and this chat in the one that made it', aSuggestionNamesTheChatsItGathersAndThisChatInTheOneThatMadeIt );

function theMockReadsAnswersAsTheSessionDoes()
{
    for ( const said of [ 'yes', 'Yes, please!', 'OK', 'sure.', 'please do', 'Go ahead', '  yep  ', 'Sounds good.' ] )
        assert.equal( readOfferAnswer( said ), 'accept', said );

    for ( const said of [ 'no', 'Not now', 'No thanks.', 'don\'t', 'don’t', 'nope', 'Maybe later' ] )
        assert.equal( readOfferAnswer( said ), 'decline', said );

    for ( const said of [ 'Keep it separate', 'undo', 'Put it back.' ] )
        assert.equal( readOfferAnswer( said ), 'keep_separate', said );

    for ( const said of [ 'yes, but call it Garden', '', 'What about the tomatoes?', 'no idea what you mean', 'ok so how do I water them', 'yesterday' ] )
        assert.equal( readOfferAnswer( said ), 'other', said );
}

test( 'the mock reads an answer to an offer as the session does', theMockReadsAnswersAsTheSessionDoes );

// ---- The mock, as the session answers ---------------------------------------------------------

function startScenario19()
{
    const clock = createClock( now );
    const events = [];
    const session = createMockSession( { scenario: 19, emit: ( message ) => events.push( message ), schedule: clock.schedule, cancel: clock.cancel, now: clock.now } );

    session.handle( { v: 1, type: 'ready' } );

    const say = ( text ) =>
    {
        const before = events.length;

        session.handle( { v: 1, type: 'turn.send', stream_id: 'stream-1', text, attachment_ids: [] } );
        clock.run();

        return events.slice( before );
    };

    const latest = ( type ) => events.filter( ( event ) => event.type === type ).at( -1 );

    return { session, events, say, latest };
}

function scenario19OffersAProjectAndYesMakesItWithNoReply()
{
    const { say, latest, session } = startScenario19();
    const offered = say( 'Help me load my CSV files into DuckDB' );
    const offerEvent = offered.find( ( event ) => event.type === 'turn.offer' );

    assert.equal( offered.some( ( event ) => event.type === 'turn.tool_approval' ), false, 'an offer is not an approval' );
    assert.equal( offerEvent.offer.name, 'CSV analytics' );

    const asking = offerEvent.offer.chats[ 0 ].id;
    const listed = latest( 'conversations.list' ).items.find( ( item ) => item.id === asking );

    assert.equal( listed.offers[ 0 ].state, 'open' );
    assert.equal( listed.offers[ 0 ].call_id, offerEvent.call_id );

    const answered = say( 'yes' );

    assert.equal( answered.some( ( event ) => event.type === 'turn.text' ), false, 'Mila writes no reply to an answer' );
    assert.equal( answered.find( ( event ) => event.type === 'turn.answered' ).text,
        'Made the project CSV analytics and moved this chat and 1 other into it.' );

    const made = latest( 'projects.list' ).items.find( ( project ) => project.name === 'CSV analytics' );

    assert.equal( made.undo, true );

    const chats = latest( 'conversations.list' ).items;

    assert.equal( chats.find( ( item ) => item.id === asking ).project, made.id );
    assert.equal( chats.find( ( item ) => item.id === 'conversation-7' ).project, made.id );
    assert.equal( chats.find( ( item ) => item.id === asking ).offers[ 0 ].state, 'accepted' );

    // Undone while nothing has changed it: the project goes and its chats return to no project.
    session.handle( { v: 1, type: 'project.undo', id: made.id } );

    assert.equal( latest( 'projects.list' ).items.some( ( project ) => project.id === made.id ), false );
    assert.equal( latest( 'conversations.list' ).items.find( ( item ) => item.id === 'conversation-7' ).project, null );
    assert.equal( latest( 'conversations.list' ).items.find( ( item ) => item.id === asking ).offers[ 0 ].state, 'undone' );
    assert.ok( latest( 'activity.list' ).items.some( ( event ) => event.kind === 'project.undone' ) );
}

test( 'scenario 19 offers a project, and yes makes it with no reply from Mila', scenario19OffersAProjectAndYesMakesItWithNoReply );

function scenario19NotNowDeclinesAndAnythingElseLeavesTheCard()
{
    const declined = startScenario19();
    declined.say( 'Help me load my CSV files into DuckDB' );
    declined.say( 'Not now' );

    assert.equal( declined.latest( 'conversations.list' ).items.find( ( item ) => item.offers.length > 0 ).offers[ 0 ].state, 'declined' );
    assert.ok( declined.latest( 'activity.list' ).items.some( ( event ) => event.kind === 'suggestion.declined' ) );

    const waiting = startScenario19();
    waiting.say( 'Help me load my CSV files into DuckDB' );

    const replied = waiting.say( 'And how do I query them?' );

    assert.ok( replied.some( ( event ) => event.type === 'turn.text' ), 'anything else goes to Mila' );

    const state = apply( waiting.events.map( ( event ) => ( { ...event } ) ) );
    const rows = suggestions( state ).filter( ( row ) => row.kind === 'suggestion' );

    assert.equal( rows.length, 1, 'the offer waits on the card' );
    assert.equal( rows.at( -1 ).offer.state, 'undecided' );

    // Made from the card with the name the person typed.
    const first = rows.at( -1 );
    waiting.session.handle( { v: 1, type: 'suggestion.decide', conversation_id: first.item.id, call_id: first.offer.call_id,
        decision: 'accept', name: '  Analytics  ' } );

    assert.ok( waiting.latest( 'projects.list' ).items.some( ( project ) => project.name === 'Analytics' ) );

    const before = waiting.events.length;
    waiting.session.handle( { v: 1, type: 'suggestion.decide', conversation_id: first.item.id, call_id: first.offer.call_id, decision: 'decline' } );

    assert.equal( waiting.events.slice( before ).find( ( event ) => event.type === 'error' ).text, 'That suggestion has already been answered.' );
}

test( 'scenario 19: not now declines, and anything else leaves the offer on the overview\'s card', scenario19NotNowDeclinesAndAnythingElseLeavesTheCard );

function scenario19OffersAMoveTheFirstTimeAndYesMovesTheChat()
{
    const { say, latest } = startScenario19();
    const offered = say( 'This belongs with the calculator' );
    const offerEvent = offered.find( ( event ) => event.type === 'turn.offer' );

    assert.equal( offerEvent.offer.kind, 'move' );
    assert.equal( offerEvent.offer.state, 'open' );

    const answered = say( 'Sure' );

    assert.equal( answered.find( ( event ) => event.type === 'turn.answered' ).text, 'Moved this chat into Calculator app.' );

    const moved = latest( 'activity.list' ).items.find( ( event ) => event.kind === 'chat.moved' && event.suggested === 'move' );

    assert.equal( moved.project, 'project-mock-calculator' );
    assert.equal( latest( 'conversations.list' ).items.find( ( item ) => item.id === moved.conversation ).project, 'project-mock-calculator' );
}

test( 'scenario 19 offers a move the first time for a project, and yes moves the chat', scenario19OffersAMoveTheFirstTimeAndYesMovesTheChat );
