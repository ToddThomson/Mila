// Going somewhere by asking (AssistantInterface.md 5.2): the pane follows turn.went as a click would take it, the call
// keeps the line saying where Mila went, a chat opens once the reply ends, several places in an exposed chat are the
// person's to choose, and the mock's scenario 18 answers as the session would.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createInitialState, reduce, barMode } from '../state.js';
import { createMockSession } from '../mock/session.js';
import { createClock } from './clock.js';

const now = Date.UTC( 2026, 10, 20, 15, 0, 0 );

function apply( messages, state = createInitialState() )
{
    for ( const message of messages )
        state = reduce( state, message );

    return state;
}

const project = { v: 1, id: 'p-calc', name: 'Calculator app', purpose: '', instructions: '', folders: [],
    created: '2026-11-10T10:00:00.000Z', updated: '2026-11-10T10:00:00.000Z' };

/** A turn under way from the bar over the workspace, with one open_place call announced. */
function turnWithACall( place = 'workspace' )
{
    return apply( [
        { type: 'projects.list', items: [ project ] },
        { type: 'session.state', model: 'm', busy: true, streams: [ { stream_id: 'stream-1', conversation_id: 'c-bar' } ] },
        { type: 'place.open', place },
        { type: 'turn.send', stream_id: 'stream-1', text: 'Open the calculator project', attachment_ids: [] },
        { type: 'turn.begin', stream_id: 'stream-1', turn_id: 'turn-1', conversation_id: 'c-bar' },
        { type: 'turn.tool_call', stream_id: 'stream-1', turn_id: 'turn-1', round: 1, call_id: 'call-1', name: 'open_place',
            summary: 'Going to calculator', arguments: '{"place":"calculator"}' } ] );
}

function toolBlock( state )
{
    return state.transcript.find( ( entry ) => entry.kind === 'assistant' ).blocks.find( ( block ) => block.kind === 'tool' );
}

function thePaneGoesToTheProjectMilaOpenedAndTheCallSaysSo()
{
    const state = reduce( turnWithACall(), { type: 'turn.went', stream_id: 'stream-1', turn_id: 'turn-1', call_id: 'call-1',
        place: { view: 'project', id: 'p-calc' }, text: 'Opened Calculator app' } );

    assert.equal( state.place, 'project' );
    assert.equal( state.projectId, 'p-calc' );
    assert.equal( state.risen, true, 'the conversation went out of sight' );
    assert.deepEqual( toolBlock( state ).went, { place: { view: 'project', id: 'p-calc' }, text: 'Opened Calculator app' } );
}

test( 'the pane goes to the project Mila opened, the conversation stays risen, and the call says where it went', thePaneGoesToTheProjectMilaOpenedAndTheCallSaysSo );

function aViewMilaOpenedIsOpenedAndAnUnknownOneIsNot()
{
    const went = ( view ) => reduce( turnWithACall(), { type: 'turn.went', stream_id: 'stream-1', turn_id: 'turn-1',
        call_id: 'call-1', place: { view, id: null }, text: 'Opened' } );

    assert.equal( went( 'projects' ).place, 'projects' );
    assert.equal( went( 'activity' ).place, 'activity' );
    assert.equal( went( 'reach' ).place, 'reach' );
    assert.equal( went( 'settings' ).place, 'workspace' );
}

test( 'a view Mila opened is opened, and one the page does not have is not', aViewMilaOpenedIsOpenedAndAnUnknownOneIsNot );

function theBarGoesOnWithTheConversationInTheProjectItOpened()
{
    const went = apply( [
        { type: 'turn.went', stream_id: 'stream-1', turn_id: 'turn-1', call_id: 'call-1', place: { view: 'project', id: 'p-calc' },
            text: 'Opened Calculator app' },
        { type: 'session.state', model: 'm', busy: false, project: null, streams: [ { stream_id: 'stream-1', conversation_id: 'c-bar' } ] } ],
        turnWithACall() );

    assert.equal( barMode( went ), 'continue', 'the next question began a chat in the project Mila opened' );

    // A project the person clicks into is still the place a new chat begins in.
    const other = { ...project, id: 'p-other', name: 'Other' };
    const clicked = apply( [ { type: 'projects.list', items: [ project, other ] }, { type: 'place.open', place: 'project', project: 'p-other' } ], went );

    assert.equal( barMode( clicked ), 'start' );
    assert.equal( reduce( went, { type: 'conversation.new' } ).barWentTo, null );
}

test( 'the bar goes on with the conversation in the project Mila opened for it, and not in one clicked into', theBarGoesOnWithTheConversationInTheProjectItOpened );

function askedFromTheWholeChatTheChatRisesOverThePlace()
{
    const state = reduce( turnWithACall( 'chat' ), { type: 'turn.went', stream_id: 'stream-1', turn_id: 'turn-1',
        call_id: 'call-1', place: { view: 'projects', id: null }, text: 'Opened All projects' } );

    assert.equal( state.place, 'projects' );
    assert.equal( state.risen, true );
}

test( 'asked from the whole chat, the chat rises over the place Mila opened', askedFromTheWholeChatTheChatRisesOverThePlace );

function anotherChatOpensOnceTheReplyEndsAndTheOneReplyingStays()
{
    const other = reduce( turnWithACall(), { type: 'turn.went', stream_id: 'stream-1', turn_id: 'turn-1', call_id: 'call-1',
        place: { view: 'chat', id: 'c-kyoto' }, text: 'Opened Kyoto' } );

    assert.equal( other.wentToChat, 'c-kyoto' );
    assert.equal( other.place, 'workspace', 'the pane left the reply before it ended' );
    assert.equal( reduce( other, { type: 'went.chat_opened' } ).wentToChat, null );

    const same = reduce( turnWithACall(), { type: 'turn.went', stream_id: 'stream-1', turn_id: 'turn-1', call_id: 'call-1',
        place: { view: 'chat', id: 'c-bar' }, text: 'Opened this chat' } );

    assert.equal( same.wentToChat, null );
}

test( 'another chat Mila opened opens once the reply ends, and the one replying stays', anotherChatOpensOnceTheReplyEndsAndTheOneReplyingStays );

function aChatOpenedAgainKeepsWhereMilaWent()
{
    const state = apply( [ { type: 'conversation.loaded', id: 'c-bar', messages: [
        { role: 'user', content: 'Open the calculator project', attachments: null },
        { role: 'assistant', content: 'Opened it.', thought: null, stats: null, tool_calls: [ { call_id: 'call-1', name: 'open_place',
            summary: 'Going to calculator', arguments: '{"place":"calculator"}', result: 'Opened the project Calculator app.',
            is_error: false, decision: null, went: { place: { view: 'project', id: 'p-calc' }, text: 'Opened Calculator app' } } ] } ] } ] );

    assert.equal( toolBlock( state ).went.text, 'Opened Calculator app' );
    assert.equal( state.place, 'workspace', 'opening a stored chat went somewhere again' );
}

test( 'a chat opened again keeps where Mila went, and goes nowhere', aChatOpenedAgainKeepsWhereMilaWent );

function severalPlacesInAnExposedChatAreThePersonsToChoose()
{
    const state = reduce( turnWithACall(), { type: 'turn.tool_approval', stream_id: 'stream-1', turn_id: 'turn-1', call_id: 'call-1',
        choices: [ 'allow_once', 'deny' ], places: [ { view: 'project', id: 'p-calc', name: 'Calculator app' },
            { view: 'chat', id: 'c-keys', name: 'Keys of the calculator' } ] } );

    assert.deepEqual( toolBlock( state ).places.map( ( place ) => place.name ), [ 'Calculator app', 'Keys of the calculator' ] );
    assert.deepEqual( toolBlock( reduce( turnWithACall(), { type: 'turn.tool_approval', stream_id: 'stream-1', turn_id: 'turn-1',
        call_id: 'call-1', choices: [ 'allow_once', 'deny' ] } ) ).places, [] );
}

test( 'several places that fit in an exposed chat are the person\'s to choose', severalPlacesInAnExposedChatAreThePersonsToChoose );

// ---- The mock, as the session answers ---------------------------------------------------------

function play( text )
{
    const clock = createClock( now );
    const events = [];
    const session = createMockSession( { scenario: 18, emit: ( message ) => events.push( message ), schedule: clock.schedule, cancel: clock.cancel, now: clock.now } );

    session.handle( { v: 1, type: 'ready' } );
    session.handle( { v: 1, type: 'turn.send', stream_id: 'stream-1', text, attachment_ids: [] } );
    clock.run();

    return events;
}

function scenario18OpensTheProjectAskedForWithoutAsking()
{
    const events = play( 'Open the calculator project' );
    const went = events.find( ( event ) => event.type === 'turn.went' );

    assert.deepEqual( went.place, { view: 'project', id: 'project-mock-calculator' } );
    assert.equal( events.some( ( event ) => event.type === 'turn.tool_approval' ), false, 'the person was asked' );

    const state = apply( events.map( ( event ) => ( { ...event } ) ) );

    assert.equal( state.place, 'project' );
    assert.equal( state.projectId, 'project-mock-calculator' );
}

test( 'scenario 18 opens the project asked for, without asking', scenario18OpensTheProjectAskedForWithoutAsking );

function scenario18FindsWhatAnEarlierChatSaidAndNamesIt()
{
    const events = play( 'What did we decide about Fushimi Inari earlier?' );
    const call = events.find( ( event ) => event.type === 'turn.tool_call' );
    const end = events.find( ( event ) => event.type === 'turn.end' );

    assert.equal( call.name, 'find_records' );
    assert.equal( events.some( ( event ) => event.type === 'turn.went' ), false );
    assert.match( end.answer, /Plan a three-day trip to Kyoto/ );

    const stored = events.filter( ( event ) => event.type === 'conversations.list' ).at( -1 ).items;

    assert.ok( stored.some( ( item ) => item.title.startsWith( 'What did we decide' ) ) );
}

test( 'scenario 18 finds what an earlier chat said and names the chat', scenario18FindsWhatAnEarlierChatSaidAndNamesIt );
