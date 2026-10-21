import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createMockSession } from '../mock/session.js';
import { createInitialState, reduce } from '../state.js';
import { titleFor, splitIntoTokens } from '../mock/fixtures.js';
import { createClock } from './clock.js';

const start = Date.UTC( 2026, 9, 21, 15, 0, 0 );

function startSession( scenario )
{
    const clock = createClock( start );
    const events = [];
    const record = ( message ) => events.push( message );
    const options = { scenario, emit: record, schedule: clock.schedule, cancel: clock.cancel, now: clock.now };
    const session = createMockSession( options );

    return { clock, events, session };
}

function ofType( events, type )
{
    return events.filter( ( event ) => event.type === type );
}

function sendTurn( session, text )
{
    const request = { v: 1, type: 'turn.send', stream_id: 'stream-1', text, attachment_ids: [] };

    session.handle( request );
}

function readyIsAnsweredWithTheSessionStateTheConversationsAndTheMode()
{
    const { events, session } = startSession( 1 );
    const ready = { v: 1, type: 'ready' };

    session.handle( ready );

    const types = events.map( ( event ) => event.type );
    assert.deepEqual( types, [ 'session.state', 'conversations.list', 'models.list' ] );
    assert.equal( events[ 0 ].model, 'gemma-4-12b-it-fp4' );
    assert.equal( events[ 0 ].busy, false );
    assert.equal( events[ 1 ].items.length, 5 );
}

test( 'ready is answered with the session state, the conversations and the models', readyIsAnsweredWithTheSessionStateTheConversationsAndTheMode );

function everyEventCarriesTheVersionAndSurvivesJson()
{
    const { clock, events, session } = startSession( 1 );

    sendTurn( session, 'Hello' );
    clock.run();

    for ( const event of events )
    {
        assert.equal( event.v, 1 );

        const text = JSON.stringify( event );
        assert.deepEqual( JSON.parse( text ), event );
    }
}

test( 'every event carries the version and survives JSON', everyEventCarriesTheVersionAndSurvivesJson );

function scenario1StreamsThinkingAToolCallMoreThinkingThenTheAnswer()
{
    const { clock, events, session } = startSession( 1 );

    sendTurn( session, 'What is the weather in Toronto?' );
    clock.run();

    const turnEvents = events.filter( ( event ) => event.type.startsWith( 'turn.' ) );
    const begin = turnEvents[ 0 ];
    const end = turnEvents[ turnEvents.length - 1 ];

    assert.equal( begin.type, 'turn.begin' );
    assert.equal( end.type, 'turn.end' );
    assert.equal( end.finish, 'stop' );

    for ( const event of turnEvents )
    {
        assert.equal( event.turn_id, begin.turn_id );
        assert.equal( event.stream_id, 'stream-1' );
    }

    const texts = ofType( events, 'turn.text' );
    const answer = texts.filter( ( event ) => event.channel === 'answer' ).map( ( event ) => event.text ).join( '' );
    assert.equal( answer, end.answer );
    assert.ok( texts.length > 100, 'streamed token by token' );

    const call = ofType( events, 'turn.tool_call' )[ 0 ];
    assert.equal( call.name, 'get_weather' );
    assert.equal( call.summary, 'Getting weather for Toronto, Canada' );
    assert.deepEqual( JSON.parse( call.arguments ), { location: 'Toronto, Canada' } );

    const order = turnEvents.map( ( event ) => event.type === 'turn.text' ? event.channel : event.type );
    const collapsed = order.filter( ( kind, index ) => index === 0 || kind !== order[ index - 1 ] );
    assert.deepEqual( collapsed, [
        'turn.begin', 'thought', 'turn.tool_call', 'turn.round_end', 'turn.tool_result',
        'turn.thought_break', 'thought', 'answer', 'turn.round_end', 'turn.end'
    ] );

    const rounds = ofType( events, 'turn.round_end' );

    for ( const round of rounds )
    {
        const thought = texts.filter( ( event ) => event.round === round.round && event.channel === 'thought' );
        assert.equal( round.thought_tokens, thought.length );
        assert.ok( round.tokens >= round.thought_tokens );
    }

    assert.ok( answer.includes( '| Temperature |' ), 'the answer has a table' );
    assert.ok( answer.includes( '```python' ), 'the answer has a code block' );
}

test( 'scenario 1 streams thinking, a tool call, more thinking, then the answer', scenario1StreamsThinkingAToolCallMoreThinkingThenTheAnswer );

function aReplyArrivesWithRealisticTiming()
{
    const { clock, session } = startSession( 1 );

    sendTurn( session, 'Hello' );
    clock.run();

    const seconds = ( clock.now() - start ) / 1000;
    assert.ok( seconds > 5 && seconds < 30, 'took ' + seconds + ' s' );
}

test( 'a reply arrives with realistic timing', aReplyArrivesWithRealisticTiming );

function aNewConversationIsStoredUnderItsFirstMessageOnceTheReplyEnds()
{
    const { clock, events, session } = startSession( 1 );
    const text = 'What is the weather in Toronto right now, and should I cycle to work today or take the train?';

    sendTurn( session, text );
    clock.run();

    const lists = ofType( events, 'conversations.list' );
    const latest = lists[ lists.length - 1 ];
    const begin = ofType( events, 'turn.begin' )[ 0 ];
    const stored = latest.items.find( ( item ) => item.id === begin.conversation_id );

    assert.equal( latest.items.length, 6 );
    assert.equal( stored.title, titleFor( text ) );
    assert.ok( stored.title.length < 60 );

    const states = ofType( events, 'session.state' );
    const last = states[ states.length - 1 ];
    assert.equal( last.busy, false );
    assert.equal( last.streams[ 0 ].conversation_id, begin.conversation_id );
}

test( 'a new conversation is stored under its first message once the reply ends', aNewConversationIsStoredUnderItsFirstMessageOnceTheReplyEnds );

function aSendWhileBusyIsRefusedWithAnError()
{
    const { clock, events, session } = startSession( 1 );
    const isBegin = () => events.some( ( event ) => event.type === 'turn.begin' );

    sendTurn( session, 'First' );
    clock.run( isBegin );
    sendTurn( session, 'Second' );

    const errors = ofType( events, 'error' );
    assert.equal( errors.length, 1 );
    assert.equal( errors[ 0 ].during, 'turn' );
}

test( 'a send while busy is refused with an error', aSendWhileBusyIsRefusedWithAnError );

function stopEndsTheReplyAsCancelledWithTheAnswerSoFar()
{
    const { clock, events, session } = startSession( 1 );
    const answering = () => events.some( ( event ) => event.type === 'turn.text' && event.channel === 'answer' );
    const stop = { v: 1, type: 'turn.stop', stream_id: 'stream-1' };

    sendTurn( session, 'Hello' );
    clock.run( answering );
    session.handle( stop );
    clock.run();

    const ends = ofType( events, 'turn.end' );
    assert.equal( ends.length, 1 );
    assert.equal( ends[ 0 ].finish, 'cancelled' );

    const answer = ofType( events, 'turn.text' ).filter( ( event ) => event.channel === 'answer' ).map( ( event ) => event.text ).join( '' );
    assert.equal( ends[ 0 ].answer, answer );
}

test( 'stop ends the reply as cancelled with the answer so far', stopEndsTheReplyAsCancelledWithTheAnswerSoFar );

function thePageReducesScenario1ToOneUserTurnAndOneAssistantTurnInOrd()
{
    const { clock, events, session } = startSession( 1 );
    const request = { v: 1, type: 'turn.send', stream_id: 'stream-1', text: 'Weather?', attachment_ids: [] };
    let state = createInitialState();

    state = reduce( state, request );
    session.handle( request );
    clock.run();

    for ( const event of events )
        state = reduce( state, event );

    assert.equal( state.transcript.length, 2 );

    const turn = state.transcript[ 1 ];
    const kinds = turn.blocks.map( ( block ) => block.kind );
    assert.deepEqual( kinds, [ 'thought', 'tool', 'thought', 'answer' ] );
    assert.equal( turn.streaming, false );
    assert.equal( turn.blocks[ 1 ].result, '{"temperature_c": 18, "condition": "cloudy"}' );
    assert.equal( turn.blocks[ 0 ].tokens, ofType( events, 'turn.round_end' )[ 0 ].thought_tokens );
    assert.equal( turn.blocks[ 3 ].text, turn.answer );
    assert.equal( state.conversations.length, 6 );
}

test( 'the page reduces scenario 1 to one user turn and one assistant turn in order', thePageReducesScenario1ToOneUserTurnAndOneAssistantTurnInOrd );

function aStoredConversationReopensWithWhatItHeld()
{
    const { events, session } = startSession( 1 );
    const open = { v: 1, type: 'conversation.open', id: 'conversation-3' };

    session.handle( open );

    const loaded = ofType( events, 'conversation.loaded' )[ 0 ];
    assert.equal( loaded.title, 'Plan a three-day trip to Kyoto' );
    assert.equal( loaded.messages.length, 2 );
    assert.equal( loaded.messages[ 1 ].role, 'assistant' );
}

test( 'a stored conversation reopens with what it held', aStoredConversationReopensWithWhatItHeld );

function settingsChangeTheStateAndAnUnknownOneIsRefused()
{
    const { events, session } = startSession( 1 );
    const context = { v: 1, type: 'settings.set', context: 32768 };
    const effort = { v: 1, type: 'settings.set', effort: 4 };

    session.handle( context );
    session.handle( effort );

    const states = ofType( events, 'session.state' );
    assert.equal( states[ 0 ].context_length, 32768 );
    assert.equal( states[ 0 ].context_automatic, false );

    const errors = ofType( events, 'error' );
    assert.equal( errors.length, 1, 'Gemma has no effort levels' );
    assert.equal( errors[ 0 ].during, 'settings' );
}

test( 'settings change the state, and an unknown one is refused', settingsChangeTheStateAndAnUnknownOneIsRefused );

function titlesAreCutAtAWordUnder60Characters()
{
    assert.equal( titleFor( 'Short one' ), 'Short one' );

    const long = 'one two three four five six seven eight nine ten eleven twelve thirteen';
    const title = titleFor( long );
    assert.ok( title.length < 60 );
    assert.ok( long.startsWith( title ) );
    assert.ok( long[ title.length ] === ' ' );
}

test( 'titles are cut at a word under 60 characters', titlesAreCutAtAWordUnder60Characters );

function tokenPiecesJoinBackToTheText()
{
    const text = 'It\'s **18 °C**, cloudy.\n\n```python\nprint(1)\n```';
    const pieces = splitIntoTokens( text );

    assert.equal( pieces.join( '' ), text );
}

test( 'token pieces join back to the text', tokenPiecesJoinBackToTheText );

function attachSample( session, identifier, kind )
{
    const request = { v: 1, type: 'attachment.add', attachment_id: identifier, kind, name: 'harbour-at-dusk.png', path: '/pictures/harbour-at-dusk.png' };

    session.handle( request );
}

function scenarioSixAcceptsAnImageAndDescribesIt()
{
    const { clock, events, session } = startSession( 6 );
    const request = { v: 1, type: 'turn.send', stream_id: 'stream-1', text: 'What is in this picture?', attachment_ids: [ 'a1' ] };

    attachSample( session, 'a1', 'image' );
    clock.run();

    const accepted = ofType( events, 'attachment.accepted' );
    assert.equal( accepted.length, 1 );
    assert.equal( accepted[ 0 ].kind, 'image' );
    assert.ok( accepted[ 0 ].size_bytes > 0 );

    session.handle( request );
    clock.run();

    const end = ofType( events, 'turn.end' )[ 0 ];
    assert.match( end.answer, /lighthouse/ );

    const open = { v: 1, type: 'conversation.open', id: ofType( events, 'turn.begin' )[ 0 ].conversation_id };
    session.handle( open );

    const loaded = ofType( events, 'conversation.loaded' )[ 0 ];
    assert.deepEqual( loaded.messages[ 0 ].attachments, [ { kind: 'image', name: 'harbour-at-dusk.png' } ] );
}

test( 'scenario 6 accepts an image on Gemma and describes it', scenarioSixAcceptsAnImageAndDescribesIt );

function anAttachmentIsRefusedOnAModelWithoutItsKind()
{
    const { clock, events, session } = startSession( 6 );
    const load = { v: 1, type: 'model.load', name: 'Llama-3.2-3B-Instruct-fp4', quantization: null };

    session.handle( load );
    clock.run();
    attachSample( session, 'a2', 'image' );
    clock.run();

    const refused = ofType( events, 'attachment.refused' );
    assert.equal( refused.length, 1 );
    assert.equal( refused[ 0 ].reason, 'Llama-3.2-3B-Instruct-fp4 cannot take images.' );

    const states = ofType( events, 'session.state' );
    assert.deepEqual( states[ states.length - 1 ].modalities, [] );
}

test( 'an attachment is refused on a model without its kind', anAttachmentIsRefusedOnAModelWithoutItsKind );

function anAttachmentWithNoPathIsRefused()
{
    const { clock, events, session } = startSession( 6 );
    const request = { v: 1, type: 'attachment.add', attachment_id: 'a3', kind: 'image', name: 'x.png', path: null };

    session.handle( request );
    clock.run();

    assert.equal( ofType( events, 'attachment.refused' ).length, 1 );
}

test( 'an attachment with no path is refused', anAttachmentWithNoPathIsRefused );

function loadingAModelReportsProgressThenTheNewState()
{
    const { clock, events, session } = startSession( 1 );
    const load = { v: 1, type: 'model.load', name: 'qwen3.8-27b-fp4', quantization: null };

    session.handle( load );
    clock.run();

    const progress = ofType( events, 'progress' );
    const last = progress[ progress.length - 1 ];
    assert.equal( last.task, 'load' );
    assert.equal( last.done, last.total );

    const states = ofType( events, 'session.state' );
    const final = states[ states.length - 1 ];
    assert.equal( final.model, 'qwen3.8-27b-fp4' );
    assert.equal( final.effort_levels, true );
    assert.equal( final.thinking, 'balanced' );
}

test( 'loading a model reports progress, then the new state', loadingAModelReportsProgressThenTheNewState );
