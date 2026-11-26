// The onboarding's panes are gone through once (AssistantOnboarding.md 4.4): a first run shows them from the welcome,
// and a launch part way through the interview shows Almost Ready while Mila gets ready, then the interview's chat.
// Each plays the mock's scenario through the page's own reducer.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createMockSession } from '../mock/session.js';
import { createInitialState, reduce, setupPane, setupChatClosed, needsYou } from '../state.js';
import { attentionLine } from '../format.js';
import { createClock } from './clock.js';

const start = Date.UTC( 2026, 10, 25, 15, 0, 0 );

/** The scenario through the page: every event reduced as the window's page reduces it, and each pane it showed. */
function play( scenario )
{
    const clock = createClock( start );
    const played = { state: createInitialState(), panes: [], events: [] };
    const emit = ( message ) =>
    {
        played.events.push( message );
        played.state = reduce( played.state, message );
        played.panes.push( setupPane( played.state ) );
    };
    const session = createMockSession( { scenario, emit, schedule: clock.schedule, cancel: clock.cancel, now: clock.now } );

    session.handle( { v: 1, type: 'ready' } );

    return { clock, session, played };
}

function aFirstRunBeginsAtTheWelcome()
{
    const { played } = play( 21 );

    assert.equal( setupPane( played.state ), 'welcome' );
}

test( 'a first run begins at the welcome', aFirstRunBeginsAtTheWelcome );

function aLaunchPartWayThroughTheInterviewReturnsToItsChatWithoutThePanes()
{
    const { clock, played } = play( 22 );

    clock.run( () => played.events.some( ( event ) => event.type === 'interview' ) );

    const shown = new Set( played.panes.filter( ( pane ) => pane !== null ) );

    for ( const pane of [ 'welcome', 'license', 'ready', 'capabilities' ] )
        assert.equal( shown.has( pane ), false, pane + ' is shown again' );

    assert.equal( shown.has( 'getting-ready' ), true, 'Almost Ready while Mila gets ready' );
    assert.equal( setupPane( played.state ), null, 'the setup has ended' );
    assert.equal( played.state.place, 'chat', 'on the interview\'s chat' );
}

test( 'a launch part way through the interview returns to its chat without the panes', aLaunchPartWayThroughTheInterviewReturnsToItsChatWithoutThePanes );

/** Scenario 21 through its panes to the interview's first question: the terms, the download, the panes' end. */
function toTheInterview()
{
    const played = play( 21 );
    const { clock, session } = played;
    const license = played.played.events.filter( ( event ) => event.type === 'license' ).pop();

    session.handle( { v: 1, type: 'license.accept', sha256: license.sha256 } );
    session.handle( { v: 1, type: 'onboarding.download' } );
    session.handle( { v: 1, type: 'onboarding.panes_done' } );
    clock.run( () => played.played.events.some( ( event ) => event.type === 'interview' ) );
    clock.run( () => played.played.state.session.busy === false );

    return played;
}

/** The person's message, and the reply played to its end. */
function say( played, text )
{
    const ended = played.played.events.filter( ( event ) => event.type === 'turn.end' ).length;

    played.session.handle( { v: 1, type: 'turn.send', stream_id: 'stream-1', text, attachment_ids: [] } );
    played.clock.run( () => played.played.events.filter( ( event ) => event.type === 'turn.end' ).length > ended );
}

/** What the setup has still to accomplish, as the workspace reads its chat; null for a chat that is not the setup's. */
function stillToDo( state )
{
    const item = state.conversations.find( ( conversation ) => conversation.id === state.session.setup_conversation );

    return item === undefined ? null : item.setup;
}

function anUnfinishedSetupNeedsThePersonWithWhatIsStillToDoAndShowsNoPlan()
{
    const played = toTheInterview();

    assert.deepEqual( stillToDo( played.played.state ), [ 'What to call you', 'What you will use Mila for', 'What Mila needs for your work' ] );

    say( played, 'Alex' );

    const named = played.played.state;
    assert.deepEqual( stillToDo( named ), [ 'What you will use Mila for', 'What Mila needs for your work' ] );
    assert.equal( named.session.plan, null, 'the setup is not a plan, and none is drawn' );

    const waiting = needsYou( named ).find( ( row ) => row.item.id === named.session.setup_conversation );
    assert.ok( waiting !== undefined, 'the unfinished setup is under Needs you' );
    assert.equal( waiting.kind, 'setup' );
    assert.equal( attentionLine( waiting ), 'Still to do: what you will use Mila for and what Mila needs for your work' );
}

test( 'an unfinished setup needs the person, with what is still to do, and shows no plan', anUnfinishedSetupNeedsThePersonWithWhatIsStillToDoAndShowsNoPlan );

function whenMilaNeedsNothingTheSetupEndsWithNoQuestionAndItsChatCloses()
{
    const played = toTheInterview();

    say( played, 'Alex' );
    say( played, 'writing and planning trips' );

    const { state } = played.played;

    assert.equal( played.played.events.some( ( event ) => event.type === 'turn.offer' ), false, 'nothing is asked' );
    assert.equal( state.session.onboarding, 'done' );
    assert.deepEqual( stillToDo( state ), [] );
    assert.equal( setupChatClosed( state ), true, 'the setup\'s chat closes' );
    assert.equal( needsYou( state ).some( ( row ) => row.item.id === state.session.setup_conversation ), false, 'and needs nothing more' );

    // It takes no more messages: what comes next is a new chat.
    const errors = played.played.events.filter( ( event ) => event.type === 'error' ).length;
    say( played, 'And now?' );
    assert.equal( played.played.events.filter( ( event ) => event.type === 'error' ).length, errors + 1 );
}

test( 'when Mila needs nothing, the setup ends with no question and its chat closes', whenMilaNeedsNothingTheSetupEndsWithNoQuestionAndItsChatCloses );

/** Scenario 23, the panes gone through, to the interview's first question; then the name and coding work. */
function toTheDownload()
{
    const played = play( 23 );

    played.clock.run( () => played.played.events.some( ( event ) => event.type === 'interview' ) );
    played.clock.run( () => played.played.state.session.busy === false );
    say( played, 'Alex' );
    say( played, 'coding and building programs' );

    return played;
}

function forCodingWorkMilaAsksToDownloadNamingNoModel()
{
    const played = toTheDownload();
    const { state } = played.played;
    const offered = played.played.events.filter( ( event ) => event.type === 'turn.offer' ).pop();

    assert.equal( offered.offer.kind, 'download' );
    assert.deepEqual( offered.offer.download.gains, [ 'work of several steps with your files and tools' ] );
    assert.deepEqual( stillToDo( state ), [ 'What Mila needs for your work' ] );
    assert.equal( setupChatClosed( state ), false, 'still open while the download waits' );

    const reply = state.transcript.filter( ( entry ) => entry.kind === 'assistant' ).pop();
    assert.match( reply.answer, /download about 14\.4 GB/ );
    assert.equal( /gemma|qwen|model/i.test( reply.answer ), false, 'Mila names no model' );
}

test( 'for coding work, Mila asks to download what it needs, naming no model', forCodingWorkMilaAsksToDownloadNamingNoModel );

function theDownloadAcceptedWithItsButtonEndsTheSetup()
{
    const played = toTheDownload();
    const offered = played.played.events.filter( ( event ) => event.type === 'turn.offer' ).pop();
    const conversation = played.played.state.session.setup_conversation;

    played.session.handle( { v: 1, type: 'suggestion.decide', conversation_id: conversation, call_id: offered.call_id, decision: 'accept' } );
    played.clock.run();

    const { state } = played.played;
    const answered = played.played.events.filter( ( event ) => event.type === 'turn.answered' ).pop();

    assert.equal( answered.offer.state, 'accepted' );
    assert.equal( answered.text, 'I\'m set up: I\'ll call you Alex, and I\'m downloading what I need for your work, about 14.4 GB; your next new chat uses it. Start a new chat whenever you\'re ready.' );
    assert.equal( state.session.onboarding, 'done' );
    assert.equal( setupChatClosed( state ), true );
}

test( 'the download accepted with its button ends the setup', theDownloadAcceptedWithItsButtonEndsTheSetup );

function theDownloadDeclinedInWordsEndsTheSetupAndMilaCarriesOn()
{
    const played = toTheDownload();

    say( played, 'No thanks, not now.' );

    const { state } = played.played;
    const answered = played.played.events.filter( ( event ) => event.type === 'turn.answered' ).pop();

    assert.equal( answered.offer.state, 'declined' );
    assert.equal( answered.text, 'I\'m set up: I\'ll call you Alex, and I\'ll carry on as I am. Start a new chat whenever you\'re ready.' );
    assert.equal( state.session.onboarding, 'done' );
    assert.deepEqual( stillToDo( state ), [] );
    assert.equal( setupChatClosed( state ), true );
}

test( 'the download declined in words ends the setup, and Mila carries on as it is', theDownloadDeclinedInWordsEndsTheSetupAndMilaCarriesOn );