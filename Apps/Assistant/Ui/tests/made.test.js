// What Mila made (AssistantInterface.md phase 8): the page's selectors over each chat's files and kept replies, the
// drawer's request and answer, Keep under a reply, and the mock's answers, which stand in for the session's.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createInitialState, reduce, madeIn, madeInProject, changedOnDisk, needsYou, projectState, projectSummaries,
    activityDays, isKept, searchResults } from '../state.js';
import { activityText, attentionLine } from '../format.js';
import { createMockSession } from '../mock/session.js';
import { createClock } from './clock.js';

const now = Date.UTC( 2026, 10, 19, 15, 0, 0 );
const hour = 60 * 60 * 1000;

function at( age )
{
    return new Date( now - age ).toISOString();
}

function apply( messages, state = createInitialState() )
{
    for ( const message of messages )
        state = reduce( state, message );

    return state;
}

function chat( id, project, fields )
{
    return { id, title: 'Chat ' + id, created: at( 5 * hour ), updated: fields?.updated ?? at( 2 * hour ), model: 'm', folder: 'C:\\work',
        project, folder_from_project: false, progress: null, finish: 'stop', reply_start: 'Hello', wrote: fields?.wrote ?? [],
        sent: [], kept: fields?.kept ?? [] };
}

const workProject = { v: 1, id: 'p-work', name: 'Work', purpose: '', instructions: '', folders: [], created: at( 9 * hour ), updated: at( 9 * hour ) };

function workspace()
{
    return apply( [
        { type: 'projects.list', items: [ workProject ] },
        { type: 'conversations.list', items: [
            chat( 'c-notes', 'p-work', { wrote: [ { path: 'notes.md', tool: 'write_file', time: at( 3 * hour ), disk: 'changed' },
                { path: 'plan.txt', tool: 'edit_file', time: at( 1 * hour ), disk: 'same' } ],
                kept: [ { message: 1, time: at( 2 * hour ), start: 'The **answer** is here.' } ] } ),
            chat( 'c-old', null, { wrote: [ { path: 'old.py', tool: 'write_file', time: null, disk: null } ] } ) ] } ] );
}

function whatAChatMadeIsItsFilesAndKeptRepliesNewestFirst()
{
    const state = workspace();
    const made = madeIn( state.conversations[ 0 ] );

    assert.deepEqual( made.map( ( entry ) => entry.kind === 'file' ? entry.path : 'reply ' + entry.message ), [ 'plan.txt', 'reply 1', 'notes.md' ] );
    assert.equal( madeInProject( state, 'p-work' ).length, 3 );
    assert.equal( projectSummaries( state ).find( ( summary ) => summary.project.id === 'p-work' ).made, 3 );
}

test( 'what a chat made is its files and kept replies, the newest first', whatAChatMadeIsItsFilesAndKeptRepliesNewestFirst );

function aFileChangedOnDiskIsNoticedAndWaitsOnNobody()
{
    const state = workspace();
    const noticed = changedOnDisk( state );

    assert.equal( noticed.length, 1 );
    assert.equal( noticed[ 0 ].file.path, 'notes.md' );
    assert.equal( attentionLine( noticed[ 0 ] ), 'Changed on disk after Mila last wrote it, in Chat c-notes' );
    assert.equal( needsYou( state ).length, 0 );
    assert.equal( projectState( state, 'p-work' ), 'quiet' );
}

test( 'a file changed on disk is noticed, and waits on nobody', aFileChangedOnDiskIsNoticedAndWaitsOnNobody );

function activityShowsEachFileAtItsWriteAndEachKeptReply()
{
    const events = activityDays( workspace(), 'file' ).flatMap( ( day ) => day.events );
    const texts = events.map( activityText );

    assert.deepEqual( texts, [ 'Changed plan.txt', 'Kept a reply: The answer is here.', 'Wrote old.py', 'Wrote notes.md' ] );
    assert.equal( events[ 1 ].who, 'you' );

    // A file written before calls had a time is shown at its chat's last change, as before.
    assert.equal( events[ 2 ].time, at( 2 * hour ) );
    assert.equal( events[ 2 ].file, undefined );
}

test( 'Activity shows each file at its write, and each kept reply', activityShowsEachFileAtItsWriteAndEachKeptReply );

function theDrawerHoldsItsRequestThenTheAnswerOrWhyThereIsNone()
{
    const asked = apply( [ { type: 'creation.open', conversation_id: 'c-notes', path: 'notes.md' } ], workspace() );

    assert.equal( asked.creation.loaded, null );

    const loaded = apply( [ { type: 'creation.loaded', creation: { kind: 'file' }, content: 'x', content_kind: 'text', disk: 'same' } ], asked );
    assert.equal( loaded.creation.loaded.content, 'x' );

    const failed = apply( [ { type: 'error', text: 'That is no longer among what Mila made.', during: null } ], asked );
    assert.equal( failed.creation.problem, 'That is no longer among what Mila made.' );
    assert.equal( failed.transcript.length, 0 );

    assert.equal( apply( [ { type: 'creation.close' } ], loaded ).creation, null );

    // An answer that comes after the drawer closed is not shown.
    assert.equal( apply( [ { type: 'creation.close' }, { type: 'creation.loaded', creation: {} } ], loaded ).creation, null );
}

test( 'the drawer holds its request, then the answer or why there is none', theDrawerHoldsItsRequestThenTheAnswerOrWhyThereIsNone );

function aReplyKnowsItsPlaceAmongTheMessagesAndWhetherItIsKept()
{
    const opened = apply( [ { type: 'conversation.loaded', id: 'c-notes', title: 'Chat c-notes', model: 'm', folder: null, project: 'p-work', messages: [
        { role: 'user', content: 'Q', attachments: null },
        { role: 'assistant', content: 'A', thought: null, tool_calls: null, stats: null } ] } ], workspace() );

    const reply = opened.transcript.find( ( entry ) => entry.kind === 'assistant' );
    assert.equal( reply.message, 1 );
    assert.equal( isKept( opened, 1 ), true );
    assert.equal( isKept( opened, 3 ), false );

    const live = apply( [ { type: 'turn.begin', turn_id: 't1', conversation_id: 'c-notes' },
        { type: 'turn.end', turn_id: 't1', finish: 'stop', answer: 'B', message: 3 } ], opened );

    assert.equal( live.transcript[ live.transcript.length - 1 ].message, 3 );
}

test( 'a reply knows its place among the messages, and whether it is kept', aReplyKnowsItsPlaceAmongTheMessagesAndWhetherItIsKept );

function aFileFoundBySearchCarriesItsChat()
{
    const results = searchResults( workspace(), 'notes' ).filter( ( result ) => result.kind === 'file' );

    assert.equal( results.length, 1 );
    assert.equal( results[ 0 ].item.id, 'c-notes' );
    assert.equal( results[ 0 ].disk, 'changed' );
}

test( 'a file found by Search carries the chat that wrote it', aFileFoundBySearchCarriesItsChat );

// ---- The mock, as the session ------------------------------------------------------------------

function startMock()
{
    const clock = createClock( now );
    const events = [];
    const session = createMockSession( { scenario: 17, emit: ( message ) => events.push( message ), schedule: clock.schedule, cancel: clock.cancel, now: clock.now } );

    session.handle( { v: 1, type: 'ready' } );

    return { clock, events, session };
}

function latest( events, type )
{
    return events.filter( ( event ) => event.type === type ).at( -1 );
}

function scenarioSeventeenDescribesAFileAndWhereItCameFrom()
{
    const { events, session } = startMock();
    const calculator = latest( events, 'conversations.list' ).items.find( ( item ) => item.id === 'conversation-6' );

    assert.deepEqual( calculator.wrote.map( ( file ) => [ file.path, file.disk ] ), [ [ 'tests/test_parser.py', 'same' ], [ 'TESTING.md', 'changed' ] ] );

    session.handle( { v: 1, type: 'creation.open', conversation_id: 'conversation-6', path: 'TESTING.md' } );

    const loaded = latest( events, 'creation.loaded' );
    assert.equal( loaded.disk, 'changed' );
    assert.equal( loaded.content_kind, 'markdown' );
    assert.equal( loaded.creation.made.step, 'Move the parser cases into their own file' );
    assert.deepEqual( loaded.creation.read_before.map( ( read ) => read.what ), [ 'Reading tests/test_calc.py' ] );
    assert.equal( loaded.creation.exposed, true );

    session.handle( { v: 1, type: 'creation.open', conversation_id: 'conversation-6', path: 'never.txt' } );
    assert.equal( latest( events, 'error' ).text, 'That is no longer among what Mila made.' );
}

test( 'scenario 17 describes a file and where it came from', scenarioSeventeenDescribesAFileAndWhereItCameFrom );

function aReplyIsKeptAndLetGo()
{
    const { events, session } = startMock();
    const kept = () => latest( events, 'conversations.list' ).items.find( ( item ) => item.id === 'conversation-1' ).kept;

    assert.equal( kept().length, 0 );

    session.handle( { v: 1, type: 'creation.keep', conversation_id: 'conversation-1', message: 1, keep: true } );
    assert.deepEqual( kept().map( ( reply ) => reply.message ), [ 1 ] );

    session.handle( { v: 1, type: 'creation.open', conversation_id: 'conversation-1', message: 1 } );
    assert.equal( latest( events, 'creation.loaded' ).creation.kind, 'reply' );

    session.handle( { v: 1, type: 'creation.keep', conversation_id: 'conversation-1', message: 1, keep: false } );
    assert.equal( kept().length, 0 );

    // A person's message is not a reply, and is never kept.
    session.handle( { v: 1, type: 'creation.keep', conversation_id: 'conversation-1', message: 0, keep: true } );
    assert.equal( kept().length, 0 );
}

test( 'a reply is kept and let go, and a person\'s message is never kept', aReplyIsKeptAndLetGo );
