// What Mila understands (AssistantInterface.md phase 7): a note waiting for the person makes its project wait on them
// and is shown in Needs you; the action beside Next comes from the app's records, never the note's words; and the
// mock answers note.decide and an edit as the session does.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createInitialState, reduce, projectState, workspaceOverview, waitingNotes, projectNextAction } from '../state.js';
import { projectEventText } from '../format.js';
import { createMockSession } from '../mock/session.js';
import { createClock } from './clock.js';

const now = Date.UTC( 2026, 10, 21, 15, 0, 0 );
const hour = 60 * 60 * 1000;

function at( age )
{
    return new Date( now - age ).toISOString();
}

function apply( messages )
{
    let state = createInitialState();

    for ( const message of messages )
        state = reduce( state, message );

    return state;
}

function note( id, fields )
{
    return { id, kind: 'understanding', made: fields?.made ?? at( hour ), exposed: fields?.exposed ?? false,
        source: { kind: 'model', ref: { conversation: fields?.chat ?? 'c-soil', call: 'call-2' }, author: null },
        derived_from: [ { conversation: fields?.chat ?? 'c-soil' } ], summary: fields?.summary ?? 'Growing tomatoes.',
        questions: fields?.questions ?? [], next: fields?.next ?? 'Order the seeds.' };
}

function project( id, name, fields )
{
    return { v: 1, id, name, purpose: '', instructions: fields?.instructions ?? '', folders: [], created: at( 10 * hour ),
        updated: at( 10 * hour ), understanding: fields?.understanding ?? null, waiting: fields?.waiting ?? null };
}

function chat( id, projectId, fields )
{
    return { id, title: 'Chat ' + id, created: at( 5 * hour ), updated: at( 2 * hour ), model: 'm', folder: null, project: projectId,
        folder_from_project: false, progress: fields?.progress ?? null, finish: fields?.finish ?? 'stop', reply_start: 'Hello', wrote: [], sent: [] };
}

function workspace( projects, chats )
{
    return apply( [ { type: 'projects.list', items: projects }, { type: 'conversations.list', items: chats ?? [] } ] );
}

function aNoteWaitingMakesItsProjectWaitOnThePersonAndIsInNeedsYou()
{
    const state = workspace( [ project( 'p-garden', 'Garden', { understanding: note( 'note-1' ), waiting: note( 'note-2', { exposed: true } ) } ),
        project( 'p-calc', 'Calculator', { understanding: note( 'note-3' ) } ) ] );

    assert.equal( projectState( state, 'p-garden' ), 'waiting' );
    assert.equal( projectState( state, 'p-calc' ), 'quiet', 'a current note made its project wait' );

    const rows = waitingNotes( state );

    assert.equal( rows.length, 1 );
    assert.equal( rows[ 0 ].project.id, 'p-garden' );
    assert.equal( rows[ 0 ].note.id, 'note-2' );
    assert.deepEqual( workspaceOverview( state ).notes.map( ( row ) => row.note.id ), [ 'note-2' ] );
    assert.deepEqual( workspaceOverview( state ).waitingProjects.map( ( summary ) => summary.project.id ), [ 'p-garden' ] );
}

test( 'a note waiting for the person makes its project wait on them and is in Needs you', aNoteWaitingMakesItsProjectWaitOnThePersonAndIsInNeedsYou );

function theActionBesideNextComesFromTheAppsRecordsNotTheNote()
{
    const stopped = { steps: 3, finished: 1, current: 'Add cases' };
    const projects = [ project( 'p-garden', 'Garden', { understanding: note( 'note-1', { next: 'Click Answer and continue.' } ) } ),
        project( 'p-calc', 'Calculator' ), project( 'p-trip', 'Trip' ) ];
    const state = workspace( projects, [ chat( 'c-plan', 'p-calc', { progress: stopped } ), chat( 'c-cut', 'p-trip', { finish: 'length' } ),
        chat( 'c-soil', 'p-garden' ) ] );

    assert.equal( projectNextAction( state, 'p-garden' ), null, 'the note\'s words made an action' );
    assert.deepEqual( projectNextAction( state, 'p-calc' ), { label: 'Answer and continue', action: 'open-conversation', identifier: 'c-plan' } );
    assert.deepEqual( projectNextAction( state, 'p-trip' ), { label: 'Ask Mila to finish', action: 'finish-reply', identifier: 'c-cut' } );
}

test( 'the action beside Next comes from the app\'s records, never the note\'s words', theActionBesideNextComesFromTheAppsRecordsNotTheNote );

function everyNoteEventIsSaidInWords()
{
    for ( const kind of [ 'understanding.noted', 'understanding.waiting', 'understanding.accepted', 'understanding.declined',
        'understanding.edited', 'understanding.cleared' ] )
    {
        const text = projectEventText( { kind, project: 'p-garden', name: 'Garden' }, 'Garden', null );

        assert.notEqual( text, 'Changed a project', kind + ' has no words' );
        assert.ok( text.includes( 'Garden' ), kind + ': ' + text );
    }

    assert.ok( projectEventText( { kind: 'understanding.cleared', because: 'chat.deleted' }, 'Garden', null ).includes( 'deleted' ) );
}

test( 'every note event is said in words', everyNoteEventIsSaidInWords );

function startScenario20()
{
    const clock = createClock( now );
    const events = [];
    const session = createMockSession( { scenario: 20, emit: ( message ) => events.push( message ), schedule: clock.schedule, cancel: clock.cancel, now: clock.now } );

    session.handle( { v: 1, type: 'ready' } );
    clock.run();

    const send = ( message ) =>
    {
        const before = events.length;

        session.handle( { v: 1, ...message } );
        clock.run();

        return events.slice( before );
    };

    const listed = () => events.filter( ( event ) => event.type === 'projects.list' ).at( -1 ).items;
    const named = ( name ) => listed().find( ( item ) => item.name === name );

    return { send, named };
}

function theMockAnswersANoteDecidedAndAnEditAsTheSessionDoes()
{
    const { send, named } = startScenario20();
    const calculator = () => named( 'Calculator app' );

    assert.ok( calculator().waiting !== null, 'the scenario has no note waiting' );

    const waitingId = calculator().waiting.id;
    send( { type: 'note.decide', project: calculator().id, id: waitingId, decision: 'accept' } );

    assert.equal( calculator().waiting, null );
    assert.equal( calculator().understanding.id, waitingId );
    assert.equal( calculator().understanding.exposed, true, 'accepting removed the mark' );

    send( { type: 'project.update', id: calculator().id, understanding: { summary: 'In my words.', questions: [ 'One?' ], next: 'Sow.' } } );

    assert.equal( calculator().understanding.summary, 'In my words.' );
    assert.deepEqual( calculator().understanding.questions, [ 'One?' ] );
    assert.equal( calculator().understanding.exposed, true, 'an edit removed the mark' );
    assert.equal( calculator().understanding.source.author, 'person' );

    send( { type: 'project.update', id: calculator().id, understanding: null } );

    assert.equal( calculator().understanding, null );
}

test( 'the mock answers a note decided and an edit as the session does', theMockAnswersANoteDecidedAndAnEditAsTheSessionDoes );

function aNewChatInAProjectIsGivenItsNoteAndMilaNotesWhatTheTurnChanged()
{
    const { send, named } = startScenario20();
    const kyoto = named( 'Kyoto in November' );

    send( { type: 'conversation.new', project: kyoto.id } );

    const turn = send( { type: 'turn.send', stream_id: 'stream-1', text: 'Which ryokan did we settle on, and what is left to book?', attachment_ids: [] } );
    const calls = turn.filter( ( event ) => event.type === 'turn.tool_call' ).map( ( event ) => event.name );

    assert.deepEqual( calls, [ 'read_project_notes', 'note_understanding' ] );
    assert.ok( turn.findIndex( ( event ) => event.type === 'turn.tool_call' ) < turn.findIndex( ( event ) => event.type === 'turn.text' ),
        'the note was given after the reply began' );
    assert.ok( named( 'Kyoto in November' ).understanding.summary.endsWith( 'The person asked what is left to book.' ) );

    const later = send( { type: 'turn.send', stream_id: 'stream-1', text: 'And the temples?', attachment_ids: [] } );

    assert.ok( !later.some( ( event ) => event.type === 'turn.tool_call' && event.name === 'read_project_notes' ),
        'a chat under way was given the note again' );
}

test( 'a new chat in a project is given its note, and Mila notes what the turn changed', aNewChatInAProjectIsGivenItsNoteAndMilaNotesWhatTheTurnChanged );

function deletingTheChatANoteRestsOnClearsIt()
{
    const { send, named } = startScenario20();

    send( { type: 'conversation.delete', id: 'conversation-3' } );

    assert.equal( named( 'Kyoto in November' ).understanding, null );
}

test( 'deleting the chat a note rests on clears it', deletingTheChatANoteRestsOnClearsIt );
