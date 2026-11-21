// Projects (AssistantInterface.md phase 4): the page's selectors over projects, chats and events, and the mock's
// answers to the project messages, which stand in for the session's.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createInitialState, reduce, projectState, workspaceOverview, barMode, barProject, activityDays, searchResults,
    reachOverview, projectSummaries, matchingProjects, filteredChats } from '../state.js';
import { projectEventText } from '../format.js';
import { createMockSession } from '../mock/session.js';
import { createClock } from './clock.js';

const now = Date.UTC( 2026, 10, 19, 15, 0, 0 );
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

function sessionState( fields )
{
    const stream = { stream_id: 's', parent_stream_id: null, conversation_id: fields.conversation ?? null, busy: fields.busy ?? false };

    return { type: 'session.state', model: 'm', busy: fields.busy ?? false, project: fields.project ?? null, folder: fields.folder ?? null,
        folder_access: fields.folder ? 'read' : null, folder_from_project: fields.fromProject ?? false, streams: [ stream ] };
}

function chat( id, project, fields )
{
    return { id, title: 'Chat ' + id, created: at( 5 * hour ), updated: fields?.updated ?? at( 2 * hour ), model: 'm', folder: fields?.folder ?? null,
        project, folder_from_project: fields?.fromProject ?? false, progress: fields?.progress ?? null, finish: fields?.finish ?? 'stop',
        reply_start: 'Hello', wrote: fields?.wrote ?? [], sent: [] };
}

function project( id, name, fields )
{
    return { v: 1, id, name, purpose: fields?.purpose ?? '', instructions: '', folders: fields?.folders ?? [], created: at( 10 * hour ),
        updated: fields?.updated ?? at( 10 * hour ) };
}

const stopped = { steps: 3, finished: 1, current: 'Add cases' };

function workspace( session )
{
    return apply( [
        sessionState( session ?? {} ),
        { type: 'projects.list', items: [ project( 'p-calc', 'Calculator app', { folders: [ { path: 'C:\\calc', access: 'read_write' } ] } ),
            project( 'p-kyoto', 'Kyoto in November' ) ] },
        { type: 'conversations.list', items: [ chat( 'c-plan', 'p-calc', { progress: stopped } ), chat( 'c-trip', 'p-kyoto' ),
            chat( 'c-loose', null ), chat( 'c-cut', null, { finish: 'length' } ) ] } ] );
}

function aProjectWaitsOnThePersonWhenOneOfItsChatsNeedsThem()
{
    const state = workspace();

    assert.equal( projectState( state, 'p-calc' ), 'waiting' );
    assert.equal( projectState( state, 'p-kyoto' ), 'quiet' );
}

test( 'a project waits on the person when one of its chats needs them, and is quiet otherwise', aProjectWaitsOnThePersonWhenOneOfItsChatsNeedsThem );

function aProjectIsActiveWhileAReplyIsWrittenInOneOfItsChats()
{
    const replying = workspace( { busy: true, conversation: 'c-trip', project: 'p-kyoto' } );
    const newChat = workspace( { busy: true, conversation: null, project: 'p-kyoto' } );

    assert.equal( projectState( replying, 'p-kyoto' ), 'active' );
    assert.equal( projectState( newChat, 'p-kyoto' ), 'active' );
    assert.equal( workspaceOverview( replying ).activeCount, 1 );
}

test( 'a project is active while a reply is written in one of its chats, a new one included', aProjectIsActiveWhileAReplyIsWrittenInOneOfItsChats );

function theOverviewWithProjectsListsTheChatsInNone()
{
    const overview = workspaceOverview( workspace() );

    assert.equal( overview.projects.length, 2 );
    assert.deepEqual( overview.chats.map( ( item ) => item.id ).sort(), [ 'c-cut', 'c-loose' ] );
    assert.equal( overview.looseCount, 2 );
    assert.equal( overview.chatCount, 4 );
    assert.deepEqual( overview.waitingProjects.map( ( summary ) => summary.project.id ), [ 'p-calc' ] );
}

test( 'the overview with projects lists the chats in none, and names the projects waiting', theOverviewWithProjectsListsTheChatsInNone );

function aProjectIsAsRecentAsItsLatestChat()
{
    const state = apply( [
        { type: 'projects.list', items: [ project( 'p-old', 'Old', { updated: at( 50 * hour ) } ), project( 'p-new', 'New', { updated: at( 40 * hour ) } ) ] },
        { type: 'conversations.list', items: [ chat( 'c1', 'p-old', { updated: at( hour ), wrote: [ { path: 'a.txt', tool: 'write_file' } ] } ) ] } ] );
    const summaries = projectSummaries( state, new Date( now ) );

    assert.equal( summaries[ 0 ].project.id, 'p-old' );
    assert.equal( summaries[ 0 ].made, 1 );
    assert.equal( summaries[ 0 ].pulse.length, 14 );
    assert.equal( summaries[ 0 ].pulse[ 13 ], 1 );
}

test( 'a project is as recent as its latest chat, and counts what Mila wrote in its chats', aProjectIsAsRecentAsItsLatestChat );

function allProjectsFilterAndSort()
{
    const state = workspace();

    assert.deepEqual( matchingProjects( state, 'kyoto', 'all', 'recent' ).map( ( summary ) => summary.project.id ), [ 'p-kyoto' ] );
    assert.deepEqual( matchingProjects( state, '', 'waiting', 'recent' ).map( ( summary ) => summary.project.id ), [ 'p-calc' ] );
    assert.deepEqual( matchingProjects( state, '', 'all', 'name' ).map( ( summary ) => summary.project.name ), [ 'Calculator app', 'Kyoto in November' ] );
    assert.deepEqual( filteredChats( state, '', 'loose' ).map( ( item ) => item.id ).sort(), [ 'c-cut', 'c-loose' ] );
    assert.deepEqual( filteredChats( state, '', 'placed' ).map( ( item ) => item.id ).sort(), [ 'c-plan', 'c-trip' ] );
}

test( 'all projects are searched, filtered by state and sorted; all chats are filtered by project', allProjectsFilterAndSort );

function theBarOnAProjectsPageBeginsAChatThere()
{
    const elsewhere = reduce( reduce( workspace(), { type: 'turn.begin', turn_id: 't1', conversation_id: null } ),
        { type: 'place.open', place: 'project', project: 'p-kyoto' } );

    assert.equal( barMode( elsewhere ), 'start' );
    assert.equal( barProject( elsewhere ), 'p-kyoto' );

    const here = reduce( reduce( workspace( { project: 'p-kyoto' } ), { type: 'turn.begin', turn_id: 't1', conversation_id: null } ),
        { type: 'place.open', place: 'project', project: 'p-kyoto' } );

    assert.equal( barMode( here ), 'continue' );

    const overview = reduce( workspace( { project: 'p-kyoto' } ), { type: 'place.open', place: 'workspace' } );

    assert.equal( barMode( overview ), 'start' );
    assert.equal( barProject( overview ), null );
}

test( 'the bar on a project\'s page begins a chat there, and goes on with one already there', theBarOnAProjectsPageBeginsAChatThere );

function aProjectOpensOnlyWhileItIsStored()
{
    const unknown = reduce( workspace(), { type: 'place.open', place: 'project', project: 'p-gone' } );

    assert.equal( unknown.place, 'workspace' );

    const open = reduce( workspace(), { type: 'place.open', place: 'project', project: 'p-kyoto' } );
    const gone = reduce( open, { type: 'projects.list', items: [ project( 'p-calc', 'Calculator app' ) ] } );

    assert.equal( open.place, 'project' );
    assert.equal( gone.place, 'projects' );
    assert.equal( gone.projectId, null );

    const made = reduce( workspace(), { type: 'project.made', id: 'p-kyoto' } );

    assert.equal( made.place, 'project' );
    assert.equal( made.projectId, 'p-kyoto' );
}

test( 'a project opens only while it is stored, and one just made opens', aProjectOpensOnlyWhileItIsStored );

function activitySaysWhatHappenedToProjectsAsTheyAreNamedNow()
{
    const state = reduce( workspace(), { type: 'activity.list', before: null, items: [
        { time: at( hour ), who: 'you', kind: 'chat.moved', conversation: 'c-trip', name: 'Chat c-trip', project: 'p-kyoto', from: 'p-calc' },
        { time: at( 2 * hour ), who: 'you', kind: 'project.made', project: 'p-gone', name: 'Old name' },
        { time: at( 3 * hour ), who: 'you', kind: 'project.deleted' } ] } );
    const events = activityDays( state, 'project' ).flatMap( ( day ) => day.events );

    assert.equal( events.length, 3 );
    assert.equal( events[ 0 ].to, 'Kyoto in November' );
    assert.equal( events[ 0 ].item.id, 'c-trip' );
    assert.equal( events[ 1 ].to, null );
    assert.equal( projectEventText( events[ 0 ].recorded, events[ 0 ].to, events[ 0 ].from ), 'Moved Chat c-trip from Calculator app to Kyoto in November' );
    assert.equal( projectEventText( events[ 1 ].recorded, null, null ), 'Made the project Old name' );
    assert.equal( projectEventText( events[ 2 ].recorded, null, null ), 'Deleted a project' );
    assert.equal( projectEventText( { kind: 'chat.moved' }, null, null ), 'Moved a chat' );
    assert.equal( projectEventText( { kind: 'project.folder_added', path: 'C:\\calc', access: 'read' }, 'Calc', null ), 'Gave Calc the folder calc, to read' );
}

test( 'activity says what happened to projects, named as they are now, and nothing of what was deleted', activitySaysWhatHappenedToProjectsAsTheyAreNamedNow );

function searchPutsProjectsFirst()
{
    const results = searchResults( workspace(), 'calculator' );

    assert.equal( results[ 0 ].kind, 'project' );
    assert.equal( results[ 0 ].project.id, 'p-calc' );
}

test( 'search puts projects first', searchPutsProjectsFirst );

function aProjectsFolderIsReachedThroughTheProject()
{
    const state = reduce( workspace( { conversation: 'c-plan', project: 'p-calc', folder: 'C:\\calc', fromProject: true } ),
        { type: 'conversations.list', items: [ chat( 'c-plan', 'p-calc', { folder: 'C:\\calc', fromProject: true } ), chat( 'c-own', null, { folder: 'C:\\mine' } ) ] } );
    const reach = reachOverview( state );

    assert.deepEqual( reach.projectFolders.map( ( folder ) => [ folder.project.id, folder.path, folder.changes ] ), [ [ 'p-calc', 'C:\\calc', true ] ] );
    assert.deepEqual( reach.folders.map( ( folder ) => folder.path ), [ 'C:\\mine' ] );
}

test( 'a project\'s folder is listed under the project, not under each chat that holds it through the project', aProjectsFolderIsReachedThroughTheProject );

// ---- The mock, as the session answers ---------------------------------------------------------

function startMock()
{
    const clock = createClock( now );
    const events = [];
    const session = createMockSession( { scenario: 16, emit: ( message ) => events.push( message ), schedule: clock.schedule, cancel: clock.cancel, now: clock.now } );

    session.handle( { v: 1, type: 'ready' } );

    return { clock, events, session };
}

function latest( events, type )
{
    return events.filter( ( event ) => event.type === type ).at( -1 );
}

function theMockStartsWithThreeProjectsHoldingSomeChats()
{
    const { events } = startMock();
    const projects = latest( events, 'projects.list' ).items;
    const chats = latest( events, 'conversations.list' ).items;

    assert.equal( projects.length, 3 );
    assert.equal( chats.find( ( item ) => item.id === 'conversation-6' ).project, 'project-mock-calculator' );
    assert.equal( chats.find( ( item ) => item.id === 'conversation-2' ).project, null );
    assert.ok( latest( events, 'activity.list' ).items.length > 0 );
}

test( 'scenario 16 starts with three projects holding some of the chats', theMockStartsWithThreeProjectsHoldingSomeChats );

function aChatBegunInAProjectHoldsItsOneFolderAtItsAccess()
{
    const { events, session } = startMock();

    session.handle( { v: 1, type: 'conversation.new', project: 'project-mock-calculator' } );

    const state = latest( events, 'session.state' );

    assert.equal( state.project, 'project-mock-calculator' );
    assert.equal( state.folder, 'C:\\Users\\you\\Projects\\calc' );
    assert.equal( state.folder_access, 'read_write' );
    assert.equal( state.folder_from_project, true );

    session.handle( { v: 1, type: 'project.update', id: 'project-mock-calculator', folders: [] } );

    assert.equal( latest( events, 'session.state' ).folder, null );
}

test( 'a chat begun in a project holds its one folder at its access, and lets it go with the project', aChatBegunInAProjectHoldsItsOneFolderAtItsAccess );

function aFolderAskedForInAProjectOfSeveralIsChosenAmongThem()
{
    const { clock, events, session } = startMock();

    session.handle( { v: 1, type: 'conversation.new', project: 'project-mock-llms' } );
    session.handle( { v: 1, type: 'turn.send', stream_id: 'stream-1', text: 'What is in my project folder?', attachment_ids: [] } );
    clock.run();

    const approval = latest( events, 'turn.tool_approval' );

    assert.deepEqual( approval.folders.map( ( folder ) => folder.path ), [ 'C:\\Users\\you\\Notes\\llms', 'C:\\Users\\you\\Projects\\bench' ] );

    session.handle( { v: 1, type: 'turn.tool_decision', stream_id: 'stream-1', turn_id: approval.turn_id, call_id: approval.call_id,
        decision: 'allow_once', folder: 'C:\\Users\\you\\Projects\\bench' } );

    const state = latest( events, 'session.state' );

    assert.equal( state.folder, 'C:\\Users\\you\\Projects\\bench' );
    assert.equal( state.folder_from_project, true );
}

test( 'a folder asked for in a project of several is chosen among them', aFolderAskedForInAProjectOfSeveralIsChosenAmongThem );

function aProjectIsMadeMovedIntoAndDeletedKeepingItsChats()
{
    const { events, session } = startMock();

    session.handle( { v: 1, type: 'project.new', name: '  Garden  ', purpose: 'Beds and seeds', conversation_ids: [ 'conversation-2' ] } );

    const made = latest( events, 'project.made' ).id;

    assert.equal( latest( events, 'projects.list' ).items.find( ( item ) => item.id === made ).name, 'Garden' );
    assert.equal( latest( events, 'conversations.list' ).items.find( ( item ) => item.id === 'conversation-2' ).project, made );

    session.handle( { v: 1, type: 'conversation.move', id: 'conversation-4', project: made } );
    session.handle( { v: 1, type: 'project.delete', id: made, conversations: 'keep' } );

    const chats = latest( events, 'conversations.list' ).items;
    const activity = latest( events, 'activity.list' ).items;

    assert.equal( chats.find( ( item ) => item.id === 'conversation-2' ).project, null );
    assert.equal( chats.find( ( item ) => item.id === 'conversation-4' ).project, null );
    assert.ok( !latest( events, 'projects.list' ).items.some( ( item ) => item.id === made ) );
    assert.ok( activity.some( ( event ) => event.kind === 'project.deleted' ) );
    assert.ok( !JSON.stringify( activity ).includes( 'Garden' ) );
}

test( 'a project is made with a chat, given another, and deleted keeping its chats, its name forgotten', aProjectIsMadeMovedIntoAndDeletedKeepingItsChats );

function aProjectWithNoNameIsRefused()
{
    const { events, session } = startMock();

    session.handle( { v: 1, type: 'project.new', name: '   ' } );

    assert.equal( latest( events, 'error' ).text, 'A project needs a name.' );
    assert.equal( latest( events, 'projects.list' ).items.length, 3 );
}

test( 'a project with no name is refused', aProjectWithNoNameIsRefused );
