// What the page shows before anything leaves the computer, and of what Mila can reach (AssistantInterface.md 4.12, 5.7).

import test from 'node:test';
import assert from 'node:assert/strict';
import { createInitialState, reduce, activityDays, reachOverview } from '../state.js';
import { activityText, approvalWords, linkDescription } from '../format.js';

function factsOf( description )
{
    return Object.fromEntries( description.facts );
}

test( 'a link shows its site and its whole address, whatever it was labelled', () =>
{
    const url = 'https://evil.test:8443/collect?note=my%20private%20words#end';
    const description = linkDescription( url );

    assert.equal( factsOf( description ).Site, 'evil.test:8443' );
    assert.equal( factsOf( description )[ 'Full address' ], url );
} );

test( 'a link over http says it is not encrypted', () =>
{
    assert.match( linkDescription( 'http://example.test/' ).note, /not encrypted/ );
} );

test( 'a mail link shows who it goes to and everything it would fill in', () =>
{
    const facts = factsOf( linkDescription( 'mailto:a@example.test?subject=Hello&body=My%20private%20words&bcc=b@example.test' ) );

    assert.equal( facts.To, 'a@example.test' );
    assert.equal( facts.Subject, 'Hello' );
    assert.equal( facts.Message, 'My private words' );
    assert.equal( facts[ 'Hidden copy to' ], 'b@example.test' );
} );

test( 'what cannot be opened has no sheet', () =>
{
    assert.equal( linkDescription( 'file:///c:/secret.txt' ), null );
    assert.equal( linkDescription( 'not an address' ), null );
} );

test( 'a page read and a search show exactly what would go', () =>
{
    const page = approvalWords( { name: 'read_web_page', arguments: '{"url":"http://example.test/?q=1"}', changesFolder: null } );
    const search = approvalWords( { name: 'web_search', arguments: '{"query":"my words"}', changesFolder: null } );

    assert.deepEqual( page.facts, [ [ 'Address', 'http://example.test/?q=1' ] ] );
    assert.match( page.notes[ 0 ], /not encrypted/ );
    assert.deepEqual( search.facts, [ [ 'Search words', 'my words' ] ] );
    assert.match( search.notes[ 0 ], /search provider/ );
} );

test( 'the first change in a folder says it also allows changes there', () =>
{
    const words = approvalWords( { name: 'write_file', arguments: '{"path":"a.txt"}', changesFolder: 'D:\\Notes' } );

    assert.equal( words.question, 'Allow this change?' );
    assert.match( words.notes[ 0 ], /change files in D:\\Notes for the rest of this chat/ );
} );

test( 'the question keeps the folder a call would open to changes', () =>
{
    let state = reduce( createInitialState(), { type: 'turn.send', stream_id: 'stream-1', text: 'Fix it', attachment_ids: [] } );

    state = reduce( state, { type: 'turn.begin', stream_id: 'stream-1', turn_id: 't1', parent_turn_id: null } );
    state = reduce( state, { type: 'turn.tool_call', turn_id: 't1', round: 1, call_id: 'c', name: 'write_file', summary: '', arguments: '{}' } );
    state = reduce( state, { type: 'turn.tool_approval', turn_id: 't1', call_id: 'c', choices: [ 'allow_once', 'deny' ], changes_folder: 'D:\\Notes' } );

    const turn = state.transcript.find( ( entry ) => entry.blocks !== undefined );

    assert.equal( turn.blocks.find( ( block ) => block.kind === 'tool' ).changesFolder, 'D:\\Notes' );
} );

test( 'Activity lists what left the computer, at the time it left', () =>
{
    const item = { id: 'a', title: 'Trip', created: '2026-11-18T09:00:00.000Z', updated: '2026-11-18T12:00:00.000Z', progress: null, wrote: [],
        sent: [ { kind: 'page', what: 'https://example.test/', where: 'example.test', time: '2026-11-18T10:00:00.000Z' },
            { kind: 'link', what: 'https://link.test/', where: 'link.test', time: '2026-11-18T11:00:00.000Z' } ] };
    const state = { ...createInitialState(), conversations: [ item ] };
    const events = activityDays( state, 'sent' ).flatMap( ( day ) => day.events );

    assert.deepEqual( events.map( ( event ) => event.who ), [ 'you', 'mila' ] );
    assert.equal( activityText( events[ 1 ] ), 'Read https://example.test/' );
    assert.equal( activityText( events[ 0 ] ), 'Opened https://link.test/' );
} );

test( 'what Mila can reach lists every folder, the open chat with the access it has now', () =>
{
    const conversations = [ { id: 'open', title: 'Open', folder: 'D:\\A' }, { id: 'other', title: 'Other', folder: 'D:\\B' }, { id: 'none', title: 'None', folder: null } ];
    const session = { folder: 'D:\\A', folder_access: 'read_write', allowed_for_chat: [ 'read_file' ], web_search: true, run_commands: false,
        model: null, streams: [ { conversation_id: 'open' } ] };
    const state = reduce( { ...createInitialState(), conversations }, { type: 'session.state', ...session } );
    const reach = reachOverview( state );

    assert.deepEqual( reach.folders.map( ( folder ) => [ folder.id, folder.changes ] ), [ [ 'open', true ], [ 'other', false ] ] );
    assert.deepEqual( reach.allowed, [ 'read_file' ] );
    assert.equal( reach.webSearch, true );
    assert.equal( reach.commands, false );
} );

test( 'What Mila can reach is a place the person can go', () =>
{
    assert.equal( reduce( createInitialState(), { type: 'place.open', place: 'reach' } ).place, 'reach' );
} );
