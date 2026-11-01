// The mock session answers over stdio as the window will drive it: JSON lines in, JSON lines out,
// and end of input ends it.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import readline from 'node:readline';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname( fileURLToPath( import.meta.url ) );
const mockSession = path.resolve( here, '..', '..', 'Ui', 'mock', 'stdio.js' );
const patienceMilliseconds = 20000;

/**
 * A running mock session, with what it has written so far and a way to wait for a line.
 */
function startSession( scenario )
{
    const processArguments = [ mockSession, '--scenario', String( scenario ), '--speed', '20' ];
    const spawnOptions = { stdio: [ 'pipe', 'pipe', 'pipe' ] };
    const child = spawn( process.execPath, processArguments, spawnOptions );
    const lineOptions = { input: child.stdout, crlfDelay: Infinity };
    const lines = readline.createInterface( lineOptions );
    const events = [];
    const waiters = [];
    const exited = new Promise( ( resolve ) => child.on( 'exit', resolve ) );
    let errorText = '';

    function onLine( line )
    {
        const event = JSON.parse( line );
        events.push( event );

        for ( const waiter of waiters.slice() )
            waiter.check();
    }

    function onError( chunk )
    {
        errorText = errorText + chunk.toString();
    }

    lines.on( 'line', onLine );
    child.stderr.on( 'data', onError );

    function send( message )
    {
        const line = JSON.stringify( message ) + '\n';

        child.stdin.write( line );
    }

    /** Resolves with the first event, from position `from` on, that the predicate accepts. */
    function waitFor( predicate, from )
    {
        const start = from === undefined ? 0 : from;

        function executor( resolve, reject )
        {
            const timer = setTimeout( onTimeout, patienceMilliseconds );
            const waiter = { check };

            function onTimeout()
            {
                const error = new Error( 'timed out; stderr: ' + errorText );

                reject( error );
            }

            function check()
            {
                const found = events.slice( start ).find( predicate );

                if ( found === undefined )
                    return;

                clearTimeout( timer );
                waiters.splice( waiters.indexOf( waiter ), 1 );
                resolve( found );
            }

            waiters.push( waiter );
            check();
        }

        return new Promise( executor );
    }

    function stderr()
    {
        return errorText;
    }

    return { child, events, send, waitFor, exited, stderr };
}

function isType( type )
{
    return ( event ) => event.type === type;
}

async function readyIsAnsweredOverStdio()
{
    const session = startSession( 1 );
    const ready = { v: 1, type: 'ready' };

    session.send( ready );
    await session.waitFor( isType( 'models.list' ) );

    const types = session.events.map( ( event ) => event.type );
    assert.deepEqual( types, [ 'session.state', 'conversations.list', 'models.list' ] );

    session.child.stdin.end();

    const code = await session.exited;
    assert.equal( code, 0 );
}

test( 'ready is answered over stdio, and end of input ends the process', readyIsAnsweredOverStdio );

async function scenarioOnePlaysOverStdio()
{
    const session = startSession( 1 );
    const turn = { v: 1, type: 'turn.send', stream_id: 'stream-1', text: 'Weather in Toronto?', attachment_ids: [] };

    session.send( turn );

    const end = await session.waitFor( isType( 'turn.end' ) );
    assert.equal( end.finish, 'stop' );

    const calls = session.events.filter( isType( 'turn.tool_call' ) );
    assert.equal( calls.length, 1 );
    assert.equal( calls[ 0 ].name, 'get_weather' );

    for ( const event of session.events )
        assert.equal( event.v, 1 );

    session.child.stdin.end();
    await session.exited;
}

test( 'scenario 1 plays over stdio', scenarioOnePlaysOverStdio );

async function scenarioSixAcceptsAnImageOverStdio()
{
    const session = startSession( 6 );
    const add = { v: 1, type: 'attachment.add', attachment_id: 'a1', kind: 'image', name: 'harbour-at-dusk.png', path: '/pictures/harbour-at-dusk.png' };
    const turn = { v: 1, type: 'turn.send', stream_id: 'stream-1', text: 'What is in this picture?', attachment_ids: [ 'a1' ] };

    session.send( add );
    await session.waitFor( isType( 'attachment.accepted' ) );
    session.send( turn );

    const end = await session.waitFor( isType( 'turn.end' ) );
    assert.match( end.answer, /lighthouse/ );

    session.child.stdin.end();
    await session.exited;
}

test( 'scenario 6 accepts an image over stdio', scenarioSixAcceptsAnImageOverStdio );

async function stopIsReadWhileAReplyStreams()
{
    const session = startSession( 1 );
    const turn = { v: 1, type: 'turn.send', stream_id: 'stream-1', text: 'Hello', attachment_ids: [] };
    const stop = { v: 1, type: 'turn.stop', stream_id: 'stream-1' };

    session.send( turn );
    await session.waitFor( isType( 'turn.text' ) );
    session.send( stop );

    const end = await session.waitFor( isType( 'turn.end' ) );
    assert.equal( end.finish, 'cancelled' );

    session.child.stdin.end();
    await session.exited;
}

test( 'stop is read while a reply streams', stopIsReadWhileAReplyStreams );

async function badLinesAndUnknownTypesAreIgnoredAndLogged()
{
    const session = startSession( 1 );
    const unknown = { v: 1, type: 'something.new' };
    const ready = { v: 1, type: 'ready' };

    session.child.stdin.write( 'not json\n' );
    session.send( unknown );
    session.send( ready );
    await session.waitFor( isType( 'models.list' ) );

    assert.equal( session.events.length, 3 );
    assert.match( session.stderr(), /not JSON/ );
    assert.match( session.stderr(), /something\.new/ );

    session.child.stdin.end();
    await session.exited;
}

test( 'bad lines and unknown types are ignored, and logged on standard error', badLinesAndUnknownTypesAreIgnoredAndLogged );
