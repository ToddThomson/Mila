// The window's hold on the session: stopping it ends its input and waits for it to end, so what it was doing stops
// as Stop or Cancel would stop it before the window quits.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startSessionProcess } from '../src/session-process.js';

const here = path.dirname( fileURLToPath( import.meta.url ) );
const mockScript = path.resolve( here, '..', '..', 'Ui', 'mock', 'stdio.js' );

function startMock( onMessage )
{
    const configuration = { scenario: '8', sessionPath: null, mockScript };
    const exits = [];
    const handlers = { onMessage, onExit: ( code ) => exits.push( code ) };

    return { session: startSessionProcess( configuration, handlers ), exits };
}

async function stoppingWaitsForTheSessionToEnd()
{
    let answered = null;
    const ready = new Promise( ( resolve ) => { answered = resolve; } );
    const { session, exits } = startMock( ( message ) =>
    {
        if ( message.type === 'models.list' )
            answered();
    } );

    session.send( { v: 1, type: 'ready' } );
    await ready;

    assert.equal( await session.stop( 10000 ), true, 'the session ended within the wait' );
    assert.deepEqual( exits, [], 'an end the window asked for is not reported as the session stopping' );
    assert.equal( await session.stop( 10000 ), true, 'stopping an ended session answers at once' );
}

test( 'stopping the session waits for it to end', stoppingWaitsForTheSessionToEnd );
