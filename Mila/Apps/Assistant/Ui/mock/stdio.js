// The mock session as a process, in place of the real one: MilaApp.md 4.2's requests arrive on
// standard input and 4.1's events leave on standard output, one JSON object per line. Standard error
// carries its log. End of input ends it. Node only; the page never loads this file.
//
//   node stdio.js [--scenario <n>] [--speed <factor>]
//
// --speed divides every scripted delay, so a test can play a scenario in a fraction of its time.

import readline from 'node:readline';
import { createMockSession } from './session.js';

function argumentValue( name, fallback )
{
    const index = process.argv.indexOf( name );

    if ( index < 0 || index + 1 >= process.argv.length )
        return fallback;

    return process.argv[ index + 1 ];
}

const scenario = argumentValue( '--scenario', '1' );
const speedText = argumentValue( '--speed', '1' );
const speed = Number( speedText ) > 0 ? Number( speedText ) : 1;

function log( text )
{
    process.stderr.write( 'mock session: ' + text + '\n' );
}

function emit( message )
{
    const line = JSON.stringify( message ) + '\n';

    process.stdout.write( line );
}

function schedule( callback, milliseconds )
{
    const scaled = milliseconds / speed;

    return setTimeout( callback, scaled );
}

function cancel( handle )
{
    clearTimeout( handle );
}

function now()
{
    return Date.now();
}

const options = { scenario, emit, schedule, cancel, now, log };
const session = createMockSession( options );
const inputOptions = { input: process.stdin, crlfDelay: Infinity };
const input = readline.createInterface( inputOptions );

function onLine( line )
{
    if ( line.trim().length === 0 )
        return;

    let request = null;

    try
    {
        request = JSON.parse( line );
    }
    catch
    {
        log( 'ignored a line that is not JSON' );

        return;
    }

    if ( request === null || typeof request !== 'object' || typeof request.type !== 'string' )
    {
        log( 'ignored a line with no message type' );

        return;
    }

    session.handle( request );
}

function onClose()
{
    process.exit( 0 );
}

input.on( 'line', onLine );
input.on( 'close', onClose );
log( 'scenario ' + session.scenario.number + ', ' + session.scenario.title );
