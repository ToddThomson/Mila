// Starts the session as a child process and speaks JSON lines with it (MilaApp.md 4,
// Transport): requests on its standard input, events from its standard output, its log on
// standard error. Ending its input ends it.

import { spawn } from 'node:child_process';
import readline from 'node:readline';

/**
 * The session program, or the mock session when a scenario is asked for, run by this Electron's
 * own Node so nothing else needs installing.
 */
function commandFor( configuration )
{
    if ( configuration.sessionPath !== null )
    {
        const realEnvironment = { ...process.env };

        return { file: configuration.sessionPath, arguments: [], environment: realEnvironment };
    }

    const mockArguments = [ configuration.mockScript, '--scenario', configuration.scenario ];
    const mockEnvironment = { ...process.env, ELECTRON_RUN_AS_NODE: '1' };

    return { file: process.execPath, arguments: mockArguments, environment: mockEnvironment };
}

/**
 * @param handlers.onMessage Given each event the session writes, parsed.
 * @param handlers.onExit Given the exit code when the session ends.
 */
export function startSessionProcess( configuration, handlers )
{
    const command = commandFor( configuration );
    const options = { stdio: [ 'pipe', 'pipe', 'pipe' ], env: command.environment, windowsHide: true };
    const child = spawn( command.file, command.arguments, options );
    const lineOptions = { input: child.stdout, crlfDelay: Infinity };
    const lines = readline.createInterface( lineOptions );
    let ending = false;

    function onLine( line )
    {
        if ( line.trim().length === 0 )
            return;

        let message = null;

        try
        {
            message = JSON.parse( line );
        }
        catch
        {
            process.stderr.write( 'window: ignored a line from the session that is not JSON\n' );

            return;
        }

        handlers.onMessage( message );
    }

    function onLog( chunk )
    {
        process.stderr.write( chunk );
    }

    function onExit( code )
    {
        if ( !ending )
            handlers.onExit( code );
    }

    function onSpawnError( error )
    {
        process.stderr.write( 'window: the session could not start: ' + error.message + '\n' );
        onExit( null );
    }

    lines.on( 'line', onLine );
    child.stderr.on( 'data', onLog );
    child.on( 'exit', onExit );
    child.on( 'error', onSpawnError );

    function send( message )
    {
        if ( ending || !child.stdin.writable )
            return;

        const line = JSON.stringify( message ) + '\n';
        child.stdin.write( line );
    }

    function stop()
    {
        ending = true;

        if ( child.stdin.writable )
            child.stdin.end();
    }

    return { send, stop };
}
