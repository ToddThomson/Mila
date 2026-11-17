// The third-party licences the app carries, for About Mila Assistant's Licences (MilaDesktop.md 6.2). Kept out of
// main.js, which needs Electron, so node --test reaches it.

import fs from 'node:fs';
import path from 'node:path';

// Every text carried is a few hundred kilobytes at most; a larger file is not a licence the package put there.
const textLimitBytes = 1024 * 1024;

/**
 * Where the licences are: Electron's and Chromium's, which electron-builder leaves at the app's root on Windows and
 * Linux and the Mac's package puts in its Resources, or in development the copies beside Electron itself; and the
 * session's Licences/, one folder per project (Apps/Desktop/Session/CMakeLists.txt, mila_session_licence).
 */
export function licenceLocations( packaged, platform, resourcesPath, executablePath, windowRoot )
{
    if ( !packaged )
    {
        const electronFolder = path.join( windowRoot, 'node_modules', 'electron', 'dist' );

        return {
            electron: path.join( electronFolder, 'LICENSE' ),
            chromium: path.join( electronFolder, 'LICENSES.chromium.html' ),
            session: path.join( windowRoot, 'session', 'Licences' )
        };
    }

    const root = platform === 'darwin' ? resourcesPath : path.dirname( executablePath );

    return {
        electron: path.join( root, 'LICENSE.electron.txt' ),
        chromium: path.join( root, 'LICENSES.chromium.html' ),
        session: path.join( resourcesPath, 'Session', 'Licences' )
    };
}

function isFile( file )
{
    try
    {
        return fs.statSync( file ).isFile();
    }
    catch
    {
        return false;
    }
}

/** The session's licences, each named by its folder's path under Licences/: nlohmann/json, CUDA. */
function sessionLicences( folder )
{
    let relatives = [];

    try
    {
        relatives = fs.readdirSync( folder, { recursive: true } );
    }
    catch
    {
        return [];
    }

    const found = [];

    for ( const relative of relatives )
    {
        const file = path.join( folder, relative );

        if ( !isFile( file ) )
            continue;

        const parent = path.dirname( relative );
        const name = parent === '.' ? path.basename( relative ) : parent.split( path.sep ).join( '/' );

        found.push( { name, file, opens: 'text' } );
    }

    return found;
}

function byName( left, right )
{
    return left.name.localeCompare( right.name, 'en', { sensitivity: 'base' } );
}

/**
 * Every licence found, in the order a person reads them: Electron's and Chromium's, then the session's by name. Each
 * has an id the page names it by, so the page never names a path. Chromium's is a web page of hundreds of projects,
 * which the browser opens; every other is text the page shows.
 */
export function findLicences( locations )
{
    const electron = [];

    if ( isFile( locations.electron ) )
        electron.push( { name: 'Electron', file: locations.electron, opens: 'text' } );

    if ( isFile( locations.chromium ) )
        electron.push( { name: 'Chromium', file: locations.chromium, opens: 'browser' } );

    const session = sessionLicences( locations.session ).sort( byName );
    const all = [ ...electron, ...session ];

    return all.map( ( entry, index ) => ( { ...entry, id: String( index + 1 ) } ) );
}

/** What the page is told: each licence's id and name, and nothing of where it is. */
export function describeLicences( licences )
{
    return licences.map( ( entry ) => ( { id: entry.id, name: entry.name } ) );
}

/** A text licence's words, or why they cannot be shown. */
export function readLicence( entry )
{
    try
    {
        const size = fs.statSync( entry.file ).size;

        if ( size > textLimitBytes )
            return { text: null, problem: 'This licence is too long to show here.' };

        return { text: fs.readFileSync( entry.file, 'utf8' ), problem: null };
    }
    catch
    {
        return { text: null, problem: 'Mila could not read this licence.' };
    }
}
