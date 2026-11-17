// The licences About Mila Assistant lists: found where each platform's package puts them, named as NOTICE.md names
// them, and never named to the page by a path.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { licenceLocations, findLicences, describeLicences, readLicence } from '../src/licences.js';

function makePackage()
{
    const root = fs.mkdtempSync( path.join( os.tmpdir(), 'mila-licences-' ) );
    const session = path.join( root, 'Session', 'Licences' );
    const write = ( relative, text ) =>
    {
        const file = path.join( session, relative );

        fs.mkdirSync( path.dirname( file ), { recursive: true } );
        fs.writeFileSync( file, text );
    };

    write( path.join( 'nlohmann', 'json', 'LICENSE.MIT' ), 'MIT for json' );
    write( path.join( 'CUDA', 'EULA.txt' ), 'NVIDIA terms' );
    write( path.join( 'richgel999', 'miniz', 'LICENSE' ), 'MIT for miniz' );
    fs.writeFileSync( path.join( root, 'LICENSE.electron.txt' ), 'MIT for Electron' );
    fs.writeFileSync( path.join( root, 'LICENSES.chromium.html' ), '<html></html>' );

    return { root, session };
}

function eachPlatformLooksWhereItsPackagePutsThem()
{
    // Paths in this computer's own form, since the window splits them with its own path module.
    const app = path.join( path.sep, 'Mila' );
    const contents = path.join( path.sep, 'Mila.app', 'Contents' );
    const windows = licenceLocations( true, 'win32', path.join( app, 'resources' ), path.join( app, 'Mila.exe' ), 'unused' );
    const mac = licenceLocations( true, 'darwin', path.join( contents, 'Resources' ), path.join( contents, 'MacOS', 'Mila' ), 'unused' );
    const linux = licenceLocations( true, 'linux', path.join( app, 'resources' ), path.join( app, 'mila' ), 'unused' );

    assert.equal( windows.electron, path.join( app, 'LICENSE.electron.txt' ) );
    assert.equal( windows.session, path.join( app, 'resources', 'Session', 'Licences' ) );
    assert.equal( mac.electron, path.join( contents, 'Resources', 'LICENSE.electron.txt' ) );
    assert.equal( mac.chromium, path.join( contents, 'Resources', 'LICENSES.chromium.html' ) );
    assert.equal( mac.session, path.join( contents, 'Resources', 'Session', 'Licences' ) );
    assert.equal( linux.electron, path.join( app, 'LICENSE.electron.txt' ) );
    assert.equal( linux.chromium, path.join( app, 'LICENSES.chromium.html' ) );

    const development = licenceLocations( false, 'win32', 'unused', 'unused', '/window' );

    assert.equal( development.electron, path.join( '/window', 'node_modules', 'electron', 'dist', 'LICENSE' ) );
    assert.equal( development.session, path.join( '/window', 'session', 'Licences' ) );
}

test( 'each platform looks where its package puts the licences', eachPlatformLooksWhereItsPackagePutsThem );

function eachIsNamedByItsFolderElectronsFirst()
{
    const { root, session } = makePackage();
    const locations = { electron: path.join( root, 'LICENSE.electron.txt' ), chromium: path.join( root, 'LICENSES.chromium.html' ), session };
    const found = findLicences( locations );

    assert.deepEqual( found.map( ( entry ) => entry.name ), [ 'Electron', 'Chromium', 'CUDA', 'nlohmann/json', 'richgel999/miniz' ] );
    assert.deepEqual( found.map( ( entry ) => entry.opens ), [ 'text', 'browser', 'text', 'text', 'text' ] );
    assert.equal( new Set( found.map( ( entry ) => entry.id ) ).size, found.length, 'every id is its own' );

    const told = describeLicences( found );

    assert.deepEqual( Object.keys( told[ 0 ] ).sort(), [ 'id', 'name' ], 'the page is told no path' );
    assert.equal( readLicence( found[ 3 ] ).text, 'MIT for json' );
}

test( 'each licence is named by its folder, Electron and Chromium first', eachIsNamedByItsFolderElectronsFirst );

function whatIsMissingIsLeftOut()
{
    const { root } = makePackage();
    const locations = { electron: path.join( root, 'absent' ), chromium: path.join( root, 'absent.html' ), session: path.join( root, 'no-such-folder' ) };

    assert.deepEqual( findLicences( locations ), [] );
}

test( 'a licence that is not there is left out', whatIsMissingIsLeftOut );

function aTextThatCannotBeReadSaysSo()
{
    const { root } = makePackage();
    const gone = { file: path.join( root, 'gone.txt' ) };
    const large = path.join( root, 'large.txt' );

    fs.writeFileSync( large, Buffer.alloc( 1024 * 1024 + 1, 'a' ) );

    assert.equal( readLicence( gone ).text, null );
    assert.equal( readLicence( gone ).problem, 'Mila could not read this licence.' );
    assert.equal( readLicence( { file: large } ).problem, 'This licence is too long to show here.' );
}

test( 'a licence that cannot be read says so', aTextThatCannotBeReadSaysSo );
