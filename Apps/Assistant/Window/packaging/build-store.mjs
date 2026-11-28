// node packaging/build-store.mjs
//
// Builds the Microsoft Store package with the installed Windows SDK's makeappx: the one in electron-builder's
// downloaded Windows Kits bundle fails to start ("side-by-side configuration is incorrect", 2026-11-27).

import { spawnSync } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';
import path from 'node:path';

const programFiles = process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)';
const binRoot = path.join( programFiles, 'Windows Kits', '10', 'bin' );

const versions = existsSync( binRoot )
    ? readdirSync( binRoot )
        .filter( ( name ) => /^10\.\d+\.\d+\.\d+$/.test( name ) )
        .filter( ( name ) => existsSync( path.join( binRoot, name, 'x64', 'makeappx.exe' ) ) )
        .sort( ( a, b ) => a.localeCompare( b, undefined, { numeric: true } ) )
    : [];

if ( versions.length === 0 ) {
    console.error( `No Windows SDK with makeappx.exe under ${binRoot}: install the Windows SDK with Visual Studio.` );
    process.exit( 1 );
}

const kit = path.join( binRoot, versions[ versions.length - 1 ], 'x64' );
console.log( `Windows SDK tools: ${kit}` );

const result = spawnSync( 'npx', [ 'electron-builder', '--win', 'appx', '--x64', '--publish', 'never' ], {
    stdio: 'inherit',
    shell: true,
    env: { ...process.env, ELECTRON_BUILDER_WINDOWS_KITS_PATH: kit },
} );

process.exit( result.status ?? 1 );
