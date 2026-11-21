// After a package is built: it carries Electron's and Chromium's licences and the session's, or the build fails.
// electron-builder only warns when a file it was asked to copy is missing, and removes Electron's own from a Mac app.
// node packaging/check-licences.mjs <windows | linux | mac>

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const windowRoot = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const manifest = JSON.parse( fs.readFileSync( path.join( windowRoot, 'package.json' ), 'utf8' ) );
const product = manifest.build.productName;

// Where electron-builder's unpacked app puts its root and its resources, by platform.
const layouts = {
    windows: { root: 'dist/win-unpacked', resources: 'dist/win-unpacked/resources' },
    linux: { root: 'dist/linux-unpacked', resources: 'dist/linux-unpacked/resources' },
    mac: { root: `dist/mac-arm64/${product}.app/Contents/Resources`, resources: `dist/mac-arm64/${product}.app/Contents/Resources` }
};

const platform = process.argv[ 2 ];
const layout = layouts[ platform ];

if ( layout === undefined )
{
    console.error( 'Name the platform the package was built for: windows, linux or mac.' );
    process.exit( 1 );
}

const required = [
    path.join( layout.root, 'LICENSE.electron.txt' ),
    path.join( layout.root, 'LICENSES.chromium.html' ),
    path.join( layout.resources, 'Session', 'Licences' )
];

const missing = required.filter( ( relative ) => !fs.existsSync( path.join( windowRoot, relative ) ) );

if ( missing.length > 0 )
{
    console.error( 'The package carries the app without these licences: ' + missing.join( ', ' ) );
    process.exit( 1 );
}

console.log( 'The package carries Electron\'s, Chromium\'s and the session\'s licences.' );
