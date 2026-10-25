// Serves the page's own files at mila://chat/, so the page has a real origin: its modules load,
// its 'self' means its own folder, and every message from it can be checked against that origin.

import { protocol } from 'electron';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

export const pageScheme = 'mila';
export const pageHost = 'chat';
export const pageUrl = pageScheme + '://' + pageHost + '/index.html';

const contentTypes = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.json': 'application/json'
};

/**
 * Called before the app is ready, as Electron requires: a standard, secure scheme resolves
 * relative URLs and is a secure context.
 */
export function registerPageScheme()
{
    const privileges = { standard: true, secure: true, supportFetchAPI: true };
    const scheme = { scheme: pageScheme, privileges };

    protocol.registerSchemesAsPrivileged( [ scheme ] );
}

export function isPageUrl( text )
{
    try
    {
        const parsed = new URL( text );

        return parsed.protocol === pageScheme + ':' && parsed.host === pageHost;
    }
    catch
    {
        return false;
    }
}

function notFound()
{
    const options = { status: 404 };

    return new Response( 'Not found', options );
}

/**
 * Only files inside the page's folder are served; a path that climbs out of it is not found.
 */
export function servePage( pageRoot )
{
    async function handle( request )
    {
        if ( !isPageUrl( request.url ) )
            return notFound();

        const parsed = new URL( request.url );
        const relative = decodeURIComponent( parsed.pathname );
        const file = path.resolve( pageRoot, '.' + relative );
        const inside = file.startsWith( pageRoot + path.sep );
        const extension = path.extname( file ).toLowerCase();
        const contentType = contentTypes[ extension ];

        if ( !inside || contentType === undefined )
            return notFound();

        try
        {
            const body = await readFile( file );
            const headers = { 'content-type': contentType };
            const options = { headers };

            return new Response( body, options );
        }
        catch
        {
            return notFound();
        }
    }

    protocol.handle( pageScheme, handle );
}
