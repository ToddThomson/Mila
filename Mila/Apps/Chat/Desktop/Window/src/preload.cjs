// The page's bridge, and the page's only reach beyond itself: send, onMessage and pathForFile
// (ChatDesktop.md 6.2). Runs sandboxed, so it may require only what Electron's sandbox allows.

const { contextBridge, ipcRenderer, webUtils } = require( 'electron' );

function send( message )
{
    ipcRenderer.send( 'mila:send', message );
}

function onMessage( handler )
{
    const forward = ( event, message ) => handler( message );

    ipcRenderer.on( 'mila:message', forward );
}

/**
 * The file's path on this computer, or null for a File that is not backed by one -- Electron
 * returns an empty string for those.
 */
function pathForFile( file )
{
    const found = webUtils.getPathForFile( file );

    return found.length > 0 ? found : null;
}

const bridge = { send, onMessage, pathForFile };

contextBridge.exposeInMainWorld( 'mila', bridge );
