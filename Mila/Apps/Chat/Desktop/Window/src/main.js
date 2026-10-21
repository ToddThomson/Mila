// Mila Chat's window: starts Chat's session, hosts the page, and relays messages between them
// unchanged, except link.open, which it handles itself (ChatDesktop.md 6.2). The page reaches
// nothing else: no Node, no navigation, no new windows, no permissions but the clipboard.

import { app, BrowserWindow, ipcMain, Menu, nativeTheme, session, shell } from 'electron';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { registerPageScheme, servePage, isPageUrl, pageUrl } from './page-protocol.js';
import { startSessionProcess } from './session-process.js';

const here = path.dirname( fileURLToPath( import.meta.url ) );
const pageRoot = app.isPackaged ? path.join( process.resourcesPath, 'Ui' ) : path.resolve( here, '..', '..', 'Ui' );
const iconPath = path.join( here, '..', 'icons', 'icon.png' );
const allowedPermissions = [ 'clipboard-sanitized-write' ];
const externalProtocols = [ 'https:', 'http:', 'mailto:' ];

let mainWindow = null;
let sessionProcess = null;
let pageReady = false;
let held = [];

/**
 * --scenario <n> (or --scenario=<n>) picks the mock's scenario; MILA_CHAT_SESSION names the real
 * session program, and without it the mock runs.
 */
function argumentValue( name )
{
    const prefix = name + '=';

    for ( let index = 0; index < process.argv.length; index = index + 1 )
    {
        const value = process.argv[ index ];

        if ( value.startsWith( prefix ) )
            return value.slice( prefix.length );

        if ( value === name && index + 1 < process.argv.length )
            return process.argv[ index + 1 ];
    }

    return null;
}

function readConfiguration()
{
    const scenario = argumentValue( '--scenario' );
    const sessionPath = process.env.MILA_CHAT_SESSION;
    const mockScript = path.join( pageRoot, 'mock', 'stdio.js' );

    return {
        scenario: scenario === null ? '1' : scenario,
        sessionPath: sessionPath === undefined || sessionPath.length === 0 ? null : sessionPath,
        mockScript
    };
}

// ---- Relay ---------------------------------------------------------------------------------

function deliver( message )
{
    if ( mainWindow === null || mainWindow.isDestroyed() )
        return;

    mainWindow.webContents.send( 'mila:message', message );
}

/**
 * Events the session writes before the page has said ready are held, so a page that is still
 * loading misses nothing.
 */
function onSessionMessage( message )
{
    if ( !pageReady )
    {
        held.push( message );

        return;
    }

    deliver( message );
}

function onSessionExit()
{
    const message = { v: 1, type: 'error', text: 'Chat stopped working unexpectedly. Close the window and open it again.', during: null };

    sessionProcess = null;
    onSessionMessage( message );
}

function openLink( url )
{
    let parsed = null;

    try
    {
        parsed = new URL( url );
    }
    catch
    {
        return;
    }

    if ( !externalProtocols.includes( parsed.protocol ) )
        return;

    const external = parsed.toString();
    shell.openExternal( external );
}

function isFromPage( event )
{
    const frame = event.senderFrame;

    if ( mainWindow === null || frame === null || frame !== mainWindow.webContents.mainFrame )
        return false;

    return isPageUrl( frame.url );
}

function onPageMessage( event, message )
{
    if ( !isFromPage( event ) || message === null || typeof message !== 'object' || typeof message.type !== 'string' )
        return;

    if ( message.type === 'link.open' )
    {
        openLink( message.url );

        return;
    }

    if ( message.type === 'ready' )
    {
        pageReady = true;

        const waiting = held;
        held = [];

        for ( const waitingMessage of waiting )
            deliver( waitingMessage );
    }

    if ( sessionProcess !== null )
        sessionProcess.send( message );
}

// ---- Window --------------------------------------------------------------------------------

/**
 * Release builds have no menu bar on Windows and Linux, so no menu shortcut opens developer
 * tools; macOS keeps the menus its shortcuts for copy and paste live in.
 */
function buildMenu()
{
    if ( !app.isPackaged )
        return;

    if ( process.platform !== 'darwin' )
    {
        Menu.setApplicationMenu( null );

        return;
    }

    const template = [ { role: 'appMenu' }, { role: 'editMenu' }, { role: 'windowMenu' } ];
    const menu = Menu.buildFromTemplate( template );
    Menu.setApplicationMenu( menu );
}

function denyNavigation( event )
{
    event.preventDefault();
}

function denyWindow()
{
    return { action: 'deny' };
}

function onPermissionRequest( contents, permission, callback )
{
    const allowed = allowedPermissions.includes( permission );

    callback( allowed );
}

function onPermissionCheck( contents, permission )
{
    return allowedPermissions.includes( permission );
}

function createWindow()
{
    const preload = path.join( here, 'preload.cjs' );

    const webPreferences = {
        preload,
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: true,
        devTools: !app.isPackaged,
        spellcheck: false
    };

    const options = {
        width: 1280,
        height: 800,
        minWidth: 560,
        minHeight: 480,
        title: 'Mila Chat',
        show: false,
        backgroundColor: nativeTheme.shouldUseDarkColors ? '#121215' : '#ffffff',
        webPreferences
    };

    // Windows and Linux take the window's icon from here; macOS takes it from the app bundle.
    if ( process.platform !== 'darwin' )
        options.icon = iconPath;

    mainWindow = new BrowserWindow( options );

    const contents = mainWindow.webContents;
    const show = () => mainWindow.show();

    contents.on( 'will-navigate', denyNavigation );
    contents.on( 'will-redirect', denyNavigation );
    contents.setWindowOpenHandler( denyWindow );
    mainWindow.once( 'ready-to-show', show );
    mainWindow.loadURL( pageUrl );
}

function startSession()
{
    const configuration = readConfiguration();
    const handlers = { onMessage: onSessionMessage, onExit: onSessionExit };

    sessionProcess = startSessionProcess( configuration, handlers );
}

function stopSession()
{
    if ( sessionProcess === null )
        return;

    sessionProcess.stop();
    sessionProcess = null;
}

function onReady()
{
    session.defaultSession.setPermissionRequestHandler( onPermissionRequest );
    session.defaultSession.setPermissionCheckHandler( onPermissionCheck );
    servePage( pageRoot );
    buildMenu();
    ipcMain.on( 'mila:send', onPageMessage );
    startSession();
    createWindow();
}

function onAllClosed()
{
    stopSession();
    app.quit();
}

registerPageScheme();
app.on( 'window-all-closed', onAllClosed );
app.on( 'before-quit', stopSession );
app.whenReady().then( onReady );
