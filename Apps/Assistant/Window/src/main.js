// Mila's window: starts the session, hosts the page, and relays messages between them
// unchanged, except link.open, folder.choose, licence.open, app.quit and setup.showing, which it handles itself, and
// about, which it sends the page when the page is ready (MilaAssistant.md 6.2). The
// page reaches nothing else: no Node, no navigation, no new windows, no permissions but the clipboard.

import { app, BrowserWindow, dialog, ipcMain, Menu, nativeTheme, powerMonitor, session, shell } from 'electron';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { registerPageScheme, servePage, isPageUrl, pageUrl } from './page-protocol.js';
import { startSessionProcess } from './session-process.js';
import { licenceLocations, findLicences, describeLicences, readLicence } from './licences.js';
import { windowDataFolder } from './window-folder.js';

const here = path.dirname( fileURLToPath( import.meta.url ) );
const windowRoot = path.resolve( here, '..' );
const pageRoot = app.isPackaged ? path.join( process.resourcesPath, 'Ui' ) : path.resolve( here, '..', '..', 'Ui' );
const iconPath = path.join( here, '..', 'icons', 'icon.png' );
const allowedPermissions = [ 'clipboard-sanitized-write' ];
const externalProtocols = [ 'https:', 'http:', 'mailto:' ];

// The page uses no network, so the window has no reason to look for a proxy on the person's network
// (AssistantInterface.md 5.7).
app.commandLine.appendSwitch( 'no-proxy-server' );

// Before ready, as Electron requires; its session data follows.
app.setPath( 'userData', windowDataFolder( process.platform, process.env, os.homedir() ) );

let mainWindow = null;
let sessionProcess = null;
let pageReady = false;
let held = [];
let licences = [];

// The page opens on the first-run setup until it says otherwise (AssistantOnboarding.md 4.1).
let setupShowing = true;

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

/**
 * The session the package carries, or in development the copy staged for it (Window/session), unless
 * MILA_SESSION names another program. --scenario <n> (or --scenario=<n>) runs the mock instead; the mock
 * is not packaged, so a packaged build ignores it.
 */
function readConfiguration()
{
    const scenario = app.isPackaged ? null : argumentValue( '--scenario' );
    const named = process.env.MILA_SESSION;
    const program = process.platform === 'win32' ? 'mila-session.exe' : 'mila-session';
    const carried = app.isPackaged ? path.join( process.resourcesPath, 'Session', program ) : path.resolve( here, '..', 'session', program );
    const sessionPath = named === undefined || named.length === 0 ? carried : named;
    const mockScript = path.join( pageRoot, 'mock', 'stdio.js' );

    return {
        scenario,
        sessionPath: scenario === null ? sessionPath : null,
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
    const message = { v: 1, type: 'error', text: 'Mila stopped working unexpectedly. Close the window and open it again.', during: null };

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

/**
 * The folder Mila asked for (AssistantInterface.md 5.5) comes from the system's own picker, so the page never names
 * a path the person did not choose.
 */
async function chooseFolder()
{
    const options = { title: 'Choose a folder for Mila', properties: [ 'openDirectory', 'createDirectory' ] };
    const choice = await dialog.showOpenDialog( mainWindow, options );
    const chosen = choice.canceled || choice.filePaths.length === 0 ? null : choice.filePaths[ 0 ];
    const message = { v: 1, type: 'folder.chosen', path: chosen };

    deliver( message );
}

/**
 * What the window knows and the session does not: who is signed in to this computer, by their login name, the
 * app's version, and the third-party licences it carries, for the person at the sidebar's foot and About Mila
 * Assistant (AssistantInterface.md 4.1).
 */
function aboutMessage()
{
    let userName = '';

    try
    {
        userName = os.userInfo().username;
    }
    catch
    {
        userName = '';
    }

    return { v: 1, type: 'about', user_name: userName, version: app.getVersion(), licences: describeLicences( licences ) };
}

/**
 * A licence the page names by its id: Chromium's opens in the system's browser, and every other's text is sent to
 * the page. An id the window did not give names nothing.
 */
async function openLicence( id )
{
    const entry = licences.find( ( candidate ) => candidate.id === id );

    if ( entry === undefined )
        return;

    if ( entry.opens === 'browser' )
    {
        const failure = await shell.openPath( entry.file );

        if ( failure.length > 0 )
            deliver( { v: 1, type: 'licence.text', id, text: null, problem: 'Mila could not open this licence.' } );

        return;
    }

    const read = readLicence( entry );

    deliver( { v: 1, type: 'licence.text', id, text: read.text, problem: read.problem } );
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

    if ( message.type === 'folder.choose' )
    {
        chooseFolder();

        return;
    }

    if ( message.type === 'licence.open' )
    {
        openLicence( message.id );

        return;
    }

    // Declining Mila's licence is quitting.
    if ( message.type === 'app.quit' )
    {
        app.quit();

        return;
    }

    if ( message.type === 'setup.showing' )
    {
        setupShowing = message.showing === true;
        buildMenu();

        return;
    }

    if ( message.type === 'ready' )
    {
        pageReady = true;

        const waiting = held;
        held = [];

        deliver( aboutMessage() );

        for ( const waitingMessage of waiting )
            deliver( waitingMessage );
    }

    if ( sessionProcess !== null )
        sessionProcess.send( message );
}

// ---- Window --------------------------------------------------------------------------------

/**
 * Release builds have no menu bar on Windows and Linux, so no menu shortcut opens developer
 * tools; macOS keeps the menus its shortcuts for copy and paste live in. While the first-run
 * setup shows, a setup assistant's menus: the application's and Edit (AssistantOnboarding.md 4.1).
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

    const setup = [ { role: 'appMenu' }, { role: 'editMenu' } ];
    const template = setupShowing ? setup : [ ...setup, { role: 'windowMenu' } ];
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
        title: 'Mila Assistant',
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

// ---- In use or not ---------------------------------------------------------------------------

/**
 * The session holds the model's memory while Mila is the application in use and lets it go when it is not
 * (Deployment.md 12.10); macOS says which. The screen locking, the user's session going inactive and the Mac
 * going to sleep all count as not in use.
 */
function tellSession( type )
{
    if ( sessionProcess !== null )
        sessionProcess.send( { v: 1, type } );
}

function onActive()
{
    tellSession( 'app.active' );
}

function onBackground()
{
    tellSession( 'app.background' );
}

function watchUse()
{
    app.on( 'did-become-active', onActive );
    app.on( 'did-resign-active', onBackground );
    powerMonitor.on( 'lock-screen', onBackground );
    powerMonitor.on( 'user-did-resign-active', onBackground );
    powerMonitor.on( 'suspend', onBackground );
}

function onReady()
{
    const locations = licenceLocations( app.isPackaged, process.platform, process.resourcesPath, process.execPath, windowRoot );

    licences = findLicences( locations );
    session.defaultSession.setPermissionRequestHandler( onPermissionRequest );
    session.defaultSession.setPermissionCheckHandler( onPermissionCheck );
    servePage( pageRoot );
    buildMenu();
    ipcMain.on( 'mila:send', onPageMessage );
    startSession();
    watchUse();
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
