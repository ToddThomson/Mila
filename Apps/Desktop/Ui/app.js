// The page's entry: one state reduced from every message in both directions, rendered once per
// animation frame however many messages arrived in it.

import { createTransport } from './transport.js';
import { setIcon } from './dom.js';
import { createInitialState, reduce, isBusy, streamIdentifier } from './state.js';
import { createSidebarView } from './views/sidebar.js';
import { createHeaderView } from './views/header.js';
import { createTranscriptView } from './views/transcript.js';
import { createComposerView } from './views/composer.js';
import { createSettingsView } from './views/settings.js';
import { createDevelopmentView } from './views/development.js';
import { createLogView } from './views/log.js';
import { createPlanView } from './views/plan.js';
import { createRailView } from './views/rail.js';

const narrowQuery = window.matchMedia( '(max-width: 899.98px)' );

// A Mac's shortcuts are on Command; everywhere else they are on Ctrl.
const onMac = /Mac/.test( navigator.platform );
const commandKey = onMac ? 'metaKey' : 'ctrlKey';
const otherKey = onMac ? 'ctrlKey' : 'metaKey';

const transport = await createTransport( window.location );

let state = createInitialState();
let renderPending = false;
let pendingActivity = null;

const surface = {
    narrow: narrowQuery.matches,
    sidebarCollapsed: false,
    sidebarOverlay: false,
    settingsOpen: false,
    settingsArea: 'models',
    logOpen: false,
    expanded: new Set()
};

function scheduleRender()
{
    if ( renderPending )
        return;

    renderPending = true;
    window.requestAnimationFrame( render );
}

function dispatch( message )
{
    state = reduce( state, message );
    scheduleRender();
}

function send( message )
{
    dispatch( message );
    transport.send( message );
}

function changeSurface( change )
{
    Object.assign( surface, change );
    scheduleRender();
}

function toggleSidebar()
{
    if ( surface.narrow )
    {
        const overlay = { sidebarOverlay: !surface.sidebarOverlay };
        changeSurface( overlay );

        return;
    }

    const collapsed = { sidebarCollapsed: !surface.sidebarCollapsed };
    changeSurface( collapsed );
}

function toggleExpanded( identifier )
{
    if ( surface.expanded.has( identifier ) )
        surface.expanded.delete( identifier );
    else
        surface.expanded.add( identifier );

    scheduleRender();
}

function closeOverlay()
{
    const closed = { sidebarOverlay: false };

    changeSurface( closed );
}

function newConversation()
{
    const message = { type: 'conversation.new', activity: 'ask' };

    const closed = { sidebarOverlay: false, settingsOpen: false };

    send( message );
    changeSurface( closed );
    composer.focus();
}

/** Opens Settings at an area, or with none named at the one last shown. */
function openSettings( area )
{
    const settingsArea = area === undefined || area === null ? surface.settingsArea : area;
    const opened = { settingsOpen: true, settingsArea, sidebarOverlay: false };

    changeSurface( opened );
}

function closeSettings()
{
    const closed = { settingsOpen: false };

    changeSurface( closed );
}

/**
 * A card on the home screen. Ask & Explore is the conversation already open; an activity that works in
 * a folder starts once the person has chosen one in the window's picker.
 */
function startActivity( activity )
{
    if ( activity === 'ask' )
    {
        composer.focus();

        return;
    }

    // Build & Code and Get Things Done work in a folder; the others start at once.
    if ( activity !== 'build' && activity !== 'tasks' )
    {
        const start = { type: 'conversation.new', activity };
        send( start );
        composer.focus();

        return;
    }

    const choose = { type: 'folder.choose' };
    transport.send( choose );
    pendingActivity = activity;
}

function onFolderChosen( message )
{
    const activity = pendingActivity;
    pendingActivity = null;

    if ( activity === null || message.path === null )
        return;

    const start = { type: 'conversation.new', activity, folder: message.path };
    send( start );
    composer.focus();
}

function onTransportMessage( message )
{
    if ( message.type === 'folder.chosen' )
    {
        onFolderChosen( message );

        return;
    }

    dispatch( message );
}

function decideTool( turnId, callId, decision )
{
    const stream = streamIdentifier( state );
    const message = { type: 'turn.tool_decision', stream_id: stream, turn_id: turnId, call_id: callId, decision };

    send( message );
}

function openLink( url )
{
    const message = { type: 'link.open', url };

    transport.send( message );
}

const actions = {
    send,
    dispatch,
    changeSurface,
    toggleSidebar,
    toggleExpanded,
    closeOverlay,
    newConversation,
    openSettings,
    closeSettings,
    startActivity,
    decideTool,
    openLink,
    transmit: ( message ) => transport.send( message ),
    pathForFile: ( file ) => transport.pathForFile( file ),
    getState: () => state
};

const application = document.getElementById( 'app' );
const sidebarElement = document.getElementById( 'sidebar' );
const headerElement = document.getElementById( 'header' );
const transcriptScroller = document.getElementById( 'transcript-scroll' );
const transcriptElement = document.getElementById( 'transcript' );
const composerElement = document.getElementById( 'composer-area' );
const mainElement = document.getElementById( 'main' );
const settingsElement = document.getElementById( 'settings' );
const developmentElement = document.getElementById( 'development' );
const logElement = document.getElementById( 'log-drawer' );
const planElement = document.getElementById( 'plan' );

const rail = createRailView( document.getElementById( 'rail' ), actions );
const sidebar = createSidebarView( sidebarElement, actions );
const header = createHeaderView( headerElement, actions );
const transcript = createTranscriptView( transcriptScroller, transcriptElement, actions );
const composer = createComposerView( composerElement, mainElement, actions );
const settings = createSettingsView( settingsElement, actions );
const development = createDevelopmentView( developmentElement, transport.development, composer );
const log = createLogView( logElement, actions );
const plan = createPlanView( planElement );

function render()
{
    renderPending = false;

    const sidebarHidden = surface.narrow ? !surface.sidebarOverlay : surface.sidebarCollapsed;
    application.classList.toggle( 'sidebar-hidden', sidebarHidden );
    application.classList.toggle( 'sidebar-overlay', surface.narrow && surface.sidebarOverlay );
    const busy = isBusy( state );
    application.classList.toggle( 'busy', busy );

    rail.render( state, surface );
    sidebar.render( state, surface );
    header.render( state, surface );
    transcript.render( state, surface );
    composer.render( state, surface );
    settings.render( state, surface );
    log.render( state, surface );
    plan.render( state );
}

function onNarrowChange( event )
{
    const change = { narrow: event.matches, sidebarOverlay: false };

    changeSurface( change );
}

function isShortcut( event, key )
{
    return event[ commandKey ] && !event[ otherKey ] && !event.shiftKey && !event.altKey && event.key.toLowerCase() === key;
}

function onKeyDown( event )
{
    if ( isShortcut( event, 'n' ) )
    {
        event.preventDefault();
        newConversation();

        return;
    }

    if ( isShortcut( event, ',' ) )
    {
        event.preventDefault();
        openSettings( null );

        return;
    }

    if ( event.key !== 'Escape' )
        return;

    // Settings is over everything, so Esc closes it before it stops a reply.
    if ( surface.settingsOpen )
    {
        event.preventDefault();
        closeSettings();

        return;
    }

    if ( isBusy( state ) )
    {
        const stop = { type: 'turn.stop' };
        send( stop );

        return;
    }

    const closed = { sidebarOverlay: false };
    changeSurface( closed );
}

/** The shortcuts' names, in the keys of the computer the window runs on. */
function fillShortcuts()
{
    const prefix = onMac ? '⌘ ' : 'Ctrl ';

    for ( const holder of document.querySelectorAll( '[data-shortcut]' ) )
        holder.textContent = prefix + holder.dataset.shortcut;

    for ( const holder of document.querySelectorAll( '[data-shortcut-title]' ) )
    {
        const [ label, key ] = holder.dataset.shortcutTitle.split( '|' );
        holder.title = label + ' (' + prefix + key + ')';
    }
}

/**
 * A link in model text never navigates the page: the window opens it in the system browser.
 */
function onLinkActivate( event )
{
    const link = event.target.closest( '[data-url]' );

    if ( link === null )
        return;

    if ( event.type === 'keydown' && event.key !== 'Enter' )
        return;

    event.preventDefault();
    openLink( link.dataset.url );
}

function fillIcons()
{
    for ( const holder of document.querySelectorAll( '[data-icon]' ) )
        setIcon( holder, holder.dataset.icon );
}

const scrim = document.querySelector( '.scrim' );

fillIcons();
fillShortcuts();
scrim.addEventListener( 'click', closeOverlay );
narrowQuery.addEventListener( 'change', onNarrowChange );
document.addEventListener( 'keydown', onKeyDown );
document.addEventListener( 'click', onLinkActivate );
document.addEventListener( 'keydown', onLinkActivate );
transport.onMessage( onTransportMessage );

development.render();
render();

const ready = { type: 'ready' };
send( ready );
