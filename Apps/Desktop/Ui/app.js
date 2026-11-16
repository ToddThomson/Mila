// The page's entry: one state reduced from every message in both directions, rendered once per
// animation frame however many messages arrived in it.

import { createTransport } from './transport.js';
import { createInitialState, reduce, isBusy, isWelcoming, isAskingForModel, isRisen, barMode, streamIdentifier } from './state.js';
import { createSidebarView } from './views/sidebar.js';
import { createHeaderView } from './views/header.js';
import { createPlacesView } from './views/places.js';
import { createChatHeadView } from './views/chat-head.js';
import { createTranscriptView } from './views/transcript.js';
import { createDockView } from './views/dock.js';
import { createComposerView } from './views/composer.js';
import { createSettingsView } from './views/settings.js';
import { createSearchView } from './views/search.js';
import { createLicenseView } from './views/license.js';
import { createDevelopmentView } from './views/development.js';
import { createLogView } from './views/log.js';
import { createPlanView } from './views/plan.js';

const narrowQuery = window.matchMedia( '(max-width: 899.98px)' );

// A Mac's shortcuts are on Command; everywhere else they are on Ctrl.
const onMac = /Mac/.test( navigator.platform );
const commandKey = onMac ? 'metaKey' : 'ctrlKey';
const otherKey = onMac ? 'ctrlKey' : 'metaKey';

const transport = await createTransport( window.location );

let state = createInitialState();
let renderPending = false;
let setupShown = null;
let shownPlace = null;

// A turn to send once a chat opened from a button has loaded: "Ask Mila to finish".
let afterOpen = null;

const surface = {
    narrow: narrowQuery.matches,
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
    changeSurface( { sidebarOverlay: !surface.sidebarOverlay } );
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
    changeSurface( { sidebarOverlay: false } );
}

/** Going somewhere from the sidebar, the path or a figure; a narrow window's sidebar closes once the person has. */
function openPlace( place )
{
    dispatch( { type: 'place.open', place } );
    changeSurface( { sidebarOverlay: false, settingsOpen: false } );
}

/** A chat opened from anywhere has the pane to itself; `then` is a turn to send once it has loaded. */
function openConversation( identifier, then )
{
    afterOpen = then === undefined ? null : { id: identifier, text: then };

    // The chat already open is shown whole without being read again from the store.
    if ( identifier === state.conversationId && state.transcript.length > 0 && afterOpen === null )
        dispatch( { type: 'place.open', place: 'chat' } );
    else
        send( { type: 'conversation.open', id: identifier } );

    changeSurface( { sidebarOverlay: false, settingsOpen: false } );
}

/** A new conversation is the bar's; from a chat the person goes back to the workspace to begin it. */
function newConversation()
{
    send( { type: 'conversation.new', activity: 'ask' } );

    if ( state.place === 'chat' )
        dispatch( { type: 'place.open', place: 'workspace' } );

    changeSurface( { sidebarOverlay: false, settingsOpen: false } );
    composer.focus();
}

/**
 * What the bar sends (AssistantInterface.md 5.3): a reply in the open chat; elsewhere the next turn of the bar's
 * conversation, or the first of a new chat when the session holds one opened from a list. The attachments waiting
 * in the bar go with the new chat.
 */
function sendTurn( text, attachmentIds )
{
    if ( barMode( state ) === 'start' )
    {
        const waiting = state.attachments;

        send( { type: 'conversation.new', activity: 'ask' } );
        dispatch( { type: 'attachments.keep', attachments: waiting } );
    }

    const stream = streamIdentifier( state );

    send( { type: 'turn.send', stream_id: stream, text, attachment_ids: attachmentIds } );
}

function stopTurn()
{
    const stream = streamIdentifier( state );

    send( { type: 'turn.stop', stream_id: stream } );
}

/** Opens Settings at an area, or with none named at the one last shown. */
function openSettings( area )
{
    const settingsArea = area === undefined || area === null ? surface.settingsArea : area;

    changeSurface( { settingsOpen: true, settingsArea, sidebarOverlay: false } );
}

function closeSettings()
{
    changeSurface( { settingsOpen: false } );
}

function onTransportMessage( message )
{
    if ( message.type === 'folder.chosen' )
        return;

    dispatch( message );

    if ( message.type === 'conversation.loaded' && afterOpen !== null && afterOpen.id === message.id )
    {
        const text = afterOpen.text;

        afterOpen = null;
        sendTurn( text, [] );
    }
}

function decideTool( turnId, callId, decision )
{
    const stream = streamIdentifier( state );

    send( { type: 'turn.tool_decision', stream_id: stream, turn_id: turnId, call_id: callId, decision } );
}

function openLink( url )
{
    transport.send( { type: 'link.open', url } );
}

/** A shortcut's name in the keys of the computer the window runs on. */
function shortcutLabel( key )
{
    return ( onMac ? '⌘' : 'Ctrl ' ) + key;
}

const actions = {
    send,
    dispatch,
    changeSurface,
    toggleSidebar,
    toggleExpanded,
    closeOverlay,
    openPlace,
    openConversation,
    newConversation,
    sendTurn,
    stopTurn,
    openSettings,
    closeSettings,
    openSearch: () => search.open(),
    openLicense: () => license.open(),
    decideTool,
    openLink,
    shortcutLabel,
    transmit: ( message ) => transport.send( message ),
    pathForFile: ( file ) => transport.pathForFile( file ),
    getState: () => state
};

const application = document.getElementById( 'app' );
const mainElement = document.getElementById( 'main' );
const paneScroller = document.getElementById( 'pane-scroll' );
const transcriptScroller = document.getElementById( 'transcript-scroll' );
const conversationElement = document.getElementById( 'conversation' );
const dockElement = document.getElementById( 'dock' );
const dockFade = mainElement.querySelector( '.dock-fade' );
const planElement = document.getElementById( 'plan' );
const logElement = document.getElementById( 'log-drawer' );

const sidebar = createSidebarView( document.getElementById( 'sidebar' ), actions );
const header = createHeaderView( document.getElementById( 'header' ), actions );
const places = createPlacesView( paneScroller, document.getElementById( 'pane' ), actions );
const chatHead = createChatHeadView( document.getElementById( 'chat-head' ) );
const transcript = createTranscriptView( transcriptScroller, document.getElementById( 'transcript' ), actions );
const dock = createDockView( document.getElementById( 'conversation-head' ), actions );
const composer = createComposerView( document.getElementById( 'composer-area' ), mainElement, actions );
const settings = createSettingsView( document.getElementById( 'settings' ), actions );
const search = createSearchView( document.getElementById( 'search' ), actions );
const license = createLicenseView( document.getElementById( 'license-window' ), actions );
const development = createDevelopmentView( sidebar.development, transport.development, composer );
const log = createLogView( logElement, actions );
const plan = createPlanView( planElement );

/**
 * What the pane shows: the first-run setup, which has the window to itself; the screen asking for a model, in the
 * workspace's place while none is loaded; the open chat; or a place, with the conversation risen over it or not.
 */
function paneMode()
{
    if ( isWelcoming( state ) )
        return 'setup';

    if ( state.place === 'workspace' && isAskingForModel( state ) )
        return 'model';

    return state.place === 'chat' ? 'chat' : 'place';
}

/** The transcript is the chat's pane, or the risen conversation's: one element, moved to where it shows. */
function placeTranscript( risen )
{
    const home = risen ? conversationElement : mainElement;

    if ( transcriptScroller.parentElement === home )
        return false;

    if ( risen )
        conversationElement.append( transcriptScroller );
    else
        mainElement.insertBefore( transcriptScroller, dockFade );

    return true;
}

function render()
{
    renderPending = false;

    const mode = paneMode();
    const risen = mode === 'place' && isRisen( state );

    application.dataset.mode = mode;
    application.classList.toggle( 'risen', risen );
    application.classList.toggle( 'sidebar-overlay', surface.narrow && surface.sidebarOverlay );
    application.classList.toggle( 'busy', isBusy( state ) );

    const welcoming = mode === 'setup';

    // The window gives a setup assistant's menus while the setup shows.
    if ( welcoming !== setupShown )
    {
        setupShown = welcoming;
        transport.send( { type: 'setup.showing', showing: welcoming } );
    }

    if ( state.place !== shownPlace )
    {
        places.reset();
        paneScroller.scrollTop = 0;
        shownPlace = state.place;
    }

    const moved = placeTranscript( risen );

    sidebar.render( state, surface );
    header.render( state, surface );
    places.render( state );
    chatHead.render( state );
    transcript.render( state, surface );
    dock.render( state );
    composer.render( state, surface );
    settings.render( state, surface );
    log.render( state, surface );
    plan.render( state );

    // A moved transcript starts again at its latest turn.
    if ( moved )
        transcriptScroller.scrollTop = transcriptScroller.scrollHeight;

    document.getElementById( 'composer-area' ).hidden = mode === 'setup' || mode === 'model';

    const conversing = mode === 'chat' || risen;

    planElement.hidden = planElement.hidden || !conversing;
    logElement.hidden = logElement.hidden || !conversing;
}

// The pane's content scrolls beneath the bar, so it is padded by the bar's height, plan and log included.
const dockSize = new ResizeObserver( () =>
{
    mainElement.style.setProperty( '--dock-height', dockElement.offsetHeight + 'px' );
} );

dockSize.observe( dockElement );

function onNarrowChange( event )
{
    changeSurface( { narrow: event.matches, sidebarOverlay: false } );
}

function isShortcut( event, key )
{
    return event[ commandKey ] && !event[ otherKey ] && !event.shiftKey && !event.altKey && event.key.toLowerCase() === key;
}

function onKeyDown( event )
{
    const shortcut = isShortcut( event, 'n' ) || isShortcut( event, ',' ) || isShortcut( event, 'k' );

    // The setup's panes are the one way forward: no new conversation, Search or Settings beside them.
    if ( isWelcoming( state ) && shortcut )
    {
        event.preventDefault();

        return;
    }

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

    if ( isShortcut( event, 'k' ) )
    {
        event.preventDefault();
        search.open();

        return;
    }

    if ( event.key !== 'Escape' )
        return;

    // Search and the licence are dialogs, which Esc closes itself.
    if ( search.isOpen() || license.isOpen() )
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
        stopTurn();

        return;
    }

    if ( isRisen( state ) )
    {
        dispatch( { type: 'conversation.raise', risen: false } );

        return;
    }

    closeOverlay();
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

/** Focusing the bar with a conversation in it raises the conversation again. */
function onBarFocus()
{
    if ( state.place !== 'chat' && state.transcript.length > 0 && state.barOwned && !state.risen )
        dispatch( { type: 'conversation.raise', risen: true } );
}

const scrim = document.querySelector( '.scrim' );

scrim.addEventListener( 'click', closeOverlay );
narrowQuery.addEventListener( 'change', onNarrowChange );
document.addEventListener( 'keydown', onKeyDown );
document.addEventListener( 'click', onLinkActivate );
document.addEventListener( 'keydown', onLinkActivate );
document.getElementById( 'composer-area' ).addEventListener( 'focusin', onBarFocus );
transport.onMessage( onTransportMessage );

development.render();
render();

send( { type: 'ready' } );
