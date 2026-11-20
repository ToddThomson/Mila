// The page's entry: one state reduced from every message in both directions, rendered once per
// animation frame however many messages arrived in it.

import { createTransport } from './transport.js';
import { createInitialState, reduce, isBusy, isWelcoming, isAskingForModel, isRisen, barMode, barProject, streamIdentifier } from './state.js';
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
import { createAboutView } from './views/about.js';
import { createLinkView } from './views/link.js';
import { createNewProjectView } from './views/new-project.js';
import { createDevelopmentView } from './views/development.js';
import { createLogView } from './views/log.js';
import { createPlanView } from './views/plan.js';
import { createMadeView } from './views/made.js';

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

// What the window's picker is choosing a folder for, until it answers: the request_folder call waiting on it, or a
// project being given one.
let folderFor = null;

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

/** A project opens on its summary (4.5). */
function openProject( identifier )
{
    dispatch( { type: 'place.open', place: 'project', project: identifier } );
    changeSurface( { sidebarOverlay: false, settingsOpen: false } );
}

/** A new chat, in the project open in the pane when there is one (5.3). */
function newChatRequest()
{
    const project = barProject( state );

    return project === null ? { type: 'conversation.new' } : { type: 'conversation.new', project };
}

/**
 * A new conversation is the bar's; from a chat the person goes back to begin it, to its project when it is in one, so
 * the new chat is begun there too.
 */
function newConversation()
{
    const fromProject = state.place === 'chat' && state.session !== null ? state.session.project ?? null : null;

    if ( fromProject !== null && state.projects.some( ( project ) => project.id === fromProject ) )
        dispatch( { type: 'place.open', place: 'project', project: fromProject } );

    send( newChatRequest() );

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

        send( newChatRequest() );
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
    {
        folderChosen( message.path );

        return;
    }

    if ( message.type === 'licence.text' )
    {
        about.received( message );

        return;
    }

    dispatch( message );

    if ( message.type === 'conversation.loaded' && afterOpen !== null && afterOpen.id === message.id )
    {
        const text = afterOpen.text;

        afterOpen = null;
        sendTurn( text, [] );
    }
}

function decideTool( turnId, callId, decision, folder )
{
    const stream = streamIdentifier( state );
    const message = { type: 'turn.tool_decision', stream_id: stream, turn_id: turnId, call_id: callId, decision };

    if ( folder !== undefined )
        message.folder = folder;

    send( message );
}

/**
 * Mila asked for a folder (AssistantInterface.md 5.5): the window's picker chooses it, so the page never names a path
 * the person did not pick, and the choice is the call's answer.
 */
function chooseFolder( turnId, callId )
{
    folderFor = { turnId, callId };
    transport.send( { type: 'folder.choose' } );
}

/** A folder given to a project (10) is chosen in the window's picker, and the project starts holding it to read. */
function chooseProjectFolder( projectId )
{
    folderFor = { project: projectId };
    transport.send( { type: 'folder.choose' } );
}

/** A picker closed without a choice leaves the question waiting, and the project as it was. */
function folderChosen( path )
{
    const asking = folderFor;
    folderFor = null;

    if ( asking === null || typeof path !== 'string' || path.length === 0 )
        return;

    if ( asking.project !== undefined )
    {
        const project = state.projects.find( ( candidate ) => candidate.id === asking.project );

        if ( project === undefined )
            return;

        const folders = Array.isArray( project.folders ) ? project.folders : [];

        if ( !folders.some( ( folder ) => folder.path === path ) )
            send( { type: 'project.update', id: project.id, folders: [ ...folders, { path, access: 'read' } ] } );

        return;
    }

    decideTool( asking.turnId, asking.callId, 'allow_once', path );
}

/** A link opens once the person has seen where it goes; the window opens it and the chat keeps that it did. */
function linkConfirmed( url )
{
    transport.send( { type: 'link.open', url } );
    transport.send( { type: 'conversation.link_opened', url } );
}

/**
 * What was made, in its drawer (AssistantInterface.md 4.9): a file named by the chat that wrote it and its path, or a
 * reply kept, by its chat and its place among the chat's messages.
 */
function openCreation( conversationId, made )
{
    const request = { type: 'creation.open', conversation_id: conversationId };

    if ( typeof made.path === 'string' )
        request.path = made.path;
    else
        request.message = made.message;

    send( request );
    changeSurface( { sidebarOverlay: false } );
}

/** A reply kept as made, or let go; it is kept with its chat, and no file is written anywhere (8). */
function keepReply( message, keep )
{
    if ( state.conversationId === null || !Number.isInteger( message ) )
        return;

    send( { type: 'creation.keep', conversation_id: state.conversationId, message, keep } );
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
    openProject,
    openConversation,
    newConversation,
    newProject: () => newProjectSheet.open(),
    chooseProjectFolder,
    sendTurn,
    stopTurn,
    openSettings,
    closeSettings,
    openSearch: () => search.open(),
    openLicense: () => license.open(),
    openAbout: () => about.open(),
    decideTool,
    chooseFolder,
    linkConfirmed,
    openCreation,
    keepReply,
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
const chatHead = createChatHeadView( document.getElementById( 'chat-head' ), actions );
const transcript = createTranscriptView( transcriptScroller, document.getElementById( 'transcript' ), actions );
const dock = createDockView( document.getElementById( 'conversation-head' ), actions );
const composer = createComposerView( document.getElementById( 'composer-area' ), mainElement, actions );
const settings = createSettingsView( document.getElementById( 'settings' ), actions );
const search = createSearchView( document.getElementById( 'search' ), actions );
const license = createLicenseView( document.getElementById( 'license-window' ), actions );
const about = createAboutView( document.getElementById( 'about-window' ), actions );
const linkSheet = createLinkView( document.getElementById( 'link-window' ), actions );
const newProjectSheet = createNewProjectView( document.getElementById( 'project-window' ), actions );
const development = createDevelopmentView( sidebar.development, transport.development, composer );
const log = createLogView( logElement, actions );
const plan = createPlanView( planElement );
const made = createMadeView( document.getElementById( 'made-drawer' ), actions );

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
    made.render( state );

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

    // Search, the licence, About, a link's sheet, New project and the Made drawer are dialogs, which Esc closes itself.
    if ( search.isOpen() || license.isOpen() || about.isOpen() || linkSheet.isOpen() || newProjectSheet.isOpen() || made.isOpen() )
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
 * A link in model text never navigates the page: its sheet shows where it goes, and the window opens it in the system
 * browser once the person says so.
 */
function onLinkActivate( event )
{
    const link = event.target.closest( '[data-url]' );

    if ( link === null )
        return;

    if ( event.type === 'keydown' && event.key !== 'Enter' )
        return;

    event.preventDefault();
    linkSheet.open( link.dataset.url );
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
