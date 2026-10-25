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
import { createModelsView } from './views/models.js';
import { createDevelopmentView } from './views/development.js';
import { createLogView } from './views/log.js';

const narrowQuery = window.matchMedia( '(max-width: 899.98px)' );

const transport = await createTransport( window.location );

let state = createInitialState();
let renderPending = false;

const surface = {
    narrow: narrowQuery.matches,
    sidebarCollapsed: false,
    sidebarOverlay: false,
    settingsOpen: false,
    modelsOpen: false,
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
    const message = { type: 'conversation.new' };

    const closed = { sidebarOverlay: false, modelsOpen: false };

    send( message );
    changeSurface( closed );
    composer.focus();
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
const modelsElement = document.getElementById( 'models' );
const developmentElement = document.getElementById( 'development' );
const logElement = document.getElementById( 'log-drawer' );

const sidebar = createSidebarView( sidebarElement, actions );
const header = createHeaderView( headerElement, actions );
const transcript = createTranscriptView( transcriptScroller, transcriptElement, actions );
const composer = createComposerView( composerElement, mainElement, actions );
const settings = createSettingsView( settingsElement, actions );
const models = createModelsView( modelsElement, actions );
const development = createDevelopmentView( developmentElement, transport.development, composer );
const log = createLogView( logElement, actions );

function render()
{
    renderPending = false;

    const sidebarHidden = surface.narrow ? !surface.sidebarOverlay : surface.sidebarCollapsed;
    application.classList.toggle( 'sidebar-hidden', sidebarHidden );
    application.classList.toggle( 'sidebar-overlay', surface.narrow && surface.sidebarOverlay );
    const busy = isBusy( state );
    application.classList.toggle( 'busy', busy );

    sidebar.render( state, surface );
    header.render( state, surface );
    transcript.render( state, surface );
    composer.render( state, surface );
    settings.render( state, surface );
    models.render( state, surface );
    log.render( state, surface );
}

function onNarrowChange( event )
{
    const change = { narrow: event.matches, sidebarOverlay: false };

    changeSurface( change );
}

function onKeyDown( event )
{
    if ( event.ctrlKey && !event.shiftKey && !event.altKey && event.key.toLowerCase() === 'n' )
    {
        event.preventDefault();
        newConversation();

        return;
    }

    if ( event.key !== 'Escape' )
        return;

    if ( isBusy( state ) )
    {
        const stop = { type: 'turn.stop' };
        send( stop );

        return;
    }

    const closed = { sidebarOverlay: false, settingsOpen: false, modelsOpen: false };
    changeSurface( closed );
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
scrim.addEventListener( 'click', closeOverlay );
narrowQuery.addEventListener( 'change', onNarrowChange );
document.addEventListener( 'keydown', onKeyDown );
document.addEventListener( 'click', onLinkActivate );
document.addEventListener( 'keydown', onLinkActivate );
transport.onMessage( dispatch );

development.render();
render();

const ready = { type: 'ready' };
send( ready );
