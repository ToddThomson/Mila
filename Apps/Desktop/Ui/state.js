// The page's state, reduced from the messages of MilaDesktop.md section 4 in both directions.
// No DOM access, so the page's behaviour is testable under node --test. Every change returns new
// objects along the changed path and keeps the rest, so the view can compare by identity.

const logLimit = 2000;

export function createInitialState()
{
    return {
        connected: false,
        session: null,
        conversations: [],
        conversationId: null,
        transcript: [],
        installedModels: [],
        modelsKnown: false,
        publishedModels: null,
        suggestedModel: null,
        publishedRequested: false,
        license: null,
        licenseError: null,
        licenseRead: false,
        setupStep: 'welcome',
        setupInstalling: false,
        progress: null,
        restore: null,
        modelsError: null,
        attachments: [],
        logs: [],
        nextEntryNumber: 1,
        // Where the person is (AssistantInterface.md 4): the workspace, all chats, activity, or the open chat.
        place: 'workspace',
        // The session's conversation was begun from the Mila bar, so the bar goes on with it away from the chat.
        barOwned: true,
        // The conversation risen over the pane above the bar (5.1).
        risen: false,
        about: null
    };
}

export const places = [ 'workspace', 'chats', 'activity', 'reach', 'chat' ];

export function reduce( state, message )
{
    switch ( message.type )
    {
        case 'session.state':
            return reduceSessionState( state, message );

        case 'models.list':
            return {
                ...state,
                installedModels: message.installed,
                modelsKnown: true,
                publishedModels: message.published,
                suggestedModel: message.suggested ?? null,
                publishedRequested: false
            };

        case 'progress':
            // The model's memory coming back is the transcript's to show, not Settings'.
            if ( message.task === 'restore' )
                return { ...state, restore: message.done < message.total ? message : null };

            return { ...state, progress: message };

        case 'conversations.list':
            return { ...state, conversations: message.items };

        case 'conversation.loaded':
            return reduceConversationLoaded( state, message );

        case 'turn.begin':
            return reduceTurnBegin( state, message );

        case 'turn.text':
            return updateTurn( state, message.turn_id, reduceTurnText, message );

        case 'turn.thought_break':
            return updateTurn( state, message.turn_id, reduceThoughtBreak, message );

        case 'turn.tool_call':
            return updateTurn( state, message.turn_id, reduceToolCall, message );

        case 'turn.tool_approval':
            return updateTurn( state, message.turn_id, reduceToolApproval, message );

        case 'turn.tool_result':
            return updateTurn( state, message.turn_id, reduceToolResult, message );

        case 'turn.round_end':
            return updateTurn( state, message.turn_id, reduceRoundEnd, message );

        case 'turn.end':
            return updateTurn( state, message.turn_id, reduceTurnEnd, message );

        case 'attachment.accepted':
            return reduceAttachmentAccepted( state, message );

        case 'attachment.refused':
            return reduceAttachmentRefused( state, message );

        case 'info':
            return reduceInfo( state, message );

        case 'error':
            return reduceError( state, message );

        case 'log':
            return reduceLog( state, message );

        // The page's own requests: those that change what the page shows before any reply.

        case 'ready':
            return { ...state, connected: true };

        case 'turn.send':
            return reduceTurnSend( state, message );

        case 'turn.tool_decision':
            return updateTurn( state, message.turn_id, reduceToolDecision, message );

        // Every new conversation is the bar's: it goes on there wherever the person goes, until another is opened.
        case 'conversation.new':
            return { ...state, conversationId: null, transcript: [], attachments: [], barOwned: true, risen: false };

        case 'conversation.open':
            return { ...state, place: 'chat', barOwned: false, risen: false };

        case 'conversation.delete':
            return reduceConversationDelete( state, message );

        // The page's alone, never sent: where the person goes, and the conversation risen or lowered over it.
        case 'place.open':
            return reducePlaceOpen( state, message );

        case 'conversation.raise':
            return { ...state, risen: message.risen && state.place !== 'chat' };

        // A chat begun from the bar takes the attachments waiting in it, which the new conversation would clear.
        case 'attachments.keep':
            return { ...state, attachments: message.attachments };

        // From the window, not the session: who is signed in to this computer, the app's version, and the
        // third-party licences it carries, each by an id the window gave and its name.
        case 'about':
            return { ...state, about: reduceAbout( message ) };

        case 'attachment.add':
            return reduceAttachmentAdd( state, message );

        case 'attachment.remove':
            return reduceAttachmentRemove( state, message );

        case 'models.refresh':
            return reduceModelsRefresh( state, message );

        case 'license':
            return reduceLicense( state, message );

        // Accepting moves on to the model at once; a refusal keeps the licence's pane, which then says why.
        case 'license.accept':
            return { ...state, licenseError: null, setupStep: 'model' };

        // The page's alone, never sent: moving through the first-run setup (AssistantOnboarding.md 4.1).
        case 'license.read':
            return { ...state, licenseRead: true };

        case 'setup.continue':
            return reduceSetupContinue( state );

        case 'setup.back':
            return reduceSetupBack( state );

        case 'setup.done':
            return { ...state, setupStep: 'done', place: 'workspace' };

        // An install from the setup's model pane ends on its conclusion once the model is loaded.
        case 'model.install':
            return { ...state, modelsError: null, setupInstalling: state.setupInstalling || setupPane( state ) === 'model' };

        case 'model.load':
        case 'model.remove':
            return { ...state, modelsError: null };

        default:
            return state;
    }
}

/**
 * A different model means the session cleared its history, as /model load does, so the
 * transcript and any attachment not yet sent go with it.
 */
function reduceSessionState( state, message )
{
    const stream = message.streams.length > 0 ? message.streams[ 0 ] : null;
    const conversationId = stream === null ? null : stream.conversation_id;
    const modelChanged = state.session !== null && state.session.model !== message.model;
    const transcript = modelChanged ? [] : state.transcript;
    const attachments = modelChanged ? [] : state.attachments;

    return { ...state, session: message, conversationId, transcript, attachments };
}

function appendEntry( state, entry )
{
    const identifier = 'entry-' + state.nextEntryNumber;
    const identified = { ...entry, id: identifier };

    return { ...state, transcript: [ ...state.transcript, identified ], nextEntryNumber: state.nextEntryNumber + 1 };
}

/** Asking the publisher is under way until its list comes back; an earlier failure to list is asked again. */
function reduceModelsRefresh( state, message )
{
    if ( !message.online )
        return state;

    const modelsError = state.modelsError !== null && state.modelsError.during === 'models' ? null : state.modelsError;

    return { ...state, publishedRequested: true, modelsError };
}

function reduceAbout( message )
{
    const given = Array.isArray( message.licences ) ? message.licences : [];
    const licences = given
        .filter( ( entry ) => entry !== null && typeof entry === 'object' && typeof entry.id === 'string' && typeof entry.name === 'string' )
        .map( ( entry ) => ( { id: entry.id, name: entry.name } ) );

    return { userName: message.user_name ?? '', version: message.version ?? '', licences };
}

/** The licence the session carries and whether it is accepted; a text that fits its box needs no scrolling to be read. */
function reduceLicense( state, message )
{
    const license = { text: message.text, sha256: message.sha256, accepted: message.accepted, problem: message.problem };
    const changed = state.license !== null && state.license.sha256 !== message.sha256;

    return { ...state, license, licenseRead: changed ? false : state.licenseRead };
}

/** Continue from the welcome: to the licence, or once it is accepted, to the model. */
function reduceSetupContinue( state )
{
    const accepted = state.license !== null && state.license.accepted;

    if ( state.setupStep === 'welcome' )
        return { ...state, setupStep: accepted ? 'model' : 'license' };

    if ( state.setupStep === 'license' && accepted )
        return { ...state, setupStep: 'model' };

    return state;
}

/** Go Back: the model's pane returns to the licence read on this run, or to the welcome. */
function reduceSetupBack( state )
{
    if ( state.setupStep === 'license' )
        return { ...state, setupStep: 'welcome' };

    if ( state.setupStep === 'model' )
        return { ...state, setupStep: state.licenseRead ? 'license' : 'welcome' };

    return state;
}

/**
 * What an install or a load says as it ends is a conversation's line only once one has begun: before, the
 * header already names the model, and the empty conversation keeps its screen.
 */
function reduceInfo( state, message )
{
    const aboutModels = message.during === 'install' || message.during === 'load';

    if ( aboutModels && state.transcript.length === 0 )
        return state;

    const entry = { kind: 'info', text: message.text };

    return appendEntry( state, entry );
}

/**
 * A refused load or install, or a list the publisher did not give, is also kept for the models' screens and
 * Settings' Models, which covers the transcript while it is open, until the person's next load, install or remove.
 */
function reduceError( state, message )
{
    // A refused acceptance is said on the licence's pane, and the person is back on it.
    if ( message.during === 'license' )
        return { ...state, licenseError: message.text, setupStep: 'license' };

    const entry = { kind: 'error', text: message.text, during: message.during };
    const aboutModels = message.during === 'load' || message.during === 'install' || message.during === 'models';

    if ( !aboutModels )
        return appendEntry( state, entry );

    const modelsError = { text: message.text, during: message.during };

    // Before a conversation has begun, a refusal is said where the models are offered, not as its first entry.
    if ( state.transcript.length === 0 )
        return { ...state, modelsError };

    const added = appendEntry( state, entry );

    return { ...added, modelsError };
}

function sentAttachment( attachment )
{
    return { kind: attachment.kind, name: attachment.name, previewUrl: attachment.previewUrl };
}

/** Sent from anywhere but the open chat, the conversation rises over the place the person is in. */
function reduceTurnSend( state, message )
{
    const included = state.attachments.filter( ( attachment ) => message.attachment_ids.includes( attachment.id ) );
    const attachments = included.map( sentAttachment );
    const entry = { kind: 'user', text: message.text, attachments };
    const added = appendEntry( state, entry );

    return { ...added, attachments: [], risen: state.place !== 'chat' };
}

/**
 * A chat opens with the pane to itself, its conversation lowered. Leaving a chat while Mila is still replying, the
 * reply goes on in the conversation risen over the new place, so it is never out of sight while it runs.
 */
function reducePlaceOpen( state, message )
{
    if ( !places.includes( message.place ) )
        return state;

    if ( message.place === 'chat' )
        return { ...state, place: 'chat', risen: false };

    const leavingReply = state.place === 'chat' && isBusy( state ) && state.transcript.length > 0;

    return { ...state, place: message.place, risen: state.risen || leavingReply };
}

/** The open chat deleted, the session starts a new conversation, which is the bar's; the pane goes to all chats. */
function reduceConversationDelete( state, message )
{
    if ( message.id !== state.conversationId )
        return state;

    const place = state.place === 'chat' ? 'chats' : state.place;

    return { ...state, place, conversationId: null, transcript: [], attachments: [], barOwned: true, risen: false };
}

function reduceTurnBegin( state, message )
{
    const turn = {
        kind: 'assistant',
        id: message.turn_id,
        blocks: [],
        rounds: [],
        thoughtBreak: false,
        streaming: true,
        finish: null,
        answer: null
    };

    return { ...state, conversationId: message.conversation_id, transcript: [ ...state.transcript, turn ] };
}

function updateTurn( state, turnId, update, message )
{
    const index = state.transcript.findIndex( ( entry ) => entry.kind === 'assistant' && entry.id === turnId );

    if ( index < 0 )
        return state;

    const transcript = state.transcript.slice();
    transcript[ index ] = update( transcript[ index ], message );

    return { ...state, transcript };
}

function replaceLastBlock( turn, block )
{
    const blocks = turn.blocks.slice();
    blocks[ blocks.length - 1 ] = block;

    return blocks;
}

function lastBlock( turn )
{
    return turn.blocks.length > 0 ? turn.blocks[ turn.blocks.length - 1 ] : null;
}

function reduceTurnText( turn, message )
{
    const last = lastBlock( turn );
    const kind = message.channel === 'thought' ? 'thought' : 'answer';
    const continues = last !== null && last.kind === kind && !( kind === 'thought' && turn.thoughtBreak );

    // The round's tokens so far, sent with a thought so its label moves while it is written.
    const liveTokens = message.tokens === undefined ? null : message.tokens;

    if ( continues )
    {
        const extended = { ...last, text: last.text + message.text, liveTokens };

        return { ...turn, blocks: replaceLastBlock( turn, extended ) };
    }

    const block = {
        kind,
        id: turn.id + ':' + turn.blocks.length,
        round: message.round,
        text: message.text,
        tokens: null,
        liveTokens
    };

    return { ...turn, blocks: [ ...turn.blocks, block ], thoughtBreak: false };
}

function reduceThoughtBreak( turn )
{
    return { ...turn, thoughtBreak: true };
}

function reduceToolCall( turn, message )
{
    const block = {
        kind: 'tool',
        id: turn.id + ':' + turn.blocks.length,
        turnId: turn.id,
        callId: message.call_id,
        round: message.round,
        name: message.name,
        summary: message.summary,
        arguments: message.arguments,
        result: null,
        isError: false,
        choices: null,
        reaches: null,
        changesFolder: null,
        decision: null,
        notRun: false
    };

    return { ...turn, blocks: [ ...turn.blocks, block ] };
}

function updateToolBlock( turn, callId, change )
{
    const index = turn.blocks.findIndex( ( block ) => block.kind === 'tool' && block.callId === callId );

    if ( index < 0 )
        return turn;

    const blocks = turn.blocks.slice();
    blocks[ index ] = { ...blocks[ index ], ...change };

    return { ...turn, blocks };
}

function reduceToolApproval( turn, message )
{
    const change = { choices: message.choices, reaches: message.reaches ?? null, changesFolder: message.changes_folder ?? null };

    return updateToolBlock( turn, message.call_id, change );
}

/**
 * The page's own answer: the choices go at once, so a second click has nothing to answer.
 */
function reduceToolDecision( turn, message )
{
    const change = { choices: null, decision: message.decision };

    return updateToolBlock( turn, message.call_id, change );
}

function reduceToolResult( turn, message )
{
    const change = { result: message.result, isError: message.is_error, choices: null };

    return updateToolBlock( turn, message.call_id, change );
}

/**
 * A call with no result when the reply ends was never run: the reply was stopped before it, or
 * while it waited for the person.
 */
function markNotRun( block )
{
    if ( block.kind !== 'tool' || block.result !== null )
        return block;

    return { ...block, choices: null, notRun: true };
}

function reduceRoundEnd( turn, message )
{
    const round = {
        round: message.round,
        prefillMs: message.prefill_ms,
        tokensPerSecond: message.tokens_per_second,
        tokens: message.tokens,
        thoughtTokens: message.thought_tokens,
        finish: message.finish
    };

    const blocks = turn.blocks.map( ( block ) => labelThought( block, message ) );

    return { ...turn, blocks, rounds: [ ...turn.rounds, round ] };
}

function labelThought( block, message )
{
    if ( block.kind !== 'thought' || block.round !== message.round || block.tokens !== null )
        return block;

    return { ...block, tokens: message.thought_tokens };
}

function reduceTurnEnd( turn, message )
{
    const blocks = turn.blocks.map( markNotRun );

    return { ...turn, blocks, streaming: false, finish: message.finish, answer: message.answer };
}

function reduceConversationLoaded( state, message )
{
    let entries = [];
    let number = state.nextEntryNumber;

    for ( const stored of message.messages )
    {
        if ( stored.role === 'tool' )
            continue;

        const identifier = 'entry-' + number;
        number = number + 1;

        if ( stored.role === 'user' )
        {
            const attachments = stored.attachments === null ? [] : stored.attachments;
            const user = { kind: 'user', id: identifier, text: stored.content, attachments };
            entries = [ ...entries, user ];

            continue;
        }

        const assistant = storedAssistantTurn( identifier, stored );
        entries = [ ...entries, assistant ];
    }

    return { ...state, conversationId: message.id, transcript: entries, attachments: [], nextEntryNumber: number };
}

function storedAssistantTurn( identifier, stored )
{
    const blocks = [];

    if ( stored.thought !== null && stored.thought.length > 0 )
    {
        const thought = { kind: 'thought', id: identifier + ':thought', round: 1, text: stored.thought, tokens: null };
        blocks.push( thought );
    }

    const toolCalls = stored.tool_calls === null ? [] : stored.tool_calls;

    for ( const call of toolCalls )
    {
        const tool = storedToolBlock( identifier, call );
        blocks.push( tool );
    }

    const answer = { kind: 'answer', id: identifier + ':answer', round: 1, text: stored.content, tokens: null };
    blocks.push( answer );

    const summary = stored.stats === null ? null : storedSummary( stored.stats );
    const finish = stored.stats === null ? 'stop' : stored.stats.finish;

    return {
        kind: 'assistant',
        id: identifier,
        blocks,
        rounds: [],
        summary,
        thoughtBreak: false,
        streaming: false,
        finish,
        answer: stored.content
    };
}

function storedToolBlock( identifier, call )
{
    return {
        kind: 'tool',
        id: identifier + ':' + call.call_id,
        turnId: identifier,
        callId: call.call_id,
        round: 1,
        name: call.name,
        summary: call.summary,
        arguments: call.arguments,
        result: call.result,
        isError: call.is_error,
        choices: null,
        decision: call.decision,
        notRun: call.result === null
    };
}

function storedSummary( stats )
{
    return {
        prefillMs: stats.prefill_ms,
        tokensPerSecond: stats.tokens_per_second,
        tokens: stats.tokens,
        rounds: stats.rounds
    };
}

function reduceAttachmentAdd( state, message )
{
    const attachment = {
        id: message.attachment_id,
        kind: message.kind,
        name: message.name,
        previewUrl: message.previewUrl === undefined ? null : message.previewUrl,
        status: 'pending',
        sizeBytes: null,
        reason: null
    };

    return { ...state, attachments: [ ...state.attachments, attachment ] };
}

function reduceAttachmentAccepted( state, message )
{
    const change = { status: 'accepted', sizeBytes: message.size_bytes };

    return updateAttachment( state, message.attachment_id, change );
}

function reduceAttachmentRefused( state, message )
{
    const change = { status: 'refused', reason: message.reason };

    return updateAttachment( state, message.attachment_id, change );
}

function reduceAttachmentRemove( state, message )
{
    const attachments = state.attachments.filter( ( attachment ) => attachment.id !== message.attachment_id );

    return { ...state, attachments };
}

function updateAttachment( state, attachmentId, change )
{
    const attachments = state.attachments.map( ( attachment ) => attachment.id === attachmentId ? { ...attachment, ...change } : attachment );

    return { ...state, attachments };
}

function reduceLog( state, message )
{
    const entry = { level: message.level, text: message.text };
    const logs = [ ...state.logs, entry ];
    const kept = logs.length > logLimit ? logs.slice( logs.length - logLimit ) : logs;

    return { ...state, logs: kept };
}

// ---- Derived values --------------------------------------------------------------------------

/**
 * A reply's figures across its rounds: prefill summed, tokens summed, and the decode rate over
 * all of them -- each round's decode time is its tokens over its rate.
 */
export function summarizeTurn( turn )
{
    if ( turn.summary !== undefined && turn.summary !== null )
        return turn.summary;

    if ( turn.rounds.length === 0 )
        return null;

    let prefillMs = 0;
    let tokens = 0;
    let decodeSeconds = 0;

    for ( const round of turn.rounds )
    {
        prefillMs = prefillMs + round.prefillMs;
        tokens = tokens + round.tokens;

        if ( round.tokensPerSecond > 0 )
            decodeSeconds = decodeSeconds + round.tokens / round.tokensPerSecond;
    }

    const tokensPerSecond = decodeSeconds > 0 ? tokens / decodeSeconds : 0;

    return { prefillMs, tokensPerSecond, tokens, rounds: turn.rounds.length };
}

/**
 * What the app is doing to the model, in a word for the header, or null when it is doing nothing: its memory coming
 * back, then a load, a rebuild or an install under way.
 */
export function appStatus( state )
{
    if ( state.restore !== null )
        return state.restore.label;

    const progress = state.progress;

    if ( progress === null || ( progress.total !== null && progress.done >= progress.total ) )
        return null;

    // The model a load opens is already the picker's name.
    return progress.task === 'load' || progress.task === 'rebuild' ? 'Loading' : progress.label;
}

/**
 * What an empty conversation shows. With no model loaded or loading: the installed models to choose from, or,
 * with none installed, the first run; nothing until the session has said which. A model being loaded
 * is shown as the screen it is about to have, since the header already says it is loading.
 */
export function emptyScreen( state )
{
    const session = state.session;

    if ( session === null )
        return null;

    if ( session.model === null && !session.loading )
    {
        if ( !state.modelsKnown )
            return null;

        return state.installedModels.length > 0 ? 'choose-model' : 'first-run';
    }

    return 'home';
}

/**
 * Where a first run stands (AssistantOnboarding.md 4): the publisher not yet asked, being asked, unreachable,
 * a model suggested for this computer, or none of its models running here.
 */
export function firstRunStep( state )
{
    if ( state.modelsError !== null && state.modelsError.during === 'models' )
        return 'unreachable';

    if ( state.publishedModels === null )
        return state.publishedRequested ? 'asking' : 'not-asked';

    return state.suggestedModel !== null ? 'suggested' : 'none-fits';
}

/** The published model the planner suggests for this computer, or null. */
export function suggestedModel( state )
{
    if ( state.publishedModels === null || state.suggestedModel === null )
        return null;

    return state.publishedModels.find( ( model ) => model.name === state.suggestedModel ) ?? null;
}

/**
 * Whether the page is asking for a model: an empty conversation on the first-run or choose-a-model screen. The
 * screen is then the one way forward, so the header's model menu and the message box are not shown beside it.
 */
export function isAskingForModel( state )
{
    // The setup is the one way forward too, its conclusion included.
    if ( setupPane( state ) !== null )
        return true;

    const screen = emptyScreen( state );

    const asking = screen === 'first-run' || screen === 'choose-model';

    return state.transcript.length === 0 && asking;
}

/**
 * The first-run setup's pane (AssistantOnboarding.md 4.1), or null when it is not showing: until the licence is
 * accepted, its welcome and the licence; then, with nothing installed, the welcome and the model; and once the
 * model it installed is loaded, the conclusion until Done. Nothing until the session has said where the licence
 * stands.
 */
export function setupPane( state )
{
    if ( state.session === null || state.license === undefined || state.license === null )
        return null;

    if ( !state.license.accepted )
        return state.setupStep === 'welcome' ? 'welcome' : 'license';

    if ( state.setupStep === 'done' )
        return null;

    if ( state.setupInstalling )
        return setupInstallPane( state );

    // A line the session said before the setup, a setting it could not read, does not end it; a conversation does.
    const conversed = state.transcript.some( ( entry ) => entry.kind === 'user' || entry.kind === 'assistant' );

    if ( emptyScreen( state ) !== 'first-run' || conversed )
        return null;

    return state.setupStep;
}

/** An install begun from the setup: its pane until the model is loaded, then the conclusion. */
function setupInstallPane( state )
{
    if ( state.session.model !== null )
        return 'conclusion';

    // A model installed and refused at its load is chosen among the installed, with the reason.
    const refused = state.modelsError !== null && state.modelsError.during === 'load';

    return refused && !state.session.loading ? null : 'model';
}

/**
 * Whether the page is in the first-run setup. It then fills the window, with no sidebar or header beside it, as
 * there is nothing yet to go back to or to show.
 */
export function isWelcoming( state )
{
    return setupPane( state ) !== null;
}

export function isBusy( state )
{
    return state.session !== null && state.session.busy;
}

export function streamIdentifier( state )
{
    if ( state.session === null || state.session.streams.length === 0 )
        return null;

    return state.session.streams[ 0 ].stream_id;
}

function startOfDay( date )
{
    const year = date.getFullYear();
    const month = date.getMonth();
    const day = date.getDate();
    const midnight = new Date( year, month, day );

    return midnight.getTime();
}

function compareUpdatedDescending( left, right )
{
    return new Date( right.updated ).getTime() - new Date( left.updated ).getTime();
}

// ---- The workspace (AssistantInterface.md 4) ----------------------------------------------------

const overviewChats = 5;
const sidebarChats = 4;
const searchLimit = 12;

// The endings of a reply the person has to act on; a stop, a call or their own stop is not one.
const attentionEndings = [ 'length', 'context', 'error' ];

/** Every stored chat, the most recently changed first. */
export function chatsNewestFirst( state )
{
    return state.conversations.slice().sort( compareUpdatedDescending );
}

/** The open chat as the list names it, or null for a conversation not yet stored. */
export function currentChat( state )
{
    if ( state.conversationId === null )
        return null;

    return state.conversations.find( ( item ) => item.id === state.conversationId ) ?? null;
}

function planStopped( item )
{
    return Boolean( item.progress ) && item.progress.current !== null;
}

/**
 * What needs the person (4.3): a plan stopped short of its end, and a reply that ended before it finished, newest
 * first. The chat a reply is under way in needs nothing yet.
 */
export function needsYou( state )
{
    const working = isBusy( state ) ? state.conversationId : null;
    const idle = chatsNewestFirst( state ).filter( ( item ) => item.id !== working );
    const rows = [];

    for ( const item of idle )
    {
        if ( planStopped( item ) )
            rows.push( { kind: 'plan', item } );
        else if ( attentionEndings.includes( item.finish ) )
            rows.push( { kind: 'reply', item } );
    }

    return rows;
}

/**
 * The workspace overview (4.3) of a workspace of chats: how many there are and the newest, what is waiting on the
 * person, and the most recent chats. Projects join it at phase 4.
 */
export function workspaceOverview( state )
{
    const chats = chatsNewestFirst( state );

    return {
        chatCount: chats.length,
        newest: chats.length > 0 ? chats[ 0 ] : null,
        waiting: needsYou( state ),
        chats: chats.slice( 0, overviewChats )
    };
}

/** The sidebar's recent chats (4.1). */
export function recentChats( state )
{
    return chatsNewestFirst( state ).slice( 0, sidebarChats );
}

function wordsOf( query )
{
    return query.toLowerCase().split( /\s+/ ).filter( ( word ) => word.length > 0 );
}

function holdsEvery( text, words )
{
    const lower = text.toLowerCase();

    return words.every( ( word ) => lower.includes( word ) );
}

/** All chats (4.6), newest first: with words typed, those whose title or start of Mila's last reply holds each. */
export function matchingChats( state, query )
{
    const words = wordsOf( query );
    const chats = chatsNewestFirst( state );

    if ( words.length === 0 )
        return chats;

    return chats.filter( ( item ) => holdsEvery( item.title + ' ' + ( item.reply_start ?? '' ), words ) );
}

/**
 * Search (4.11): the chats whose title holds every word, then the files Mila wrote whose path does, each file once
 * at its latest chat. Nothing typed lists the most recent chats.
 */
export function searchResults( state, query )
{
    const words = wordsOf( query );
    const chats = chatsNewestFirst( state );
    const results = [];
    const seen = new Set();

    for ( const item of chats )
    {
        if ( words.length === 0 || holdsEvery( item.title, words ) )
            results.push( { kind: 'chat', item } );
    }

    if ( words.length > 0 )
    {
        for ( const item of chats )
        {
            const wrote = Array.isArray( item.wrote ) ? item.wrote.slice().reverse() : [];

            for ( const file of wrote )
            {
                const key = ( item.folder ?? '' ) + '\u0000' + file.path;

                if ( seen.has( key ) || !holdsEvery( file.path, words ) )
                    continue;

                seen.add( key );
                results.push( { kind: 'file', path: file.path, item } );
            }
        }
    }

    return results.slice( 0, searchLimit );
}

export const activityKinds = [ 'all', 'chat', 'file', 'plan', 'sent' ];

/**
 * What Mila can reach (4.12): each folder a chat holds, the open chat's first with the access it has now, since a chat
 * opened again starts read-only; and what the person allowed for the open chat.
 */
export function reachOverview( state )
{
    const session = state.session;
    const openFolder = session === null ? null : session.folder ?? null;
    const folders = [];

    if ( openFolder !== null )
    {
        const chat = currentChat( state );

        folders.push( { id: state.conversationId, path: openFolder, title: chat === null ? 'New chat' : chat.title,
            changes: session.folder_access === 'read_write' } );
    }

    for ( const item of state.conversations )
    {
        if ( item.folder && item.id !== state.conversationId )
            folders.push( { id: item.id, path: item.folder, title: item.title, changes: false } );
    }

    return {
        folders,
        allowed: session === null ? [] : session.allowed_for_chat ?? [],
        webSearch: session !== null && session.web_search === true,
        commands: session !== null && session.run_commands === true
    };
}

/**
 * Activity (4.8), from what the store says of each chat today: when it was started, what left the computer from it,
 * and the files Mila wrote in it and how far its plan came, both as of the chat's last change, since the store keeps no
 * time for either. Newest first, by the viewer's local day; `kind` keeps one kind, or `'all'`.
 */
export function activityDays( state, kind )
{
    const events = [];

    for ( const item of state.conversations )
    {
        const started = item.created ? item.created : item.updated;

        events.push( { kind: 'chat', who: 'you', time: started, item } );

        if ( Array.isArray( item.wrote ) && item.wrote.length > 0 )
            events.push( { kind: 'file', who: 'mila', time: item.updated, item } );

        if ( item.progress )
            events.push( { kind: 'plan', who: 'mila', time: item.updated, item } );

        for ( const sent of Array.isArray( item.sent ) ? item.sent : [] )
            events.push( { kind: 'sent', who: sent.kind === 'link' ? 'you' : 'mila', time: sent.time ?? item.updated, item, sent } );
    }

    const kept = kind === 'all' ? events : events.filter( ( event ) => event.kind === kind );
    const order = { sent: 0, plan: 1, file: 2, chat: 3 };

    kept.sort( ( left, right ) => new Date( right.time ).getTime() - new Date( left.time ).getTime() || order[ left.kind ] - order[ right.kind ] );

    const days = [];

    for ( const event of kept )
    {
        const day = startOfDay( new Date( event.time ) );
        const last = days.length > 0 ? days[ days.length - 1 ] : null;

        if ( last !== null && last.day === day )
            last.events.push( event );
        else
            days.push( { day, events: [ event ] } );
    }

    return days;
}

/**
 * What the Mila bar does with what is typed (5.1, 5.3): reply in the open chat; elsewhere, go on with the
 * conversation begun from the bar, or begin a new chat when the session holds one opened from a list.
 */
export function barMode( state )
{
    if ( state.place === 'chat' )
        return 'reply';

    return state.barOwned ? 'continue' : 'start';
}

/** Whether the conversation shows risen over the place: raised, and with something in it. */
export function isRisen( state )
{
    return state.risen && state.place !== 'chat' && ( state.transcript.length > 0 || isBusy( state ) );
}
