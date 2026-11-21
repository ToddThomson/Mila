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
        // The workspace's projects and events (AssistantInterface.md phase 4), and the project open in the pane.
        projects: [],
        events: [],
        eventsBefore: null,
        eventsPaging: false,
        projectId: null,
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
        // Where the person is (AssistantInterface.md 4): the workspace, all projects, a project, all chats, activity,
        // what Mila can reach, or the open chat.
        place: 'workspace',
        // The session's conversation was begun from the Mila bar, so the bar goes on with it away from the chat.
        barOwned: true,
        // The conversation risen over the pane above the bar (5.1).
        risen: false,
        about: null,
        // The *Made* drawer (4.9): what was asked for, the session's answer, and why there is none.
        creation: null,
        // A chat Mila opened by asking (5.2), which opens once the reply that opened it ends.
        wentToChat: null,
        // The project the bar's conversation asked Mila to open (5.2), on whose page the bar goes on with it.
        barWentTo: null,
        // The sidebar's *New project* asked Mila for one (4.1), so the bar says what to tell it until something is sent.
        askingForProject: false
    };
}

export const places = [ 'workspace', 'projects', 'project', 'chats', 'activity', 'reach', 'chat' ];

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
            return syncOffers( { ...state, conversations: message.items } );

        case 'projects.list':
            return reduceProjectsList( state, message );

        case 'activity.list':
            return reduceActivityList( state, message );

        // A project the person just made opens.
        case 'project.made':
            return { ...state, place: 'project', projectId: message.id, risen: false };

        case 'conversation.loaded':
            return reduceConversationLoaded( state, message );

        case 'creation.loaded':
            return state.creation === null ? state : { ...state, creation: { ...state.creation, loaded: message, problem: null } };

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

        case 'turn.went':
            return reduceTurnWent( state, message );

        case 'turn.offer':
            return updateTurn( state, message.turn_id, ( turn ) => updateToolBlock( turn, message.call_id, { offer: message.offer } ), message );

        case 'turn.answered':
            return reduceTurnAnswered( state, message );

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
            return { ...reduceTurnSend( state, message ), askingForProject: false };

        case 'turn.tool_decision':
            return updateTurn( state, message.turn_id, reduceToolDecision, message );

        // Every new conversation is the bar's: it goes on there wherever the person goes, until another is opened.
        case 'conversation.new':
            return { ...state, conversationId: null, transcript: [], attachments: [], barOwned: true, risen: false, barWentTo: null,
                askingForProject: false };

        case 'conversation.open':
            return { ...state, place: 'chat', barOwned: false, risen: false, barWentTo: null, askingForProject: false };

        // The page's alone, never sent: the sidebar's *New project* asked Mila for one, in the new chat it began.
        case 'project.ask':
            return { ...state, askingForProject: true };

        case 'conversation.delete':
            return reduceConversationDelete( state, message );

        case 'project.delete':
            return state.projectId === message.id ? { ...state, place: state.place === 'project' ? 'projects' : state.place, projectId: null } : state;

        case 'activity.open':
            return { ...state, eventsPaging: message.before !== null && message.before !== undefined };

        case 'creation.open':
            return { ...state, creation: { request: message, loaded: null, problem: null } };

        // The page's alone: the drawer closed.
        case 'creation.close':
            return { ...state, creation: null };

        // The page's alone: the chat Mila opened has been asked for, now that the reply that opened it has ended.
        case 'went.chat_opened':
            return { ...state, wentToChat: null };

        // The page's alone, never sent: where the person goes, and the conversation risen or lowered over it.
        case 'place.open':
        {
            const opened = reducePlaceOpen( state, message );

            return opened.askingForProject ? { ...opened, askingForProject: false } : opened;
        }

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

    // A drawer still waiting has its answer: what it asked for is no longer there.
    if ( state.creation !== null && state.creation.loaded === null && state.creation.problem === null )
        return { ...state, creation: { ...state.creation, problem: message.text } };

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

    // A project opens by its id, and only while it is stored.
    if ( message.place === 'project' && !state.projects.some( ( project ) => project.id === message.project ) )
        return state;

    const leavingReply = state.place === 'chat' && isBusy( state ) && state.transcript.length > 0;
    const projectId = message.place === 'project' ? message.project : state.projectId;

    return { ...state, place: message.place, projectId, risen: state.risen || leavingReply };
}

/** A project that is no longer stored is not shown: its page goes back to all projects. */
function reduceProjectsList( state, message )
{
    const items = Array.isArray( message.items ) ? message.items : [];
    const stillThere = state.projectId !== null && items.some( ( project ) => project.id === state.projectId );

    if ( stillThere || state.projectId === null )
        return { ...state, projects: items };

    return { ...state, projects: items, projectId: null, place: state.place === 'project' ? 'projects' : state.place };
}

/** The latest events replace what was shown; a page asked for further back is added after them. */
function reduceActivityList( state, message )
{
    const items = Array.isArray( message.items ) ? message.items : [];
    const events = state.eventsPaging ? [ ...state.events, ...items ] : items;

    return { ...state, events, eventsBefore: message.before ?? null, eventsPaging: false };
}

/** The open chat deleted, the session starts a new conversation, which is the bar's; the pane goes to all chats. */
function reduceConversationDelete( state, message )
{
    if ( message.id !== state.conversationId )
        return state;

    const place = state.place === 'chat' ? 'chats' : state.place;

    return { ...state, place, conversationId: null, transcript: [], attachments: [], barOwned: true, risen: false, barWentTo: null };
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
        hasRead: null,
        notInYourWords: null,
        decision: null,
        notRun: false,
        went: null,
        offer: null
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
    const change = { choices: message.choices, reaches: message.reaches ?? null, changesFolder: message.changes_folder ?? null,
        hasRead: message.has_read ?? null, notInYourWords: message.not_in_your_words ?? null,
        projectFolders: Array.isArray( message.folders ) ? message.folders : [],
        places: Array.isArray( message.places ) ? message.places : [] };

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

/**
 * Mila took the pane somewhere (AssistantInterface.md 5.2): the call keeps the line saying where, and the pane goes
 * there, as a click would take it. Asked from the open chat, the chat rises over the place while its reply goes on. A
 * chat other than the one replying opens once the reply ends, since the session holds one conversation at a time.
 */
function reduceTurnWent( state, message )
{
    const went = { place: message.place, text: message.text };
    const marked = updateTurn( state, message.turn_id, ( turn ) => updateToolBlock( turn, message.call_id, { went } ), message );
    const place = message.place ?? {};

    if ( place.view === 'chat' )
        return place.id === marked.conversationId ? marked : { ...marked, wentToChat: place.id };

    if ( place.view === 'project' )
        return reducePlaceOpen( { ...marked, barWentTo: place.id }, { type: 'place.open', place: 'project', project: place.id } );

    if ( !places.includes( place.view ) )
        return marked;

    return reducePlaceOpen( marked, { type: 'place.open', place: place.view } );
}

/**
 * The person's words answered Mila's offer (AssistantInterface.md 5.4): the turn is the app's line, written by no model,
 * and the call that made the offer says how it was answered.
 */
function reduceTurnAnswered( state, message )
{
    const line = ( turn ) => ( { ...turn, app: true, blocks: [ ...turn.blocks, { kind: 'app', id: turn.id + ':app', text: message.text } ] } );
    const answered = updateTurn( state, message.turn_id, line, message );

    return offerInTranscript( answered, message.call_id, message.offer );
}

/** The offer kept on a call shown in the transcript, changed; the call ids of one chat are its own. */
function offerInTranscript( state, callId, offer )
{
    let changed = false;

    const transcript = state.transcript.map( ( entry ) =>
    {
        const block = entry.kind === 'assistant'
            ? entry.blocks.find( ( candidate ) => candidate.kind === 'tool' && candidate.callId === callId ) : undefined;

        if ( block === undefined || JSON.stringify( block.offer ?? null ) === JSON.stringify( offer ) )
            return entry;

        changed = true;

        return updateToolBlock( entry, callId, { offer } );
    } );

    return changed ? { ...state, transcript } : state;
}

/**
 * An offer answered elsewhere, on the overview's card or in a chat's window, reaches the open chat's transcript from
 * the list, where the session keeps every chat's offers.
 */
function syncOffers( state )
{
    const chat = currentChat( state );

    if ( chat === null || !Array.isArray( chat.offers ) )
        return state;

    let synced = state;

    for ( const offer of chat.offers )
    {
        const { message, call_id: callId, ...kept } = offer;

        synced = offerInTranscript( synced, callId, kept );
    }

    return synced;
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

    // The reply is stored as this message of the conversation, which is how it is kept (4.9).
    const stored = Number.isInteger( message.message ) ? message.message : null;

    return { ...turn, blocks, streaming: false, finish: message.finish, answer: message.answer, message: stored };
}

function reduceConversationLoaded( state, message )
{
    let entries = [];
    let number = state.nextEntryNumber;

    for ( const [ position, stored ] of message.messages.entries() )
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

        const assistant = { ...storedAssistantTurn( identifier, stored ), message: position };
        entries = [ ...entries, assistant ];
    }

    return { ...state, conversationId: message.id, transcript: entries, attachments: [], nextEntryNumber: number };
}

function storedAssistantTurn( identifier, stored )
{
    // The app's line where the person's words answered an offer (5.4): no model wrote it.
    if ( stored.app === true )
    {
        return { kind: 'assistant', id: identifier, app: true, blocks: [ { kind: 'app', id: identifier + ':app', text: stored.content } ],
            rounds: [], summary: null, thoughtBreak: false, streaming: false, finish: 'stop', answer: '' };
    }

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
        notRun: call.result === null,
        went: call.went ?? null,
        offer: call.offer ?? null
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

function filesOf( item )
{
    return Array.isArray( item.wrote ) ? item.wrote : [];
}

function keptOf( item )
{
    return Array.isArray( item.kept ) ? item.kept : [];
}

/**
 * What a chat made (4.9, 8): each file it wrote, and each reply the person kept, the most recent first. A file is
 * named by its chat and its path, since two chats may write the same name in different folders.
 */
export function madeIn( item )
{
    const files = filesOf( item ).map( ( file ) => ( { kind: 'file', path: file.path, tool: file.tool, time: file.time ?? null,
        disk: file.disk ?? null, item } ) );
    const replies = keptOf( item ).map( ( reply ) => ( { kind: 'reply', message: reply.message, time: reply.time ?? null,
        start: reply.start ?? null, item } ) );
    const made = [ ...files, ...replies ];
    const when = ( entry ) => entry.time ?? item.updated ?? '';

    return made.sort( ( left, right ) => ( when( left ) < when( right ) ? 1 : when( left ) > when( right ) ? -1 : 0 ) );
}

/** What a project's chats made, the most recent first. */
export function madeInProject( state, projectId )
{
    const made = projectChats( state, projectId ).flatMap( madeIn );
    const when = ( entry ) => entry.time ?? entry.item.updated ?? '';

    return made.sort( ( left, right ) => ( when( left ) < when( right ) ? 1 : when( left ) > when( right ) ? -1 : 0 ) );
}

/** Whether the open chat's reply, stored as message @p message, is kept. */
export function isKept( state, message )
{
    const chat = currentChat( state );

    return chat !== null && keptOf( chat ).some( ( reply ) => reply.message === message );
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
 * What Mila noticed (4.3): each file it wrote that changed on disk since its last write, newest chat first. It is shown
 * beside what needs the person and waits on nobody, so it turns no project amber.
 */
export function changedOnDisk( state )
{
    const rows = [];

    for ( const item of chatsNewestFirst( state ) )
    {
        for ( const file of filesOf( item ) )
        {
            if ( file.disk === 'changed' )
                rows.push( { kind: 'changed', item, file } );
        }
    }

    return rows;
}

/**
 * The notes waiting for the person (5.6): each written in a chat that had read from outside, and given to no new chat
 * until accepted on its project's page. They wait on the person, so their project is waiting on them.
 */
export function waitingNotes( state )
{
    const rows = state.projects
        .filter( ( project ) => project.waiting !== null && typeof project.waiting === 'object' )
        .map( ( project ) => ( { kind: 'note', project, note: project.waiting } ) );
    const when = ( row ) => row.note.made ?? '';

    return rows.sort( ( left, right ) => ( when( left ) < when( right ) ? 1 : when( left ) > when( right ) ? -1 : 0 ) );
}

/**
 * The one action beside a project's *Next* (4.5), from the app's own records, never the note's words: answering a plan
 * stopped in one of its chats, or finishing a reply cut short there; null when nothing waits.
 */
export function projectNextAction( state, projectId )
{
    const row = needsYou( state ).find( ( candidate ) => candidate.item.project === projectId );

    if ( row === undefined )
        return null;

    if ( row.kind === 'plan' )
        return { label: 'Answer and continue', action: 'open-conversation', identifier: row.item.id };

    if ( row.item.finish === 'length' )
        return { label: 'Ask Mila to finish', action: 'finish-reply', identifier: row.item.id };

    return { label: 'Open the chat', action: 'open-conversation', identifier: row.item.id };
}

/** A chat's offers, as the session keeps them on its calls. */
function offersOf( item )
{
    return Array.isArray( item.offers ) ? item.offers : [];
}

/**
 * Mila's suggestions of a project (4.3, 6), newest first: each still waiting on the person's answer, and each made
 * from one that can still be undone.
 */
export function suggestions( state )
{
    const rows = [];

    for ( const item of chatsNewestFirst( state ) )
    {
        for ( const offer of offersOf( item ) )
        {
            if ( offer.kind !== 'project' )
                continue;

            if ( offer.state === 'open' || offer.state === 'undecided' )
                rows.push( { kind: 'suggestion', item, offer } );

            const made = offer.state === 'accepted' ? state.projects.find( ( project ) => project.id === offer.project ) : undefined;

            if ( made !== undefined && made.undo === true )
                rows.push( { kind: 'made', item, offer, project: made } );
        }
    }

    const when = ( row ) => row.offer.time ?? '';

    return rows.sort( ( left, right ) => ( when( left ) < when( right ) ? 1 : when( left ) > when( right ) ? -1 : 0 ) );
}

/** The chats Mila sees as belonging together (4.3): those a suggestion still waiting would gather. */
export function chatsTogether( state )
{
    const together = new Set();

    for ( const row of suggestions( state ) )
    {
        if ( row.kind !== 'suggestion' )
            continue;

        for ( const chat of Array.isArray( row.offer.chats ) ? row.offer.chats : [] )
            together.add( chat.id );
    }

    return together;
}

/**
 * The workspace overview (4.3): how many chats there are and the newest, what is waiting on the person, the most
 * recent projects, and the most recent chats in no project.
 */
export function workspaceOverview( state )
{
    const chats = chatsNewestFirst( state );
    const loose = chats.filter( ( item ) => !inProject( item ) );
    const projects = projectSummaries( state );

    return {
        chatCount: chats.length,
        looseCount: loose.length,
        newest: chats.length > 0 ? chats[ 0 ] : null,
        waiting: needsYou( state ),
        notes: waitingNotes( state ),
        noticed: changedOnDisk( state ),
        suggested: suggestions( state ),
        projects,
        activeCount: projects.filter( ( summary ) => summary.state === 'active' ).length,
        waitingProjects: projects.filter( ( summary ) => summary.state === 'waiting' ),
        recentProjects: projects.slice( 0, overviewProjects ),
        chats: loose.slice( 0, overviewChats )
    };
}

/** The sidebar's recent chats (4.1), in any project or none. */
export function recentChats( state )
{
    return chatsNewestFirst( state ).slice( 0, sidebarChats );
}

// ---- Projects (AssistantInterface.md 4.4, 4.5) --------------------------------------------------

const overviewProjects = 3;
const sidebarProjects = 5;
const pulseDays = 14;

function inProject( item )
{
    return typeof item.project === 'string' && item.project.length > 0;
}

/** A stored project's name as it is now, or null for none or one deleted. */
export function projectName( state, projectId )
{
    if ( typeof projectId !== 'string' )
        return null;

    const project = state.projects.find( ( candidate ) => candidate.id === projectId );

    return project === undefined ? null : project.name;
}

/** The project open in the pane, or null. */
export function currentProject( state )
{
    return state.projects.find( ( project ) => project.id === state.projectId ) ?? null;
}

/** The project a chat is in, or null for one in none or in a project no longer stored. */
export function projectOfChat( state, item )
{
    if ( item === null || !inProject( item ) )
        return null;

    return state.projects.find( ( project ) => project.id === item.project ) ?? null;
}

/** The project the open chat is in, as the session holds it, or null. */
export function openChatProject( state )
{
    const named = state.session === null ? null : state.session.project ?? null;

    return named === null ? null : state.projects.find( ( project ) => project.id === named ) ?? null;
}

/** A project's chats, newest first. */
export function projectChats( state, projectId )
{
    return chatsNewestFirst( state ).filter( ( item ) => item.project === projectId );
}

/** The events that name a project, as where it went or where a chat came from; newest first, as the session sends. */
export function projectEvents( state, projectId )
{
    return state.events.filter( ( event ) => event.project === projectId || event.from === projectId );
}

/**
 * A project's state (4.1), from its chats' records: waiting on the person when one of them needs them (4.3) or a note
 * waits for them (5.6), active while a reply is being written in one, quiet otherwise.
 */
export function projectState( state, projectId )
{
    const chats = projectChats( state, projectId );
    const project = state.projects.find( ( candidate ) => candidate.id === projectId );
    const noteWaits = project !== undefined && project.waiting !== null && typeof project.waiting === 'object';
    const waiting = noteWaits || needsYou( state ).some( ( row ) => row.item.project === projectId );

    if ( waiting )
        return 'waiting';

    const replying = isBusy( state ) && ( state.session.project === projectId
        || chats.some( ( item ) => item.id === state.conversationId ) );

    return replying ? 'active' : 'quiet';
}

/** The count of days, oldest first, ending today: each a count of the project's chats changed and events that day. */
function pulseOf( chats, events, now )
{
    const today = startOfDay( now );
    const day = 24 * 60 * 60 * 1000;
    const counts = new Array( pulseDays ).fill( 0 );
    const count = ( time ) =>
    {
        const index = pulseDays - 1 - Math.round( ( today - startOfDay( new Date( time ) ) ) / day );

        if ( index >= 0 && index < pulseDays )
            counts[ index ] += 1;
    };

    for ( const item of chats )
        count( item.updated );

    for ( const event of events )
        count( event.time );

    return counts;
}

/**
 * Every project with what its pages show of it: its state, its chats, what Mila wrote in them, its plans, when it last
 * changed (itself or one of its chats), its latest event and its last fourteen days. The most recently changed first.
 */
export function projectSummaries( state, now = new Date() )
{
    const summaries = state.projects.map( ( project ) =>
    {
        const chats = projectChats( state, project.id );
        const events = projectEvents( state, project.id );
        const changes = [ project.updated, ...chats.map( ( item ) => item.updated ) ].filter( Boolean );
        const updated = changes.reduce( ( latest, time ) => ( time > latest ? time : latest ), project.created ?? '' );
        const made = chats.reduce( ( total, item ) => total + filesOf( item ).length + keptOf( item ).length, 0 );
        const plans = chats.filter( ( item ) => Boolean( item.progress ) ).length;

        return {
            project,
            state: projectState( state, project.id ),
            chats,
            made,
            plans,
            updated,
            latest: events.length > 0 ? events[ 0 ] : null,
            pulse: pulseOf( chats, events, now )
        };
    } );

    return summaries.sort( ( left, right ) => ( left.updated < right.updated ? 1 : left.updated > right.updated ? -1 : 0 ) );
}

/** The sidebar's projects (4.1): the five most recently changed. */
export function recentProjects( state )
{
    return projectSummaries( state ).slice( 0, sidebarProjects );
}

export const projectFilters = [ 'all', 'waiting', 'active', 'quiet' ];

/** All projects (4.4): with words typed, those whose name or purpose holds each; by filter, and by sort. */
export function matchingProjects( state, query, filter, sort )
{
    const words = wordsOf( query );
    let shown = projectSummaries( state );

    if ( words.length > 0 )
        shown = shown.filter( ( summary ) => holdsEvery( summary.project.name + ' ' + ( summary.project.purpose ?? '' ), words ) );

    if ( filter !== 'all' )
        shown = shown.filter( ( summary ) => summary.state === filter );

    if ( sort === 'name' )
        shown = shown.slice().sort( ( left, right ) => left.project.name.localeCompare( right.project.name ) );

    return shown;
}

export const chatFilters = [ 'all', 'loose', 'placed' ];

/** All chats (4.6) under a filter: every chat, those in no project, or those in one. */
export function filteredChats( state, query, filter )
{
    const shown = matchingChats( state, query );

    if ( filter === 'loose' )
        return shown.filter( ( item ) => !inProject( item ) );

    if ( filter === 'placed' )
        return shown.filter( inProject );

    return shown;
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
 * Search (4.11): the projects whose name holds every word, then the chats whose title does, then the files Mila wrote
 * whose path does, each file once at its latest chat. Nothing typed lists the most recent chats.
 */
export function searchResults( state, query )
{
    const words = wordsOf( query );
    const chats = chatsNewestFirst( state );
    const results = [];
    const seen = new Set();

    // Projects first (4.11), when words are typed.
    if ( words.length > 0 )
    {
        for ( const summary of projectSummaries( state ) )
        {
            if ( holdsEvery( summary.project.name, words ) )
                results.push( { kind: 'project', project: summary.project } );
        }
    }

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
                results.push( { kind: 'file', path: file.path, disk: file.disk ?? null, item } );
            }
        }
    }

    return results.slice( 0, searchLimit );
}

export const activityKinds = [ 'all', 'project', 'chat', 'file', 'plan', 'sent' ];

/**
 * What Mila can reach (4.12): each project's folders, a source of their own at the project's access (10); each folder
 * a chat holds of its own, the open chat's first with the access it has now, since a chat opened again starts
 * read-only; and what the person allowed for the open chat. A chat holding its project's folder is listed under the
 * project, which is where the folder is removed.
 */
export function reachOverview( state )
{
    const session = state.session;
    const openFolder = session === null ? null : session.folder ?? null;
    const openFromProject = session !== null && session.folder_from_project === true;
    const folders = [];
    const projectFolders = [];

    for ( const project of state.projects )
    {
        for ( const folder of Array.isArray( project.folders ) ? project.folders : [] )
            projectFolders.push( { project, path: folder.path, changes: folder.access === 'read_write' } );
    }

    if ( openFolder !== null && !openFromProject )
    {
        const chat = currentChat( state );

        folders.push( { id: state.conversationId, path: openFolder, title: chat === null ? 'New chat' : chat.title,
            changes: session.folder_access === 'read_write' } );
    }

    for ( const item of state.conversations )
    {
        if ( item.folder && item.id !== state.conversationId && item.folder_from_project !== true )
            folders.push( { id: item.id, path: item.folder, title: item.title, changes: false } );
    }

    return {
        projectFolders,
        folders,
        allowed: session === null ? [] : session.allowed_for_chat ?? [],
        webSearch: session !== null && session.web_search === true,
        commands: session !== null && session.run_commands === true
    };
}

/**
 * Activity (4.8), from what the store says of each chat: when it was started, what left the computer from it, each
 * file Mila wrote at its last write and each reply the person kept, and how far its plan came as of the chat's last
 * change, since the store keeps no time for that. Newest first, by the viewer's local day; `kind` keeps one kind, or
 * `'all'`.
 */
export function activityDays( state, kind )
{
    const events = [];

    for ( const item of state.conversations )
    {
        const started = item.created ? item.created : item.updated;

        events.push( { kind: 'chat', who: 'you', time: started, item } );

        // A file written before calls kept a time is shown with the others like it, at the chat's last change.
        const untimed = filesOf( item ).filter( ( file ) => typeof file.time !== 'string' );

        if ( untimed.length > 0 )
            events.push( { kind: 'file', who: 'mila', time: item.updated, item, files: untimed } );

        for ( const file of filesOf( item ) )
        {
            if ( typeof file.time === 'string' )
                events.push( { kind: 'file', who: 'mila', time: file.time, item, files: [ file ], file } );
        }

        for ( const reply of keptOf( item ) )
            events.push( { kind: 'file', who: 'you', time: reply.time ?? item.updated, item, reply } );

        if ( item.progress )
            events.push( { kind: 'plan', who: 'mila', time: item.updated, item } );

        for ( const sent of Array.isArray( item.sent ) ? item.sent : [] )
            events.push( { kind: 'sent', who: sent.kind === 'link' ? 'you' : 'mila', time: sent.time ?? item.updated, item, sent } );
    }

    // The session's own events (8): projects made, changed and deleted, and chats moved between them.
    for ( const recorded of state.events )
    {
        const item = typeof recorded.conversation === 'string'
            ? state.conversations.find( ( chat ) => chat.id === recorded.conversation ) ?? null : null;

        events.push( { kind: 'project', who: recorded.who === 'mila' ? 'mila' : 'you', time: recorded.time, item, recorded,
            project: recorded.project ?? null, to: projectName( state, recorded.project ), from: projectName( state, recorded.from ) } );
    }

    const kept = kind === 'all' ? events : events.filter( ( event ) => event.kind === kind );
    const order = { sent: 0, project: 1, plan: 2, file: 3, chat: 4 };

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
 * conversation begun from the bar, or begin a new chat when the session holds one opened from a list. On a project's
 * page the bar's conversation is the project's: one begun elsewhere is not continued there, unless it asked Mila to
 * open that project (5.2), when the person is still talking to it.
 */
export function barMode( state )
{
    if ( state.place === 'chat' )
        return 'reply';

    if ( !state.barOwned )
        return 'start';

    const sessionProject = state.session === null ? null : state.session.project ?? null;
    const here = state.place === 'project' ? state.projectId : null;

    // A conversation not yet begun takes the place it is begun in.
    if ( state.transcript.length === 0 )
        return sessionProject === here ? 'continue' : 'start';

    return here === null || sessionProject === here || state.barWentTo === here ? 'continue' : 'start';
}

/** The project a chat begun from the bar here belongs to: the project open in the pane, or none. */
export function barProject( state )
{
    return state.place === 'project' ? state.projectId : null;
}

/** Whether the conversation shows risen over the place: raised, and with something in it. */
export function isRisen( state )
{
    return state.risen && state.place !== 'chat' && ( state.transcript.length > 0 || isBusy( state ) );
}
