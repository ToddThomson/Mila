// The page's state, reduced from the messages of ChatDesktop.md section 4 in both directions.
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
        publishedModels: null,
        progress: null,
        attachments: [],
        logs: [],
        nextEntryNumber: 1
    };
}

export function reduce( state, message )
{
    switch ( message.type )
    {
        case 'session.state':
            return reduceSessionState( state, message );

        case 'models.list':
            return { ...state, installedModels: message.installed, publishedModels: message.published };

        case 'progress':
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

        case 'conversation.new':
            return { ...state, conversationId: null, transcript: [], attachments: [] };

        case 'attachment.add':
            return reduceAttachmentAdd( state, message );

        case 'attachment.remove':
            return reduceAttachmentRemove( state, message );

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

function reduceInfo( state, message )
{
    const entry = { kind: 'info', text: message.text };

    return appendEntry( state, entry );
}

function reduceError( state, message )
{
    const entry = { kind: 'error', text: message.text, during: message.during };

    return appendEntry( state, entry );
}

function sentAttachment( attachment )
{
    return { kind: attachment.kind, name: attachment.name, previewUrl: attachment.previewUrl };
}

function reduceTurnSend( state, message )
{
    const included = state.attachments.filter( ( attachment ) => message.attachment_ids.includes( attachment.id ) );
    const attachments = included.map( sentAttachment );
    const entry = { kind: 'user', text: message.text, attachments };
    const added = appendEntry( state, entry );

    return { ...added, attachments: [] };
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

    if ( continues )
    {
        const extended = { ...last, text: last.text + message.text };

        return { ...turn, blocks: replaceLastBlock( turn, extended ) };
    }

    const block = {
        kind,
        id: turn.id + ':' + turn.blocks.length,
        round: message.round,
        text: message.text,
        tokens: null
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
        callId: message.call_id,
        round: message.round,
        name: message.name,
        summary: message.summary,
        arguments: message.arguments,
        result: null,
        isError: false
    };

    return { ...turn, blocks: [ ...turn.blocks, block ] };
}

function reduceToolResult( turn, message )
{
    const index = turn.blocks.findIndex( ( block ) => block.kind === 'tool' && block.callId === message.call_id );

    if ( index < 0 )
        return turn;

    const blocks = turn.blocks.slice();
    blocks[ index ] = { ...blocks[ index ], result: message.result, isError: message.is_error };

    return { ...turn, blocks };
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
    return { ...turn, streaming: false, finish: message.finish, answer: message.answer };
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
        callId: call.call_id,
        round: 1,
        name: call.name,
        summary: call.name,
        arguments: call.arguments,
        result: call.result,
        isError: false
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

const dayMilliseconds = 24 * 60 * 60 * 1000;

function startOfDay( date )
{
    const year = date.getFullYear();
    const month = date.getMonth();
    const day = date.getDate();
    const midnight = new Date( year, month, day );

    return midnight.getTime();
}

/**
 * Conversations newest first, in the sidebar's four groups. Days are the viewer's local days;
 * an empty group is left out.
 */
export function groupConversations( items, now )
{
    const today = startOfDay( now );
    const yesterday = today - dayMilliseconds;
    const weekAgo = today - 7 * dayMilliseconds;
    const groups = [
        { label: 'Today', items: [] },
        { label: 'Yesterday', items: [] },
        { label: 'Previous 7 days', items: [] },
        { label: 'Older', items: [] }
    ];

    const sorted = items.slice().sort( compareUpdatedDescending );

    for ( const item of sorted )
    {
        const updated = new Date( item.updated ).getTime();
        const index = updated >= today ? 0 : updated >= yesterday ? 1 : updated >= weekAgo ? 2 : 3;
        groups[ index ].items.push( item );
    }

    return groups.filter( ( group ) => group.items.length > 0 );
}

function compareUpdatedDescending( left, right )
{
    return new Date( right.updated ).getTime() - new Date( left.updated ).getTime();
}
