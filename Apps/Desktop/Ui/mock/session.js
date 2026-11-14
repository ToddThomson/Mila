// A mock of the session process: it answers MilaDesktop.md section 4.2's requests with section
// 4.1's events, as the session would, and plays a scenario's scripted reply for each turn. No DOM
// and no timers of its own -- the host passes a clock -- so one implementation serves the page in
// a browser, a Node wrapper on stdio, and node --test.

import { findModel, models, seedConversations, titleFor } from './fixtures.js';
import { findScenario } from './scenarios.js';

const streamId = 'stream-1';
const offeredActivities = [ 'ask', 'build', 'research', 'tasks' ];
const folderActivities = [ 'build', 'tasks' ];
const mockFolder = 'C:\\Users\\you\\Projects\\calc';

/**
 * @param options.scenario The scenario's number (section 5).
 * @param options.emit Called with each event, a plain object.
 * @param options.schedule Calls a function after a delay in milliseconds; returns a handle.
 * @param options.cancel Cancels a handle from schedule.
 * @param options.now Milliseconds since the epoch.
 * @param options.log Optional; given a line for the log, such as an ignored request.
 */
export function createMockSession( options )
{
    const scenario = findScenario( options.scenario );
    const startModel = scenario.model === null ? null : findModel( scenario.model );
    const startTime = options.now();
    const seeds = seedConversations( startTime );
    const conversations = new Map();
    // The models on this computer: every fixture, unless the scenario starts with fewer or none.
    const everyModel = models.map( ( model ) => model.name );
    const installed = new Set( scenario.installed === undefined ? everyModel : scenario.installed );
    let publishedListed = false;

    for ( const seed of seeds )
        conversations.set( seed.id, seed );

    const session = {
        model: startModel,
        noModelReason: null,
        loading: null,
        contextLength: startModel === null ? 0 : startModel.context_length,
        contextAutomatic: true,
        thinking: true,
        effort: null,
        detail: 'thoughts',
        sampling: {},
        busy: false,
        conversationId: null,
        activity: 'ask',
        folder: null,
        webSearch: false,
        history: [],
        nextConversationNumber: seeds.length + 1,
        nextTurnNumber: 1
    };

    const attachments = new Map();
    const allowedTools = new Set();
    let pending = null;
    let reply = null;

    function emit( message )
    {
        const versioned = { v: 1, ...message };
        options.emit( versioned );
    }

    function thinkingValue()
    {
        if ( session.model === null || !session.model.thinking_capable || !session.thinking )
            return 'off';

        return session.model.effort_levels !== null ? effortInEffect() : 'on';
    }

    // The level chosen where the model's template lists it, or the template's own default.
    function effortInEffect()
    {
        const levels = session.model.effort_levels;

        return levels.includes( session.effort ) ? session.effort : session.model.effort_default;
    }

    function samplingState()
    {
        const state = {};

        if ( session.model === null )
            return state;

        for ( const key of Object.keys( session.model.sampling ) )
            state[ key ] = samplingEntry( key );

        return state;
    }

    function samplingEntry( key )
    {
        const chosen = Object.hasOwn( session.sampling, key );
        const value = chosen ? session.sampling[ key ] : session.model.sampling[ key ];

        return { value, is_model_default: !chosen };
    }

    function emitState()
    {
        const model = session.model;
        const stream = { stream_id: streamId, parent_stream_id: null, conversation_id: session.conversationId, busy: session.busy };

        const state = {
            type: 'session.state',
            model: model === null ? null : model.name,
            loading: session.loading,
            no_model_reason: session.noModelReason,
            context_length: session.contextLength,
            context_automatic: session.contextAutomatic,
            thinking_capable: model !== null && model.thinking_capable,
            thinking: thinkingValue(),
            effort: model !== null && model.effort_levels !== null ? effortInEffect() : null,
            effort_levels: model === null ? null : model.effort_levels,
            detail: session.detail,
            sampling: samplingState(),
            modalities: model === null ? [] : model.modalities,
            activities: offeredActivities,
            activity: session.activity,
            folder: session.folder,
            plan: null,
            web_search: session.webSearch,
            busy: session.busy,
            streams: [ stream ]
        };

        emit( state );
    }

    function conversationItem( conversation )
    {
        return {
            id: conversation.id,
            title: conversation.title,
            updated: conversation.updated,
            model: conversation.model,
            activity: conversation.activity,
            folder: conversation.folder
        };
    }

    function emitConversations()
    {
        const values = Array.from( conversations.values() );
        const items = values.map( conversationItem );
        const message = { type: 'conversations.list', items };

        emit( message );
    }

    function installedItem( model )
    {
        return { name: model.name, size_bytes: model.size_bytes, quantization: model.quantization, family: model.family };
    }

    /** A fixture as the publisher offers it, planned as the session plans it: on the whole card when one is resident. */
    function publishedItem( model )
    {
        const facts = model.published;
        const fits = facts.planned_context !== null;

        return {
            name: model.name,
            size_bytes: model.size_bytes,
            quantization: model.quantization,
            installed: installed.has( model.name ),
            license: facts.license,
            agentic: facts.agentic,
            thinking: model.thinking_capable,
            tools: facts.tools,
            modalities: model.modalities,
            fits,
            agentic_here: fits && facts.agentic && facts.planned_context >= 65536,
            context_length: fits ? facts.planned_context : null,
            why_not: fits ? null : facts.why_not
        };
    }

    /** The library's ranking: agentic here first, then what fits, the larger download first within each. */
    function rankPublished( items )
    {
        const tier = ( item ) => item.agentic_here ? 0 : item.fits ? 1 : 2;

        return [ ...items ].sort( ( left, right ) => tier( left ) - tier( right ) || right.size_bytes - left.size_bytes );
    }

    /** Every fixture is published; the list is sent once the page has asked for it online, as the session's is. */
    function emitModels()
    {
        const onComputer = models.filter( ( model ) => installed.has( model.name ) );
        const installedItems = onComputer.map( installedItem );
        const published = publishedListed ? rankPublished( models.map( publishedItem ) ) : null;
        const first = published === null ? undefined : published.find( ( item ) => item.fits );
        const suggested = first === undefined ? null : first.name;
        const message = { type: 'models.list', installed: installedItems, published, suggested };

        emit( message );
    }

    function emitError( text, during )
    {
        const message = { type: 'error', text, during };

        emit( message );
    }

    // ---- Turns -------------------------------------------------------------------------------

    function startTurn( request )
    {
        if ( session.busy )
        {
            emitError( 'A reply is still being written. Stop it first, or wait for it to finish.', 'turn' );

            return;
        }

        if ( session.model === null )
        {
            emitError( 'No model is loaded. Choose one from the model menu.', 'turn' );

            return;
        }

        if ( session.conversationId === null )
        {
            const identifier = 'conversation-' + session.nextConversationNumber;
            session.nextConversationNumber = session.nextConversationNumber + 1;
            session.conversationId = identifier;
        }

        const turnId = 'turn-' + session.nextTurnNumber;
        const sent = takeAttachments( request.attachment_ids );
        const context = { attachments: sent, model: session.model };
        session.nextTurnNumber = session.nextTurnNumber + 1;
        session.busy = true;

        reply = {
            turnId,
            request,
            attachments: sent,
            steps: scenario.createReply( request, context ),
            next: 0,
            waiting: null,
            thought: '',
            answer: '',
            toolCalls: [],
            rounds: []
        };

        emitState();
        scheduleNextStep();
    }

    function scheduleNextStep()
    {
        if ( reply.next >= reply.steps.length )
            return;

        const step = reply.steps[ reply.next ];
        pending = options.schedule( playStep, step.delay );
    }

    function playStep()
    {
        pending = null;

        const step = reply.steps[ reply.next ];
        reply.next = reply.next + 1;

        if ( step.branches !== undefined )
        {
            askOrContinue( step );

            return;
        }

        const message = { ...step.message, stream_id: streamId, turn_id: reply.turnId };

        if ( message.type === 'turn.begin' )
            message.conversation_id = session.conversationId;

        record( message );
        emit( message );

        if ( message.type === 'turn.end' )
        {
            finishTurn( message );

            return;
        }

        scheduleNextStep();
    }

    /**
     * A tool allowed in this conversation runs without asking (MilaDesktop.md section 4.5); any
     * other waits for turn.tool_decision, with nothing scheduled until it comes.
     */
    function askOrContinue( step )
    {
        if ( allowedTools.has( step.tool ) )
        {
            continueWith( step.branches.allow );

            return;
        }

        const approval = { ...step.message, stream_id: streamId, turn_id: reply.turnId };
        reply.waiting = step;
        emit( approval );
    }

    function continueWith( steps )
    {
        reply.steps = [ ...reply.steps.slice( 0, reply.next ), ...steps ];
        scheduleNextStep();
    }

    function decideTool( request )
    {
        const waiting = reply === null ? null : reply.waiting;
        const answers = waiting !== null && request.turn_id === reply.turnId && request.call_id === waiting.message.call_id;

        if ( !answers || !waiting.message.choices.includes( request.decision ) )
        {
            ignore( request );

            return;
        }

        reply.waiting = null;

        const call = reply.toolCalls.find( ( candidate ) => candidate.call_id === request.call_id );

        if ( call !== undefined )
            call.decision = request.decision;

        if ( request.decision === 'allow_conversation' )
            allowedTools.add( waiting.tool );

        const steps = request.decision === 'deny' ? waiting.branches.deny : waiting.branches.allow;
        continueWith( steps );
    }

    function record( message )
    {
        if ( message.type === 'turn.text' && message.channel === 'thought' )
            reply.thought = reply.thought + message.text;

        if ( message.type === 'turn.text' && message.channel === 'answer' )
            reply.answer = reply.answer + message.text;

        if ( message.type === 'turn.tool_call' )
        {
            const call = { call_id: message.call_id, name: message.name, summary: message.summary, arguments: message.arguments, decision: null, result: null, is_error: false };
            reply.toolCalls.push( call );
        }

        if ( message.type === 'turn.tool_result' )
            recordToolResult( message );

        if ( message.type === 'turn.round_end' )
            reply.rounds.push( message );
    }

    function recordToolResult( message )
    {
        const call = reply.toolCalls.find( ( candidate ) => candidate.call_id === message.call_id );

        if ( call === undefined )
            return;

        call.result = message.result;
        call.is_error = message.is_error;
    }

    function stopTurn()
    {
        if ( reply === null )
            return;

        if ( pending !== null )
            options.cancel( pending );

        pending = null;

        const end = { type: 'turn.end', stream_id: streamId, turn_id: reply.turnId, finish: 'cancelled', answer: reply.answer };

        emit( end );
        finishTurn( end );
    }

    function storedStats( finish )
    {
        let prefillMs = 0;
        let tokens = 0;
        let decodeSeconds = 0;

        for ( const round of reply.rounds )
        {
            prefillMs = prefillMs + round.prefill_ms;
            tokens = tokens + round.tokens;
            decodeSeconds = decodeSeconds + round.tokens / round.tokens_per_second;
        }

        const rate = decodeSeconds > 0 ? Math.round( 10 * tokens / decodeSeconds ) / 10 : 0;

        return { prefill_ms: prefillMs, tokens_per_second: rate, tokens, rounds: reply.rounds.length, finish };
    }

    function finishTurn( end )
    {
        const stamp = new Date( options.now() );
        const updated = stamp.toISOString();
        const sentAttachments = reply.attachments.length > 0 ? reply.attachments : null;
        const userMessage = { role: 'user', content: reply.request.text, thought: null, tool_calls: null, attachments: sentAttachments, stats: null };
        const toolCalls = reply.toolCalls.length > 0 ? reply.toolCalls : null;
        const thought = reply.thought.length > 0 ? reply.thought : null;
        const stats = storedStats( end.finish );
        const assistantMessage = { role: 'assistant', content: end.answer, thought, tool_calls: toolCalls, attachments: null, stats };

        session.history.push( userMessage );
        session.history.push( assistantMessage );

        const existing = conversations.get( session.conversationId );

        if ( existing === undefined )
        {
            const title = titleFor( reply.request.text );
            const created = {
                id: session.conversationId,
                title,
                model: session.model.name,
                activity: session.activity,
                folder: session.folder,
                created: updated,
                updated,
                messages: session.history.slice()
            };
            conversations.set( session.conversationId, created );
        }
        else
        {
            existing.updated = updated;
            existing.messages = session.history.slice();
        }

        session.busy = false;
        reply = null;

        emitState();
        emitConversations();
    }

    // ---- Attachments -------------------------------------------------------------------------

    const decodeMilliseconds = 140;

    /**
     * The accepted attachments a turn names, as a stored message keeps them; each is used once.
     */
    function takeAttachments( identifiers )
    {
        const taken = [];

        for ( const identifier of identifiers )
        {
            const attachment = attachments.get( identifier );

            if ( attachment === undefined || attachment.status !== 'accepted' )
                continue;

            const stored = { kind: attachment.kind, name: attachment.name };
            taken.push( stored );
            attachments.delete( identifier );
        }

        return taken;
    }

    function refusalFor( request )
    {
        if ( session.model === null )
            return 'No model is loaded to read it.';

        if ( request.kind !== 'image' && request.kind !== 'audio' )
            return 'Only PNG and JPEG images and WAV audio can be attached.';

        if ( !session.model.modalities.includes( request.kind ) )
        {
            const what = request.kind === 'image' ? 'images' : 'audio';

            return session.model.name + ' cannot take ' + what + '.';
        }

        if ( request.path === null || request.path.length === 0 )
            return 'That file could not be found on this computer.';

        return null;
    }

    function addAttachment( request )
    {
        const attachment = { kind: request.kind, name: request.name, status: 'pending' };
        const decide = () => decideAttachment( request );

        if ( options.log !== undefined )
            options.log( 'attachment ' + request.attachment_id + ' at ' + request.path );

        attachments.set( request.attachment_id, attachment );
        options.schedule( decide, decodeMilliseconds );
    }

    function decideAttachment( request )
    {
        const attachment = attachments.get( request.attachment_id );

        if ( attachment === undefined )
            return;

        const reason = refusalFor( request );

        if ( reason !== null )
        {
            attachments.delete( request.attachment_id );

            const refused = { type: 'attachment.refused', attachment_id: request.attachment_id, reason };
            emit( refused );

            return;
        }

        attachment.status = 'accepted';

        const sizeBytes = scenario.sample === undefined ? 0 : scenario.sample.sizeBytes;
        const accepted = { type: 'attachment.accepted', attachment_id: request.attachment_id, kind: request.kind, name: request.name, size_bytes: sizeBytes };
        emit( accepted );
    }

    function removeAttachment( request )
    {
        attachments.delete( request.attachment_id );
    }

    // ---- Conversations -----------------------------------------------------------------------

    /**
     * A folder is any path the mock is given, since the mock reads no files; the real session refuses one
     * that does not exist.
     */
    function newConversation( request )
    {
        const activity = request.activity === undefined ? 'ask' : request.activity;
        const folder = typeof request.folder === 'string' && request.folder.length > 0 ? request.folder : null;

        if ( !offeredActivities.includes( activity ) )
        {
            emitError( 'There is no activity called ' + activity + '.', 'conversation' );

            return;
        }

        if ( folderActivities.includes( activity ) && folder === null )
        {
            emitError( 'This works in a folder; choose one to start.', 'conversation' );

            return;
        }

        session.conversationId = null;
        session.activity = activity;
        session.folder = folderActivities.includes( activity ) ? folder : null;
        session.history = [];
        allowedTools.clear();

        emitState();
    }

    function openConversation( request )
    {
        const conversation = conversations.get( request.id );

        if ( conversation === undefined )
        {
            emitError( 'That conversation is no longer stored.', null );

            return;
        }

        session.conversationId = conversation.id;
        session.activity = conversation.activity;
        session.folder = conversation.folder;
        session.history = conversation.messages.slice();
        allowedTools.clear();

        const loaded = {
            type: 'conversation.loaded',
            id: conversation.id,
            title: conversation.title,
            model: conversation.model,
            activity: conversation.activity,
            folder: conversation.folder,
            messages: conversation.messages
        };

        emit( loaded );
        emitState();
    }

    function renameConversation( request )
    {
        const conversation = conversations.get( request.id );
        const title = request.title.trim();

        if ( conversation === undefined || title.length === 0 )
            return;

        conversation.title = title;
        emitConversations();
    }

    function deleteConversation( request )
    {
        conversations.delete( request.id );

        if ( session.conversationId === request.id )
            newConversation();

        emitConversations();
    }

    // ---- Settings ----------------------------------------------------------------------------

    function changeSetting( request )
    {
        if ( Object.hasOwn( request, 'context' ) )
            return changeContext( request.context );

        if ( Object.hasOwn( request, 'thinking' ) )
            return changeThinking( request.thinking );

        if ( Object.hasOwn( request, 'effort' ) )
            return changeEffort( request.effort );

        if ( Object.hasOwn( request, 'detail' ) )
            return changeDetail( request.detail );

        if ( Object.hasOwn( request, 'sampling' ) )
            return changeSampling( request.sampling );

        if ( Object.hasOwn( request, 'seed' ) )
            return changeSeed( request.seed );

        // The mock keeps no key: it says only whether one is set, as the session does.
        if ( Object.hasOwn( request, 'search_key' ) )
        {
            session.webSearch = typeof request.search_key === 'string' && request.search_key.trim().length > 0;

            return emitState();
        }

        emitError( 'That is not a setting this session has.', 'settings' );
    }

    function changeContext( value )
    {
        if ( session.model === null )
        {
            emitError( 'Load a model before choosing its context window.', 'settings' );

            return;
        }

        if ( value === 'auto' )
        {
            session.contextLength = session.model.context_length;
            session.contextAutomatic = true;
        }
        else if ( Number.isInteger( value ) && value > 0 && value <= session.model.context_length )
        {
            session.contextLength = value;
            session.contextAutomatic = false;
        }
        else
        {
            const largest = session.model.context_length.toLocaleString( 'en-US' );
            const text = 'A context window of ' + value + ' does not fit this card. The largest that fits is ' + largest + '.';
            emitError( text, 'settings' );

            return;
        }

        emitState();
    }

    function changeThinking( value )
    {
        if ( session.model === null || !session.model.thinking_capable )
        {
            emitError( 'This model has no reasoning channel.', 'settings' );

            return;
        }

        session.thinking = value === true;
        emitState();
    }

    function changeEffort( value )
    {
        if ( session.model === null || session.model.effort_levels === null )
        {
            emitError( 'This model\'s thinking is on or off; it has no effort levels.', 'settings' );

            return;
        }

        if ( !session.model.effort_levels.includes( value ) )
        {
            emitError( 'Effort is one of ' + session.model.effort_levels.join( ', ' ) + '.', 'settings' );

            return;
        }

        session.effort = value;
        emitState();
    }

    function changeDetail( value )
    {
        if ( value !== 'off' && value !== 'thoughts' && value !== 'all' )
        {
            emitError( 'Detail is off, thoughts or all.', 'settings' );

            return;
        }

        session.detail = value;
        emitState();
    }

    function changeSampling( change )
    {
        if ( session.model === null || !Object.hasOwn( session.model.sampling, change.key ) )
        {
            emitError( 'That is not a sampling setting.', 'settings' );

            return;
        }

        if ( change.value === 'model' )
            delete session.sampling[ change.key ];
        else
            session.sampling[ change.key ] = change.value;

        emitState();
    }

    function changeSeed( value )
    {
        if ( !Number.isInteger( value ) || value < 0 )
        {
            emitError( 'A seed is a whole number, 0 or more.', 'settings' );

            return;
        }

        const message = { type: 'info', text: 'Sampler reseeded (' + value + ').' };

        emit( message );
        emitState();
    }

    // ---- Models ------------------------------------------------------------------------------

    const loadSteps = 8;
    const loadStepMilliseconds = 180;

    function loadModel( request )
    {
        const model = findModel( request.name );

        if ( model === null || !installed.has( model.name ) )
        {
            const text = 'No installed model is called ' + request.name + '.';
            emitError( text, 'load' );

            return;
        }

        if ( session.busy )
        {
            emitError( 'Wait for the reply to finish before loading another model.', 'load' );

            return;
        }

        session.busy = true;
        session.loading = model.name;
        emitState();
        scheduleLoadStep( model, 0 );
    }

    function scheduleLoadStep( model, done )
    {
        const step = () => playLoadStep( model, done );

        pending = options.schedule( step, loadStepMilliseconds );
    }

    function playLoadStep( model, done )
    {
        pending = null;

        const progress = { type: 'progress', task: 'load', label: 'Loading ' + model.name, done, total: loadSteps };
        emit( progress );

        if ( done < loadSteps )
        {
            const next = done + 1;
            scheduleLoadStep( model, next );

            return;
        }

        // A refused load leaves nothing resident, as the session's does: the model that was loaded is gone too.
        const refusals = scenario.refusals === undefined ? {} : scenario.refusals;
        const refusal = refusals[ model.name ];

        if ( refusal !== undefined )
        {
            session.model = null;
            session.loading = null;
            session.busy = false;
            emitError( 'Could not load ' + model.name + ': ' + refusal, 'load' );
            emitState();

            return;
        }

        session.model = model;
        session.loading = null;
        session.contextLength = model.context_length;
        session.contextAutomatic = true;
        session.sampling = {};
        session.busy = false;
        session.conversationId = null;
        session.history = [];
        attachments.clear();
        allowedTools.clear();
        emitState();
    }

    const installSteps = 10;
    const installStepMilliseconds = 150;

    // Planning every published model takes a moment the page shows; the session's time is not measured yet.
    const planningMilliseconds = 900;

    function refreshModels( request )
    {
        if ( !request.online )
        {
            emitModels();

            return;
        }

        if ( scenario.publisher === 'unreachable' )
        {
            const answer = () =>
            {
                emitError( 'Could not list the models that can be installed: the connection failed.', 'models' );
                emitModels();
            };

            options.schedule( answer, planningMilliseconds );

            return;
        }

        const answer = () =>
        {
            publishedListed = true;
            emitModels();
        };

        options.schedule( answer, planningMilliseconds );
    }

    function installModel( request )
    {
        const model = findModel( request.name );

        if ( model === null )
        {
            const text = 'Install failed: nothing published is called ' + request.name + '.';
            emitError( text, 'install' );

            return;
        }

        if ( installed.has( model.name ) )
        {
            const text = 'Install failed: ' + model.name + ' is already installed.';
            emitError( text, 'install' );

            return;
        }

        scheduleInstallStep( model, 0 );
    }

    function scheduleInstallStep( model, step )
    {
        const next = () => playInstallStep( model, step );

        options.schedule( next, installStepMilliseconds );
    }

    /** The install's bytes as they arrive, then the model is on the computer. */
    function playInstallStep( model, step )
    {
        const done = Math.round( model.size_bytes * step / installSteps );
        const progress = { type: 'progress', task: 'install', label: 'Installing ' + model.name, done, total: model.size_bytes };
        emit( progress );

        if ( step < installSteps )
        {
            scheduleInstallStep( model, step + 1 );

            return;
        }

        installed.add( model.name );

        const said = { type: 'info', text: 'Installed ' + model.name + '.', during: 'install' };
        emit( said );
        emitModels();

        // As the session's: a model installed with nothing resident is loaded.
        if ( session.model === null && !session.busy )
        {
            const load = { name: model.name };
            loadModel( load );
        }
    }

    /** As the session's: the files go, a loaded model stays loaded, and a failure names no task. */
    function removeModel( request )
    {
        const model = findModel( request.name );

        if ( model === null || !installed.has( model.name ) )
        {
            const text = 'Remove failed: no installed model is called ' + request.name + '.';
            emitError( text, null );

            return;
        }

        installed.delete( model.name );
        emitModels();
    }

    // ---- Requests ----------------------------------------------------------------------------

    function handle( request )
    {
        switch ( request.type )
        {
            case 'ready':
                emitState();
                emitConversations();
                emitModels();

                return;

            case 'turn.send':
                return startTurn( request );

            case 'turn.stop':
                return stopTurn();

            case 'turn.tool_decision':
                return decideTool( request );

            case 'conversation.new':
                return newConversation( request );

            // The window answers this; with no window, as in a browser, the mock does.
            case 'folder.choose':
                return emit( { type: 'folder.chosen', path: mockFolder } );

            case 'conversation.open':
                return openConversation( request );

            case 'conversation.rename':
                return renameConversation( request );

            case 'conversation.delete':
                return deleteConversation( request );

            case 'models.refresh':
                return refreshModels( request );

            case 'model.install':
                return installModel( request );

            case 'model.remove':
                return removeModel( request );

            case 'settings.set':
                return changeSetting( request );

            case 'model.load':
                return loadModel( request );

            case 'attachment.add':
                return addAttachment( request );

            case 'attachment.remove':
                return removeAttachment( request );

            default:
                return ignore( request );
        }
    }

    function ignore( request )
    {
        if ( options.log !== undefined )
            options.log( 'ignored a request of type ' + request.type );
    }

    return { handle, scenario };
}
