// Every setting, declared once: its area, its words, what it needs before it can change, and its control.
// The Settings window shows an area's settings, or every one a search matches, from this table alone.

import { element, button } from '../dom.js';
import { formatBytes } from '../format.js';

export const areas = [
    { id: 'models', label: 'Models', note: null },
    { id: 'responses', label: 'Responses', note: null },
    { id: 'sampling', label: 'Sampling', note: 'Each model opens at its publisher\'s recommended values.' },
    { id: 'search', label: 'Web search', note: null },
    { id: 'build', label: 'Build & Code', note: null },
    { id: 'display', label: 'Display', note: null }
];

const detailChoices = [
    { value: 'off', label: 'Answer only' },
    { value: 'thoughts', label: 'With thinking' },
    { value: 'all', label: 'Everything, with logs' }
];

function switchControl( on, label, action )
{
    const toggle = button( 'switch', null, action );

    toggle.setAttribute( 'role', 'switch' );
    toggle.setAttribute( 'aria-checked', on ? 'true' : 'false' );
    toggle.setAttribute( 'aria-label', label );

    return toggle;
}

function isOn( target )
{
    return target.getAttribute( 'aria-checked' ) === 'true';
}

function progressBar( progress )
{
    const container = element( 'div', 'progress' );
    const label = element( 'div', 'progress-label', progress.label );
    const bar = element( 'progress', 'progress-bar' );

    if ( progress.total !== null )
    {
        bar.max = progress.total;
        bar.value = progress.done;
    }

    container.append( label, bar );

    return container;
}

function modelRow( model )
{
    const row = element( 'div', 'model-row' );
    const text = element( 'div', 'model-text' );
    const name = element( 'div', 'model-name', model.name );
    const facts = element( 'div', 'model-facts', formatBytes( model.size_bytes ) + ' · ' + model.quantization );
    const controls = element( 'div', 'model-controls' );

    text.append( name, facts );
    row.append( text, controls );

    return { row, controls };
}

/**
 * Above the models: a refusal in the session's own words, until the next load, install or remove, and an
 * install's progress.
 */
function modelsLead( state )
{
    const shown = [];
    const progress = state.progress;

    if ( state.modelsError !== null )
    {
        const refusal = element( 'p', 'notice error', state.modelsError.text );
        refusal.setAttribute( 'role', 'alert' );
        shown.push( refusal );
    }

    if ( progress !== null && ( progress.total === null || progress.done < progress.total ) )
        shown.push( progressBar( progress ) );

    return shown;
}

const installed = {
    id: 'installed',
    area: 'models',
    label: 'Installed',
    terms: [ 'models', 'load', 'remove', 'delete', 'quantization', 'size' ],
    needs: null,
    layout: 'list',
    reads: [ 'installedModels', 'session' ],
    note: ( state ) => state.installedModels.length === 0 ? 'No model is installed yet.' : null,
    offered: () => true,
    build( state, view )
    {
        const current = state.session === null ? null : state.session.model;
        const list = element( 'div', 'model-list' );

        if ( state.installedModels.length === 0 )
            return null;

        for ( const model of state.installedModels )
        {
            const { row, controls } = modelRow( model );

            if ( view.local.confirmingRemove === model.name )
            {
                const question = element( 'span', 'confirm-question', 'Remove it from this computer?' );
                const confirm = button( 'confirm-delete', 'Remove', 'remove-confirmed' );
                const cancel = button( 'confirm-cancel', 'Cancel', 'remove-cancelled' );
                confirm.dataset.name = model.name;
                controls.append( question, confirm, cancel );
                list.append( row );

                continue;
            }

            if ( model.name === current )
            {
                controls.append( element( 'span', 'model-badge', 'Loaded' ) );
            }
            else
            {
                const load = button( 'chip-button', 'Load', 'load' );
                load.dataset.name = model.name;
                controls.append( load );
            }

            const remove = button( 'chip-button quiet', 'Remove', 'remove' );
            remove.dataset.name = model.name;
            controls.append( remove );
            list.append( row );
        }

        return list;
    },
    onAction( action, target, view )
    {
        const name = target.dataset.name;

        if ( action === 'load' )
        {
            const message = { type: 'model.load', name, quantization: null };
            view.transmit( message );
        }

        if ( action === 'remove' )
        {
            const change = { confirmingRemove: name };
            view.changeLocal( change );
        }

        if ( action === 'remove-cancelled' )
        {
            const change = { confirmingRemove: null };
            view.changeLocal( change );
        }

        if ( action === 'remove-confirmed' )
        {
            const message = { type: 'model.remove', name };
            view.transmit( message );
            const change = { confirmingRemove: null };
            view.changeLocal( change );
        }
    }
};

const available = {
    id: 'available',
    area: 'models',
    label: 'Available to install',
    terms: [ 'models', 'install', 'download', 'published' ],
    needs: null,
    layout: 'list',
    reads: [ 'publishedModels' ],
    note: ( state ) => state.publishedModels === null ? 'This needs an internet connection.' : null,
    offered: () => true,
    build( state )
    {
        const list = element( 'div', 'model-list' );

        if ( state.publishedModels === null )
        {
            const row = element( 'div', 'model-row' );
            const text = element( 'div', 'model-text', 'The models published for Mila' );
            const show = button( 'chip-button', 'Show', 'refresh' );
            row.append( text, show );
            list.append( row );

            return list;
        }

        for ( const model of state.publishedModels )
        {
            const { row, controls } = modelRow( model );

            if ( model.installed )
            {
                controls.append( element( 'span', 'model-badge', 'Installed' ) );
            }
            else
            {
                const install = button( 'chip-button', 'Install', 'install' );
                install.dataset.name = model.name;
                controls.append( install );
            }

            list.append( row );
        }

        return list;
    },
    onAction( action, target, view )
    {
        if ( action === 'install' )
        {
            const message = { type: 'model.install', name: target.dataset.name };
            view.transmit( message );
        }

        if ( action === 'refresh' )
        {
            const message = { type: 'models.refresh', online: true };
            view.transmit( message );
        }
    }
};

const thinking = {
    id: 'thinking',
    area: 'responses',
    label: 'Thinking',
    terms: [ 'reasoning', 'think' ],
    needs: 'model',
    reads: [ 'session' ],
    note: () => 'The model reasons before it answers. Turning it off does not reload the model.',
    offered: ( session ) => session.thinking_capable,
    build: ( state ) => switchControl( state.session.thinking !== 'off', 'Thinking', 'thinking' ),
    onAction( action, target, view )
    {
        if ( action === 'thinking' )
        {
            const change = { thinking: !isOn( target ) };
            view.set( change );
        }
    }
};

const effort = {
    id: 'effort',
    area: 'responses',
    label: 'Effort',
    terms: [ 'reasoning', 'thinking', 'level' ],
    needs: 'model',
    reads: [ 'session' ],
    note: () => 'How long the model may think.',
    offered: ( session ) => Boolean( session.effort_levels ),
    build( state )
    {
        const group = element( 'div', 'segmented' );

        for ( const level of state.session.effort_levels )
        {
            const choice = button( 'segment', level, 'effort' );
            choice.dataset.value = level;
            choice.classList.toggle( 'selected', state.session.effort === level );
            group.append( choice );
        }

        return group;
    },
    onAction( action, target, view )
    {
        if ( action === 'effort' )
        {
            const change = { effort: target.dataset.value };
            view.set( change );
        }
    }
};

const context = {
    id: 'context',
    area: 'responses',
    label: 'Context window',
    terms: [ 'context length', 'tokens', 'memory', 'auto' ],
    needs: 'model',
    reads: [ 'session' ],
    note: ( state ) => state.session !== null && state.session.context_automatic
        ? 'Auto: the largest that fits this card.'
        : 'A number you give is used exactly. Changing it keeps the conversation.',
    offered: () => true,
    build( state )
    {
        const row = element( 'div', 'setting-row' );
        const field = element( 'input', 'context-input number-input' );
        const automatic = button( 'chip-button', 'Auto', 'context-auto' );

        field.type = 'number';
        field.min = '1';
        field.step = '1024';
        field.value = String( state.session.context_length );
        field.setAttribute( 'aria-label', 'Context window in tokens' );
        automatic.classList.toggle( 'selected', state.session.context_automatic );
        row.append( field, automatic );

        return row;
    },
    onAction( action, target, view )
    {
        if ( action === 'context-auto' )
        {
            const change = { context: 'auto' };
            view.set( change );
        }
    },
    onChange( target, view )
    {
        if ( target.classList.contains( 'context-input' ) )
        {
            const change = { context: Number( target.value ) };
            view.set( change );
        }
    }
};

function samplingSetting( key, label, terms )
{
    return {
        id: key,
        area: 'sampling',
        label,
        terms,
        needs: 'model',
        reads: [ 'session' ],
        note: () => null,
        offered: ( session ) => session.sampling[ key ] !== undefined,
        build( state )
        {
            const entry = state.session.sampling[ key ];
            const row = element( 'div', 'setting-row' );
            const field = element( 'input', 'sampling-input number-input' );
            const reset = button( 'chip-button', 'Model default', 'sampling-default' );

            field.type = 'number';
            field.step = key === 'top_k' ? '1' : '0.05';
            field.value = String( entry.value );
            field.setAttribute( 'aria-label', label );
            reset.classList.toggle( 'selected', entry.is_model_default );
            row.append( field, reset );

            return row;
        },
        onAction( action, target, view )
        {
            if ( action === 'sampling-default' )
            {
                const change = { sampling: { key, value: 'model' } };
                view.set( change );
            }
        },
        onChange( target, view )
        {
            if ( target.classList.contains( 'sampling-input' ) )
            {
                const change = { sampling: { key, value: Number( target.value ) } };
                view.set( change );
            }
        }
    };
}

const seed = {
    id: 'seed',
    area: 'sampling',
    label: 'Seed',
    terms: [ 'random', 'repeat', 'reproduce' ],
    needs: 'model',
    reads: [ 'session' ],
    note: () => 'The same seed, message and conversation give the same reply.',
    offered: () => true,
    build()
    {
        const row = element( 'div', 'setting-row' );
        const field = element( 'input', 'seed-input number-input' );
        const apply = button( 'chip-button', 'Set', 'seed' );

        field.type = 'number';
        field.min = '0';
        field.step = '1';
        field.setAttribute( 'aria-label', 'Seed' );
        row.append( field, apply );

        return row;
    },
    onAction( action, target, view )
    {
        if ( action !== 'seed' )
            return;

        const field = target.closest( '[data-setting]' ).querySelector( '.seed-input' );

        if ( field.value.trim().length === 0 )
            return;

        const change = { seed: Number( field.value ) };
        view.set( change );
    }
};

/**
 * The person's own search key. The page never holds it after Save: the field is cleared, and the session
 * says only whether a key is set.
 */
const searchKey = {
    id: 'search_key',
    area: 'search',
    label: 'Tavily key',
    terms: [ 'web', 'search', 'internet', 'api key', 'tavily' ],
    needs: 'session',
    reads: [ 'session' ],
    note: ( state ) => state.session !== null && state.session.web_search === true
        ? 'Research & Analyze and Get Things Done can search the web with your Tavily key. Each search asks first, and its words go to Tavily.'
        : 'Add a Tavily key to let Research & Analyze and Get Things Done search the web. tavily.com gives 1,000 searches a month free.',
    offered: () => true,
    build( state )
    {
        const row = element( 'div', 'setting-row' );

        if ( state.session.web_search === true )
        {
            row.append( button( 'chip-button', 'Remove key', 'search-key-remove' ) );

            return row;
        }

        const field = element( 'input', 'search-key-input number-input' );
        const save = button( 'chip-button', 'Save', 'search-key-save' );

        field.type = 'password';
        field.autocomplete = 'off';
        field.spellcheck = false;
        field.placeholder = 'tvly-…';
        field.setAttribute( 'aria-label', 'Tavily key' );
        row.append( field, save );

        return row;
    },
    onAction( action, target, view )
    {
        if ( action === 'search-key-remove' )
        {
            const change = { search_key: null };
            view.set( change );
        }

        if ( action !== 'search-key-save' )
            return;

        const field = target.closest( '[data-setting]' ).querySelector( '.search-key-input' );
        const key = field.value.trim();
        field.value = '';

        if ( key.length > 0 )
        {
            const change = { search_key: key };
            view.set( change );
        }
    }
};

/**
 * What the setting does and does not, and its known gaps, beside it (ContainedPrograms.md 6).
 */
const runCommands = {
    id: 'run_commands',
    area: 'build',
    label: 'Run commands (experimental)',
    terms: [ 'commands', 'container', 'terminal', 'shell', 'git', 'build', 'test' ],
    needs: 'session',
    reads: [ 'session' ],
    note: () => 'Build & Code can run commands such as builds, tests and git in a Windows container limited to the ' +
        'project folder and the program folders below, on a desktop of their own. Each command asks first. A ' +
        'command has no internet unless it asks and you allow it; one that has it can send what it reads. ' +
        'Needs Windows 11 26H2 or later.',
    offered: () => true,
    build: ( state ) => switchControl( state.session.run_commands === true, 'Run commands', 'run-commands' ),
    /** Under the note: why commands cannot run, the folders they can read, and the known gaps. */
    details( state )
    {
        const session = state.session;
        const on = session.run_commands === true;
        const shown = [];

        if ( on && session.run_commands_problem !== null )
            shown.push( element( 'p', 'setting-footer setting-problem', 'Commands cannot run here: ' + session.run_commands_problem ) );

        if ( on && session.program_folders.length > 0 )
        {
            const folders = element( 'ul', 'setting-folders' );

            for ( const folder of session.program_folders )
                folders.append( element( 'li', null, folder ) );

            shown.push( element( 'p', 'setting-footer', 'Program folders commands can read:' ), folders );
        }

        shown.push( element( 'p', 'setting-footer',
            'Known gaps: a file given a second name outside the folder while a command runs can be read through ' +
            'it, and a command can read the settings programs keep for you in the registry.' ) );

        return shown;
    },
    onAction( action, target, view )
    {
        if ( action === 'run-commands' )
        {
            const change = { run_commands: !isOn( target ) };
            view.set( change );
        }
    }
};

const detail = {
    id: 'detail',
    area: 'display',
    label: 'Detail',
    terms: [ 'logs', 'thinking', 'show', 'verbose' ],
    needs: 'model',
    reads: [ 'session' ],
    note: () => 'What a reply shows besides its answer.',
    offered: () => true,
    build( state )
    {
        const select = element( 'select', 'detail-select' );

        for ( const choice of detailChoices )
        {
            const option = element( 'option', null, choice.label );
            option.value = choice.value;
            option.selected = choice.value === state.session.detail;
            select.append( option );
        }

        select.setAttribute( 'aria-label', 'Detail' );

        return select;
    },
    onChange( target, view )
    {
        if ( target.classList.contains( 'detail-select' ) )
        {
            const change = { detail: target.value };
            view.set( change );
        }
    }
};

export const settings = [
    installed,
    available,
    thinking,
    effort,
    context,
    samplingSetting( 'temperature', 'Temperature', [ 'random', 'varied', 'creative' ] ),
    samplingSetting( 'top_k', 'Top-k', [ 'top k' ] ),
    samplingSetting( 'top_p', 'Top-p', [ 'top p', 'nucleus' ] ),
    samplingSetting( 'presence_penalty', 'Presence penalty', [ 'repetition', 'repeat' ] ),
    seed,
    searchKey,
    runCommands,
    detail
];

export const leads = { models: { reads: [ 'modelsError', 'progress' ], build: modelsLead } };

export function areaOf( identifier )
{
    return areas.find( ( area ) => area.id === identifier );
}

export function settingOf( identifier )
{
    return settings.find( ( setting ) => setting.id === identifier );
}

/** Whether what a setting needs is there; a setting without it is listed, with no control. */
export function isReady( setting, state )
{
    if ( setting.needs === 'session' )
        return state.session !== null;

    if ( setting.needs === 'model' )
        return state.session !== null && state.session.model !== null;

    return true;
}

/** A setting the loaded model does not have is left out; with no model, what it has is not known yet. */
export function isShown( setting, state )
{
    if ( !isReady( setting, state ) )
        return true;

    return setting.offered( state.session );
}

function wordsOf( setting, state )
{
    const area = areaOf( setting.area );
    const note = setting.note( state );
    const parts = [ setting.label, area.label, ...setting.terms ];

    if ( note !== null )
        parts.push( note );

    return parts.join( ' ' ).toLowerCase();
}

/** The settings a search names: every word of it in a setting's label, area, words or note. */
export function searchSettings( query, state )
{
    const words = query.toLowerCase().split( /\s+/ ).filter( ( word ) => word.length > 0 );

    if ( words.length === 0 )
        return [];

    const matches = ( setting ) => words.every( ( word ) => wordsOf( setting, state ).includes( word ) );

    return settings.filter( ( setting ) => isShown( setting, state ) && matches( setting ) );
}

export function settingsInArea( identifier, state )
{
    return settings.filter( ( setting ) => setting.area === identifier && isShown( setting, state ) );
}
