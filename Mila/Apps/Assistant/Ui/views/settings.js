// The settings panel. Every control shows what the last session.state said, and a change sends
// one settings.set; the control then follows the reply, not the edit.

import { element, button, iconButton, clear } from '../dom.js';

const samplingLabels = {
    temperature: 'Temperature',
    top_k: 'Top-k',
    top_p: 'Top-p',
    presence_penalty: 'Presence penalty'
};

const detailChoices = [
    { value: 'off', label: 'Answer only' },
    { value: 'thoughts', label: 'With thinking' },
    { value: 'all', label: 'Everything, with logs' }
];

export function createSettingsView( root, actions )
{
    const panelHeader = element( 'div', 'panel-header' );
    const title = element( 'h2', 'panel-title', 'Settings' );
    const close = iconButton( 'quiet-icon', 'close', 'Close settings', 'close-settings' );
    const body = element( 'div', 'panel-body' );
    let lastSession = null;
    let lastOpen = false;

    panelHeader.append( title, close );
    root.append( panelHeader, body );

    function sendSetting( change )
    {
        const message = { type: 'settings.set', ...change };

        actions.send( message );
    }

    function onClick( event )
    {
        const target = event.target.closest( '[data-action]' );

        if ( target === null )
            return;

        const action = target.dataset.action;

        if ( action === 'close-settings' )
        {
            const closed = { settingsOpen: false };
            actions.changeSurface( closed );
        }

        if ( action === 'thinking' )
        {
            const change = { thinking: target.getAttribute( 'aria-checked' ) !== 'true' };
            sendSetting( change );
        }

        if ( action === 'effort' )
        {
            const change = { effort: target.dataset.value };
            sendSetting( change );
        }

        if ( action === 'context-auto' )
        {
            const change = { context: 'auto' };
            sendSetting( change );
        }

        if ( action === 'sampling-default' )
        {
            const sampling = { key: target.dataset.key, value: 'model' };
            const change = { sampling };
            sendSetting( change );
        }

        if ( action === 'seed' )
            sendSeed();

        if ( action === 'search-key-save' )
            sendSearchKey();

        if ( action === 'search-key-remove' )
        {
            const change = { search_key: null };
            sendSetting( change );
        }
    }

    function sendSeed()
    {
        const field = body.querySelector( '.seed-input' );
        const value = Number( field.value );
        const change = { seed: value };

        if ( field.value.trim().length === 0 )
            return;

        sendSetting( change );
    }

    function onChange( event )
    {
        const target = event.target;

        if ( target.classList.contains( 'context-input' ) )
        {
            const change = { context: Number( target.value ) };
            sendSetting( change );
        }

        if ( target.classList.contains( 'sampling-input' ) )
        {
            const sampling = { key: target.dataset.key, value: Number( target.value ) };
            const change = { sampling };
            sendSetting( change );
        }

        if ( target.classList.contains( 'detail-select' ) )
        {
            const change = { detail: target.value };
            sendSetting( change );
        }
    }

    root.addEventListener( 'click', onClick );
    root.addEventListener( 'change', onChange );

    function section( label, note )
    {
        const container = element( 'section', 'setting' );
        const heading = element( 'h3', 'setting-label', label );
        container.append( heading );

        if ( note !== null )
        {
            const text = element( 'p', 'setting-note', note );
            container.append( text );
        }

        body.append( container );

        return container;
    }

    function buildThinking( session )
    {
        const container = section( 'Thinking', 'The model reasons before it answers. Turning it off does not reload the model.' );
        const toggle = button( 'switch', null, 'thinking' );
        const on = session.thinking !== 'off';

        toggle.setAttribute( 'role', 'switch' );
        toggle.setAttribute( 'aria-checked', on ? 'true' : 'false' );
        toggle.setAttribute( 'aria-label', 'Thinking' );
        container.append( toggle );
    }

    function buildEffort( session )
    {
        const container = section( 'Effort', 'How long the model may think.' );
        const group = element( 'div', 'segmented' );

        for ( const level of session.effort_levels )
        {
            const choice = button( 'segment', level, 'effort' );
            choice.dataset.value = level;
            choice.classList.toggle( 'selected', session.effort === level );
            group.append( choice );
        }

        container.append( group );
    }

    function buildContext( session )
    {
        const note = session.context_automatic ? 'Auto: the largest that fits this card.' : 'A number you give is used exactly. Changing it keeps the conversation.';
        const container = section( 'Context window', note );
        const row = element( 'div', 'setting-row' );
        const field = element( 'input', 'context-input number-input' );
        const automatic = button( 'chip-button', 'Auto', 'context-auto' );

        field.type = 'number';
        field.min = '1';
        field.step = '1024';
        field.value = String( session.context_length );
        field.setAttribute( 'aria-label', 'Context window in tokens' );
        automatic.classList.toggle( 'selected', session.context_automatic );
        row.append( field, automatic );
        container.append( row );
    }

    function buildSampling( session )
    {
        const container = section( 'Sampling', 'Each model opens at its publisher\'s recommended values.' );

        for ( const key of Object.keys( session.sampling ) )
        {
            const entry = session.sampling[ key ];
            const row = element( 'div', 'setting-row sampling-row' );
            const label = element( 'label', 'sampling-label', samplingLabels[ key ] );
            const field = element( 'input', 'sampling-input number-input' );
            const reset = button( 'chip-button', 'Model default', 'sampling-default' );
            const identifier = 'sampling-' + key;

            label.htmlFor = identifier;
            field.id = identifier;
            field.type = 'number';
            field.step = key === 'top_k' ? '1' : '0.05';
            field.value = String( entry.value );
            field.dataset.key = key;
            reset.dataset.key = key;
            reset.classList.toggle( 'selected', entry.is_model_default );
            row.append( label, field, reset );
            container.append( row );
        }
    }

    function buildDetail( session )
    {
        const container = section( 'Detail', 'What a reply shows besides its answer.' );
        const select = element( 'select', 'detail-select' );

        for ( const choice of detailChoices )
        {
            const option = element( 'option', null, choice.label );
            option.value = choice.value;
            option.selected = choice.value === session.detail;
            select.append( option );
        }

        select.setAttribute( 'aria-label', 'Detail' );
        container.append( select );
    }

    function buildSeed()
    {
        const container = section( 'Seed', 'The same seed, message and conversation give the same reply.' );
        const row = element( 'div', 'setting-row' );
        const field = element( 'input', 'seed-input number-input' );
        const apply = button( 'chip-button', 'Set', 'seed' );

        field.type = 'number';
        field.min = '0';
        field.step = '1';
        field.setAttribute( 'aria-label', 'Seed' );
        row.append( field, apply );
        container.append( row );
    }

    /**
     * The person's own search key. The page never holds it after Save: the field is cleared, and the session
     * says only whether a key is set.
     */
    function buildWebSearch( session )
    {
        const enabled = session.web_search === true;
        const note = enabled
            ? 'Research & Analyze and Get Things Done can search the web with your Tavily key. Each search asks first, and its words go to Tavily.'
            : 'Add a Tavily key to let Research & Analyze and Get Things Done search the web. tavily.com gives 1,000 searches a month free.';
        const container = section( 'Web search', note );
        const row = element( 'div', 'setting-row' );

        if ( enabled )
        {
            const remove = button( 'chip-button', 'Remove key', 'search-key-remove' );
            row.append( remove );
        }
        else
        {
            const field = element( 'input', 'search-key-input number-input' );
            const save = button( 'chip-button', 'Save', 'search-key-save' );

            field.type = 'password';
            field.autocomplete = 'off';
            field.spellcheck = false;
            field.placeholder = 'tvly-…';
            field.setAttribute( 'aria-label', 'Tavily key' );
            row.append( field, save );
        }

        container.append( row );
    }

    function sendSearchKey()
    {
        const field = body.querySelector( '.search-key-input' );
        const key = field.value.trim();
        field.value = '';

        if ( key.length === 0 )
            return;

        const change = { search_key: key };
        sendSetting( change );
    }

    function build( session )
    {
        clear( body );

        if ( session !== null )
            buildWebSearch( session );

        if ( session === null || session.model === null )
        {
            const empty = element( 'p', 'setting-note', 'Settings apply to a loaded model. Load one to change them.' );
            body.append( empty );

            return;
        }

        if ( session.thinking_capable )
            buildThinking( session );

        if ( session.effort_levels )
            buildEffort( session );

        buildContext( session );
        buildSampling( session );
        buildDetail( session );
        buildSeed();
    }

    function render( state, surface )
    {
        root.hidden = !surface.settingsOpen;

        if ( !surface.settingsOpen )
        {
            lastOpen = false;

            return;
        }

        if ( state.session === lastSession && lastOpen )
            return;

        lastSession = state.session;
        lastOpen = true;
        build( state.session );
    }

    return { render };
}
