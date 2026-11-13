// The Settings window, over the main window: a search and the areas beside one area's settings, or every
// setting a search names. What each setting is comes from settings-table.js; its control shows what the last
// session.state said, and a change sends one settings.set, so the control follows the reply, not the edit.

import { element, button, iconButton, clear } from '../dom.js';
import { areas, leads, areaOf, settingOf, isReady, searchSettings, settingsInArea } from './settings-table.js';

const waitingNotices = {
    model: 'These apply to a loaded model. Load one to change them.',
    session: 'Mila is starting.'
};

export function createSettingsView( root, actions )
{
    const navigation = element( 'nav', 'settings-nav' );
    const search = element( 'input', 'settings-search' );
    const areaList = element( 'div', 'settings-areas' );
    const main = element( 'div', 'settings-main' );
    const panelHeader = element( 'div', 'panel-header' );
    const title = element( 'h2', 'panel-title' );
    const close = iconButton( 'quiet-icon', 'close', 'Close settings', 'close-settings' );
    const body = element( 'div', 'settings-body' );
    const local = { query: '', confirmingRemove: null };
    let area = areas[ 0 ].id;
    let lastInputs = null;
    let open = false;

    search.type = 'search';
    search.placeholder = 'Search settings';
    search.spellcheck = false;
    search.setAttribute( 'aria-label', 'Search settings' );
    navigation.setAttribute( 'aria-label', 'Settings areas' );

    for ( const shown of areas )
    {
        const choice = button( 'settings-area', shown.label, 'settings-area' );
        choice.dataset.area = shown.id;
        areaList.append( choice );
    }

    navigation.append( search, areaList );
    panelHeader.append( title, close );
    main.append( panelHeader, body );
    root.append( navigation, main );

    /** What a setting in the table is given: the person's change, a message, and the window's own state. */
    const view = {
        local,
        set( change )
        {
            const message = { type: 'settings.set', ...change };
            actions.send( message );
        },
        transmit( message )
        {
            actions.send( message );
        },
        changeLocal( change )
        {
            Object.assign( local, change );
            refresh();
        }
    };

    function searching()
    {
        return local.query.trim().length > 0;
    }

    function refresh()
    {
        lastInputs = null;
        update( actions.getState() );
    }

    function showArea( identifier )
    {
        local.query = '';
        search.value = '';
        actions.openSettings( identifier );
    }

    function onClick( event )
    {
        if ( event.target === root )
        {
            actions.closeSettings();

            return;
        }

        const target = event.target.closest( '[data-action]' );

        if ( target === null )
            return;

        const action = target.dataset.action;

        if ( action === 'close-settings' )
            actions.closeSettings();

        if ( action === 'settings-area' )
            showArea( target.dataset.area );

        if ( action === 'go-models' )
            showArea( 'models' );

        const holder = target.closest( '[data-setting]' );

        if ( holder === null )
            return;

        const setting = settingOf( holder.dataset.setting );

        if ( setting.onAction !== undefined )
            setting.onAction( action, target, view );
    }

    function onChange( event )
    {
        const holder = event.target.closest( '[data-setting]' );

        if ( holder === null )
            return;

        const setting = settingOf( holder.dataset.setting );

        if ( setting.onChange !== undefined )
            setting.onChange( event.target, view );
    }

    function onSearch()
    {
        local.query = search.value;
        refresh();
    }

    // Esc in a search with words in it clears them; the next one closes the window.
    function onSearchKey( event )
    {
        if ( event.key !== 'Escape' || search.value.length === 0 )
            return;

        event.preventDefault();
        event.stopPropagation();
        search.value = '';
        onSearch();
    }

    function onCancel( event )
    {
        event.preventDefault();
        actions.closeSettings();
    }

    root.addEventListener( 'click', onClick );
    root.addEventListener( 'change', onChange );
    root.addEventListener( 'cancel', onCancel );
    search.addEventListener( 'input', onSearch );
    search.addEventListener( 'keydown', onSearchKey );

    function shownGroups( state )
    {
        if ( !searching() )
        {
            const group = { area: areaOf( area ), settings: settingsInArea( area, state ) };

            return [ group ];
        }

        const found = searchSettings( local.query, state );
        const groups = [];

        for ( const shown of areas )
        {
            const inArea = found.filter( ( setting ) => setting.area === shown.id );

            if ( inArea.length === 0 )
                continue;

            const group = { area: shown, settings: inArea };
            groups.push( group );
        }

        return groups;
    }

    /** The parts of the state the shown settings read: a change anywhere else leaves a half-typed field alone. */
    function inputsOf( state, groups )
    {
        const names = new Set();

        for ( const group of groups )
        {
            const lead = leads[ group.area.id ];

            for ( const setting of group.settings )
                setting.reads.forEach( ( name ) => names.add( name ) );

            if ( lead !== undefined )
                lead.reads.forEach( ( name ) => names.add( name ) );
        }

        const read = [ ...names ].sort().map( ( name ) => state[ name ] );

        return [ area, local.query, local.confirmingRemove, groups.length, ...read ];
    }

    function sameInputs( inputs )
    {
        if ( lastInputs === null || lastInputs.length !== inputs.length )
            return false;

        return inputs.every( ( input, index ) => input === lastInputs[ index ] );
    }

    /** What goes under a setting's group: its note, then what it shows besides its control. */
    function footersOf( setting, state )
    {
        const note = setting.note( state );
        const shown = [];

        if ( note !== null )
            shown.push( element( 'p', 'setting-footer', note ) );

        if ( isReady( setting, state ) && setting.details !== undefined )
            shown.push( ...setting.details( state ) );

        return shown;
    }

    /** One line: the setting's name on the left, its control on the right. */
    function buildLine( setting, state )
    {
        const line = element( 'div', 'setting-line' );
        const label = element( 'span', 'setting-line-label', setting.label );
        const control = element( 'div', 'setting-line-control' );

        line.dataset.setting = setting.id;

        if ( isReady( setting, state ) )
            control.append( setting.build( state, view ) );

        line.append( label, control );

        return line;
    }

    /** A list setting, the installed models say: its name above its own group of rows. */
    function buildList( setting, state )
    {
        const header = element( 'h3', 'settings-list-header', setting.label );
        const shown = [ header ];
        const content = isReady( setting, state ) ? setting.build( state, view ) : null;

        if ( content !== null )
        {
            const card = element( 'div', 'settings-card' );
            card.dataset.setting = setting.id;
            card.append( content );
            shown.push( card );
        }

        shown.push( ...footersOf( setting, state ) );

        return shown;
    }

    function buildGroup( group, state )
    {
        const lead = leads[ group.area.id ];
        const waiting = group.settings.find( ( setting ) => !isReady( setting, state ) );

        if ( searching() )
            body.append( element( 'h3', 'settings-group', group.area.label ) );
        else if ( group.area.note !== null )
            body.append( element( 'p', 'setting-note settings-area-note', group.area.note ) );

        if ( lead !== undefined )
            body.append( ...lead.build( state ) );

        if ( waiting !== undefined )
        {
            const notice = element( 'div', 'settings-waiting' );
            notice.append( element( 'p', 'setting-note', waitingNotices[ waiting.needs ] ) );

            if ( waiting.needs === 'model' )
                notice.append( button( 'chip-button', 'Models', 'go-models' ) );

            body.append( notice );
        }

        // Lines share a group until one has words under it, as on an iPad.
        let card = null;

        for ( const setting of group.settings )
        {
            if ( setting.layout === 'list' )
            {
                body.append( ...buildList( setting, state ) );
                card = null;

                continue;
            }

            if ( card === null )
            {
                card = element( 'div', 'settings-card' );
                body.append( card );
            }

            card.append( buildLine( setting, state ) );

            const footers = footersOf( setting, state );

            if ( footers.length > 0 )
            {
                body.append( ...footers );
                card = null;
            }
        }
    }

    function build( state, groups )
    {
        clear( body );
        title.textContent = searching() ? 'Search' : areaOf( area ).label;

        if ( groups.length === 0 )
        {
            const empty = element( 'p', 'setting-note settings-empty', 'No setting matches "' + local.query.trim() + '".' );
            body.append( empty );

            return;
        }

        for ( const group of groups )
            buildGroup( group, state );
    }

    function markArea()
    {
        for ( const choice of areaList.children )
        {
            const current = !searching() && choice.dataset.area === area;

            choice.classList.toggle( 'selected', current );

            if ( current )
                choice.setAttribute( 'aria-current', 'page' );
            else
                choice.removeAttribute( 'aria-current' );
        }
    }

    function update( state )
    {
        const groups = shownGroups( state );
        const inputs = inputsOf( state, groups );

        markArea();

        if ( sameInputs( inputs ) )
            return;

        lastInputs = inputs;
        build( state, groups );
    }

    function opened()
    {
        const reset = { query: '', confirmingRemove: null };

        Object.assign( local, reset );
        search.value = '';
        lastInputs = null;
        open = true;
        root.showModal();
        search.focus();
    }

    function render( state, surface )
    {
        if ( !surface.settingsOpen )
        {
            if ( open )
                root.close();

            open = false;

            return;
        }

        area = surface.settingsArea;

        if ( !open )
            opened();

        update( state );
    }

    return { render };
}
