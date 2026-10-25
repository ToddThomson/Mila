// The models view, over the main pane: installed models to load or remove, and the published
// list to install from. A refusal shows in the view, in the session's own words.

import { element, button, iconButton, clear } from '../dom.js';
import { formatBytes } from '../format.js';

export function createModelsView( root, actions )
{
    const panelHeader = element( 'div', 'panel-header' );
    const title = element( 'h2', 'panel-title', 'Models' );
    const close = iconButton( 'quiet-icon', 'close', 'Close models', 'close-models' );
    const body = element( 'div', 'models-body' );
    const local = { confirmingRemove: null };
    let lastKey = '';

    panelHeader.append( title, close );
    root.append( panelHeader, body );

    function onClick( event )
    {
        const target = event.target.closest( '[data-action]' );

        if ( target === null )
            return;

        const action = target.dataset.action;
        const name = target.dataset.name;

        if ( action === 'close-models' )
        {
            const closed = { modelsOpen: false };
            actions.changeSurface( closed );
        }

        if ( action === 'load' )
        {
            const message = { type: 'model.load', name, quantization: null };
            actions.send( message );
        }

        if ( action === 'remove' )
            setConfirming( name );

        if ( action === 'remove-cancelled' )
            setConfirming( null );

        if ( action === 'remove-confirmed' )
        {
            const message = { type: 'model.remove', name };
            actions.send( message );
            setConfirming( null );
        }

        if ( action === 'install' )
        {
            const message = { type: 'model.install', name };
            actions.send( message );
        }

        if ( action === 'refresh' )
        {
            const message = { type: 'models.refresh', online: true };
            actions.send( message );
        }
    }

    function setConfirming( name )
    {
        local.confirmingRemove = name;
        lastKey = '';

        const state = actions.getState();
        const surface = { modelsOpen: true };
        render( state, surface );
    }

    root.addEventListener( 'click', onClick );

    function buildInstalledRow( model, current )
    {
        const row = element( 'div', 'model-row' );
        const text = element( 'div', 'model-text' );
        const name = element( 'div', 'model-name', model.name );
        const factText = formatBytes( model.size_bytes ) + ' · ' + model.quantization;
        const facts = element( 'div', 'model-facts', factText );
        const controls = element( 'div', 'model-controls' );

        text.append( name, facts );
        row.append( text, controls );

        if ( local.confirmingRemove === model.name )
        {
            const question = element( 'span', 'confirm-question', 'Remove it from this computer?' );
            const confirm = button( 'confirm-delete', 'Remove', 'remove-confirmed' );
            const cancel = button( 'confirm-cancel', 'Cancel', 'remove-cancelled' );
            confirm.dataset.name = model.name;
            controls.append( question, confirm, cancel );

            return row;
        }

        if ( model.name === current )
        {
            const loaded = element( 'span', 'model-badge', 'Loaded' );
            controls.append( loaded );
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

        return row;
    }

    function buildPublishedRow( model )
    {
        const row = element( 'div', 'model-row' );
        const text = element( 'div', 'model-text' );
        const name = element( 'div', 'model-name', model.name );
        const factText = formatBytes( model.size_bytes ) + ' · ' + model.quantization;
        const facts = element( 'div', 'model-facts', factText );
        const controls = element( 'div', 'model-controls' );

        text.append( name, facts );
        row.append( text, controls );

        if ( model.installed )
        {
            const installed = element( 'span', 'model-badge', 'Installed' );
            controls.append( installed );

            return row;
        }

        const install = button( 'chip-button', 'Install', 'install' );
        install.dataset.name = model.name;
        controls.append( install );

        return row;
    }

    function buildProgress( progress )
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

    function build( state )
    {
        clear( body );

        const current = state.session === null ? null : state.session.model;
        const progress = state.progress;

        if ( progress !== null && ( progress.total === null || progress.done < progress.total ) )
        {
            const shown = buildProgress( progress );
            body.append( shown );
        }

        const installedHeading = element( 'h3', 'models-heading', 'Installed' );
        body.append( installedHeading );

        for ( const model of state.installedModels )
        {
            const row = buildInstalledRow( model, current );
            body.append( row );
        }

        const publishedHeading = element( 'h3', 'models-heading', 'Available to install' );
        body.append( publishedHeading );

        if ( state.publishedModels === null )
        {
            const note = element( 'p', 'models-note', 'See the models published for Mila. This needs an internet connection.' );
            const refresh = button( 'chip-button', 'Show available models', 'refresh' );
            body.append( note, refresh );

            return;
        }

        for ( const model of state.publishedModels )
        {
            const row = buildPublishedRow( model );
            body.append( row );
        }
    }

    function render( state, surface )
    {
        root.hidden = !surface.modelsOpen;

        if ( !surface.modelsOpen )
            return;

        const key = JSON.stringify( [ state.installedModels, state.publishedModels, state.progress, state.session === null ? null : state.session.model, local ] );

        if ( key === lastKey )
            return;

        lastKey = key;
        build( state );
    }

    return { render };
}
