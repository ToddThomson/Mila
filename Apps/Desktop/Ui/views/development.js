// The mock's scenario menu. Shown only over the in-page mock; the window's bridge has none.

import { element, button } from '../dom.js';

export function createDevelopmentView( root, development, composer )
{
    function onChange( event )
    {
        const parameters = new URLSearchParams( window.location.search );
        parameters.set( 'scenario', event.target.value );
        window.location.search = parameters.toString();
    }

    const attachSettleMilliseconds = 400;

    function sendPrompt()
    {
        composer.play( development.prompt );
    }

    function attachSample( response )
    {
        const received = response.blob();

        received.then( addSample );
    }

    function addSample( blob )
    {
        const properties = { type: development.sample.type };
        const file = new File( [ blob ], development.sample.name, properties );
        const files = [ file ];

        composer.addFiles( files );
        window.setTimeout( sendPrompt, attachSettleMilliseconds );
    }

    /**
     * Plays the scenario as a person would: attach its sample file when it has one, then send.
     */
    function onPlay()
    {
        if ( development.sample === null )
        {
            sendPrompt();

            return;
        }

        const fetched = window.fetch( development.sampleUrl );

        fetched.then( attachSample );
    }

    function render()
    {
        root.hidden = development === null;

        if ( development === null )
            return;

        const label = element( 'label', 'development-label', 'Mock scenario' );
        const select = element( 'select', 'development-select' );
        const play = button( 'chip-button', 'Play', 'play-scenario' );

        for ( const scenario of development.scenarios )
        {
            const text = scenario.number + '. ' + scenario.title;
            const option = element( 'option', null, text );
            option.value = String( scenario.number );
            option.selected = scenario.number === development.scenario;
            select.append( option );
        }

        select.setAttribute( 'aria-label', 'Mock scenario' );
        select.addEventListener( 'change', onChange );
        play.addEventListener( 'click', onPlay );
        root.append( label, select, play );

        if ( development.autoplay )
            window.setTimeout( onPlay, 300 );
    }

    return { render };
}
