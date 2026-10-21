// The page's one way to the session: send( message ), onMessage( handler ) and pathForFile( file ).
// The window's bridge when the page is hosted (window.mila, from the preload script), and the
// in-page mock session otherwise. Nothing else in the page knows which it has.

function bridgeTransport( bridge )
{
    return {
        send: ( message ) => bridge.send( message ),
        onMessage: ( handler ) => bridge.onMessage( handler ),
        pathForFile: ( file ) => bridge.pathForFile( file ),
        development: null
    };
}

/**
 * Every message crosses as JSON, as it would over the bridge, so a message the real session
 * could not carry fails here too.
 */
function roundTrip( message )
{
    const text = JSON.stringify( message );

    return JSON.parse( text );
}

async function mockTransport( location )
{
    const module = await import( './mock/session.js' );
    const scenarioModule = await import( './mock/scenarios.js' );
    const parameters = new URLSearchParams( location.search );
    const requested = parameters.get( 'scenario' );
    const handlers = [];

    function deliver( message )
    {
        const copy = roundTrip( message );

        for ( const handler of handlers )
            handler( copy );
    }

    function schedule( callback, milliseconds )
    {
        return setTimeout( callback, milliseconds );
    }

    function cancel( handle )
    {
        clearTimeout( handle );
    }

    function now()
    {
        return Date.now();
    }

    const options = { scenario: requested === null ? 1 : requested, emit: deliver, schedule, cancel, now };
    const session = module.createMockSession( options );

    function send( message )
    {
        const copy = roundTrip( message );
        const handleLater = () => session.handle( copy );

        setTimeout( handleLater, 0 );
    }

    function onMessage( handler )
    {
        handlers.push( handler );
    }

    function pathForFile( file )
    {
        return '/mock/' + file.name;
    }

    const listing = scenarioModule.scenarios.map( ( scenario ) => ( { number: scenario.number, title: scenario.title } ) );

    const development = {
        scenarios: listing,
        scenario: session.scenario.number,
        prompt: session.scenario.prompt,
        autoplay: parameters.get( 'autoplay' ) !== null
    };

    return { send, onMessage, pathForFile, development };
}

export async function createTransport( location )
{
    if ( window.mila !== undefined )
        return bridgeTransport( window.mila );

    return mockTransport( location );
}
