// A virtual clock for the mock session: scheduled calls run in time order, as fast as they can.

export function createClock( start )
{
    let now = start;
    let nextHandle = 1;
    const queue = [];

    function schedule( callback, milliseconds )
    {
        const handle = nextHandle;
        nextHandle = nextHandle + 1;

        const entry = { handle, at: now + milliseconds, callback };
        queue.push( entry );

        return handle;
    }

    function cancel( handle )
    {
        const index = queue.findIndex( ( entry ) => entry.handle === handle );

        if ( index >= 0 )
            queue.splice( index, 1 );
    }

    function earliestIndex()
    {
        let best = 0;

        for ( let index = 1; index < queue.length; index = index + 1 )
        {
            if ( queue[ index ].at < queue[ best ].at )
                best = index;
        }

        return best;
    }

    function step()
    {
        const index = earliestIndex();
        const entry = queue.splice( index, 1 )[ 0 ];
        now = entry.at;
        entry.callback();
    }

    /** Run scheduled calls until none remain, or until the predicate holds after one. */
    function run( until )
    {
        while ( queue.length > 0 )
        {
            step();

            if ( until !== undefined && until() )
                return;
        }
    }

    function time()
    {
        return now;
    }

    return { schedule, cancel, run, now: time };
}
