// Get Things Done's plan, above the message box while the conversation has one: each step and where
// it stands, as the model marks it (MilaApp.md 3.1).

import { element, clear } from '../dom.js';
import { planProgress } from '../format.js';

const marks = { pending: '○', in_progress: '◐', done: '●', skipped: '–' };

export function createPlanView( root )
{
    const heading = element( 'div', 'plan-heading' );
    const title = element( 'span', 'plan-title', 'Plan' );
    const progress = element( 'span', 'plan-progress' );
    const list = element( 'ol', 'plan-steps' );
    let shownPlan = null;

    heading.append( title, progress );
    root.append( heading, list );

    function render( state )
    {
        const plan = state.session === null || state.session.plan === undefined ? null : state.session.plan;
        root.hidden = plan === null;

        if ( plan === null || JSON.stringify( plan ) === shownPlan )
            return;

        shownPlan = JSON.stringify( plan );
        progress.textContent = planProgress( plan );
        clear( list );

        for ( const step of plan )
        {
            const item = element( 'li', 'plan-step' );
            const mark = element( 'span', 'plan-mark', marks[ step.status ] === undefined ? marks.pending : marks[ step.status ] );
            const text = element( 'span', 'plan-text', step.text );

            item.classList.add( step.status );
            mark.setAttribute( 'aria-hidden', 'true' );
            item.append( mark, text );
            list.append( item );
        }
    }

    return { render };
}
