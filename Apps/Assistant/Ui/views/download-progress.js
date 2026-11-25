// A model's download on its card, in the first-run setup and in Settings' Models: the bar, how much of it has come,
// and Cancel, which stops it (MilaAssistant.md 4.2, model.install_stop).

import { element, button } from '../dom.js';
import { installAmount } from '../format.js';

/**
 * The download a progress message describes, its bar moving with no amount until the total is known; Cancel carries
 * the action `stop-install`.
 */
export function downloadProgress( progress )
{
    const shown = element( 'div', 'install-progress' );
    const bar = element( 'progress', 'progress-bar' );
    const line = element( 'div', 'install-line' );
    const amount = element( 'span', 'install-amount', progress.total === null ? '' : installAmount( progress ) );
    const cancel = button( 'chip-button install-cancel', 'Cancel', 'stop-install' );

    bar.setAttribute( 'aria-label', 'Download' );

    // A bar with no value is one that moves without saying how far.
    if ( progress.total !== null )
    {
        bar.max = progress.total;
        bar.value = progress.done;
    }

    line.append( amount, cancel );
    shown.append( bar, line );

    return shown;
}
