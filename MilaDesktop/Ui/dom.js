// Element construction for the views. Text goes in through textContent; the only HTML set from a
// string is markdown.js's escaped output and the fixed icons below.

export function element( tag, className, text )
{
    const created = document.createElement( tag );

    if ( className !== undefined && className !== null && className.length > 0 )
        created.className = className;

    if ( text !== undefined && text !== null )
        created.textContent = text;

    return created;
}

export function button( className, label, action )
{
    const created = element( 'button', className, label );
    created.type = 'button';

    if ( action !== undefined && action !== null )
        created.dataset.action = action;

    return created;
}

export function iconButton( className, icon, label, action )
{
    const created = button( className, null, action );
    created.innerHTML = icons[ icon ];
    created.setAttribute( 'aria-label', label );
    created.title = label;

    return created;
}

export function setIcon( target, icon )
{
    target.innerHTML = icons[ icon ];
}

/**
 * An attached image's thumbnail while the page still has the file, and the kind's icon otherwise:
 * a conversation opened from the store carries the name only.
 */
export function previewOrIcon( attachment, icon )
{
    if ( attachment.previewUrl === undefined || attachment.previewUrl === null )
        return icon;

    const thumbnail = element( 'img', 'attachment-thumbnail' );
    thumbnail.src = attachment.previewUrl;
    thumbnail.alt = '';

    return thumbnail;
}

export function clear( target )
{
    target.replaceChildren();
}

function svg( body )
{
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + body + '</svg>';
}

export const icons = {
    plus: svg( '<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>' ),
    sidebar: svg( '<rect x="3" y="4" width="18" height="16" rx="2.5"/><line x1="9" y1="4" x2="9" y2="20"/>' ),
    chevronDown: svg( '<polyline points="6 9 12 15 18 9"/>' ),
    chevronRight: svg( '<polyline points="9 6 15 12 9 18"/>' ),
    models: svg( '<rect x="4" y="4" width="16" height="6" rx="1.5"/><rect x="4" y="14" width="16" height="6" rx="1.5"/><line x1="8" y1="7" x2="8.01" y2="7"/><line x1="8" y1="17" x2="8.01" y2="17"/>' ),
    settings: svg( '<line x1="4" y1="7" x2="14" y2="7"/><line x1="18" y1="7" x2="20" y2="7"/><circle cx="16" cy="7" r="2"/><line x1="4" y1="17" x2="6" y2="17"/><line x1="10" y1="17" x2="20" y2="17"/><circle cx="8" cy="17" r="2"/>' ),
    more: svg( '<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>' ),
    attach: svg( '<path d="M21 11.5l-8.6 8.6a5 5 0 0 1-7.1-7.1l8.6-8.6a3.3 3.3 0 0 1 4.7 4.7l-8.6 8.6a1.7 1.7 0 0 1-2.4-2.4l7.9-7.9"/>' ),
    send: svg( '<line x1="12" y1="19" x2="12" y2="5"/><polyline points="6 11 12 5 18 11"/>' ),
    stop: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor"/></svg>',
    copy: svg( '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/>' ),
    check: svg( '<polyline points="20 6 9 17 4 12"/>' ),
    tool: svg( '<path d="M14.7 6.3a4 4 0 0 0-5.4 5.2L3.6 17.2a1.4 1.4 0 0 0 2 2l5.7-5.7a4 4 0 0 0 5.2-5.4l-2.5 2.5-2.1-.4-.4-2.1z"/>' ),
    close: svg( '<line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/>' ),
    image: svg( '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.75"/><polyline points="21 16 15.5 11 5 20"/>' ),
    audio: svg( '<path d="M9 18V6l10-2v12"/><circle cx="6.5" cy="18" r="2.5"/><circle cx="16.5" cy="16" r="2.5"/>' )
};
