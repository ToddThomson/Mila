// The sheet that makes a project (AssistantInterface.md 4.1, until phase 6): a name, what it is for, and Create. The
// session makes it and the page opens it.

import { element, button, clear } from '../dom.js';

export function createNewProjectView( dialog, actions )
{
    const name = element( 'input', 'sheet-input' );
    const purpose = element( 'textarea', 'sheet-input sheet-purpose' );
    const create = button( 'setup-button primary', 'Create', 'project-create' );

    name.type = 'text';
    name.autocomplete = 'off';
    name.placeholder = 'Kyoto in November';
    name.setAttribute( 'aria-label', 'Name' );
    purpose.rows = 3;
    purpose.placeholder = 'What it is for, in a line or two. You can leave this for later.';
    purpose.setAttribute( 'aria-label', 'What it is for' );

    function canCreate()
    {
        return name.value.trim().length > 0;
    }

    function submit()
    {
        if ( !canCreate() )
            return;

        actions.send( { type: 'project.new', name: name.value.trim(), purpose: purpose.value.trim() } );
        dialog.close();
    }

    function onClick( event )
    {
        if ( event.target === dialog || event.target.closest( '[data-action="project-cancel"]' ) !== null )
        {
            dialog.close();

            return;
        }

        if ( event.target.closest( '[data-action="project-create"]' ) !== null )
            submit();
    }

    function onKeyDown( event )
    {
        if ( event.key === 'Enter' && event.target === name )
        {
            event.preventDefault();
            submit();
        }
    }

    name.addEventListener( 'input', () => { create.disabled = !canCreate(); } );
    dialog.addEventListener( 'click', onClick );
    dialog.addEventListener( 'keydown', onKeyDown );

    function open()
    {
        if ( dialog.open )
            return;

        const foot = element( 'div', 'license-foot' );
        const nameLabel = element( 'label', 'sheet-label', 'Name' );
        const purposeLabel = element( 'label', 'sheet-label', 'What it is for' );

        name.value = '';
        purpose.value = '';
        create.disabled = true;
        nameLabel.append( name );
        purposeLabel.append( purpose );
        foot.append( button( 'setup-button', 'Cancel', 'project-cancel' ), create );

        clear( dialog );
        dialog.append( element( 'h2', 'license-title', 'New project' ), nameLabel, purposeLabel, foot );
        dialog.showModal();
        name.focus();
    }

    return { open, isOpen: () => dialog.open };
}
