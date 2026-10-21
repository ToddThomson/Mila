// The composer: a growing text box, attachment chips above it, and Send that becomes Stop while a
// reply is written. A file dropped anywhere on the main pane attaches.

import { element, iconButton, setIcon, clear } from '../dom.js';
import { isBusy, streamIdentifier } from '../state.js';

const maximumRows = 12;

const imageTypes = [ 'image/png', 'image/jpeg' ];
const audioTypes = [ 'audio/wav', 'audio/x-wav', 'audio/wave' ];

export function createComposerView( root, dropZone, actions )
{
    const form = element( 'form', 'composer' );
    const chips = element( 'div', 'attachment-chips' );
    const input = element( 'textarea', 'composer-input' );
    const row = element( 'div', 'composer-row' );
    const attach = iconButton( 'quiet-icon attach-button', 'attach', 'Attach a file', 'attach' );
    const hint = element( 'span', 'composer-hint' );
    const submit = iconButton( 'send-button', 'send', 'Send', 'submit' );
    const picker = element( 'input', 'file-picker' );
    const dropOverlay = element( 'div', 'drop-overlay' );
    const dropLabel = element( 'span', 'drop-label', 'Drop to attach' );
    let nextAttachmentNumber = 1;
    let lastAttachments = null;
    let attachRefusal = null;

    input.rows = 1;
    input.setAttribute( 'aria-label', 'Message' );
    picker.type = 'file';
    picker.multiple = true;
    picker.hidden = true;
    dropOverlay.hidden = true;
    dropOverlay.append( dropLabel );
    row.append( attach, hint, submit );
    form.append( chips, input, row );
    root.append( form, picker );
    dropZone.append( dropOverlay );

    function resize()
    {
        input.style.height = 'auto';

        const style = window.getComputedStyle( input );
        const lineHeight = parseFloat( style.lineHeight );
        const limit = lineHeight * maximumRows;
        const height = Math.min( input.scrollHeight, limit );

        input.style.height = height + 'px';
    }

    function submitTurn()
    {
        const state = actions.getState();
        const text = input.value.trim();

        if ( isBusy( state ) || text.length === 0 || state.session === null || state.session.model === null )
            return;

        const attachmentIds = state.attachments.filter( ( attachment ) => attachment.status !== 'refused' ).map( ( attachment ) => attachment.id );
        const stream = streamIdentifier( state );
        const message = { type: 'turn.send', stream_id: stream, text, attachment_ids: attachmentIds };

        actions.send( message );
        input.value = '';
        resize();
    }

    function stopTurn()
    {
        const state = actions.getState();
        const stream = streamIdentifier( state );
        const message = { type: 'turn.stop', stream_id: stream };

        actions.send( message );
    }

    function onSubmit( event )
    {
        event.preventDefault();

        const state = actions.getState();

        if ( isBusy( state ) )
        {
            stopTurn();

            return;
        }

        submitTurn();
    }

    function onKeyDown( event )
    {
        if ( event.key !== 'Enter' || event.shiftKey || event.isComposing )
            return;

        event.preventDefault();
        submitTurn();
    }

    function kindOf( file )
    {
        if ( imageTypes.includes( file.type ) )
            return 'image';

        if ( audioTypes.includes( file.type ) )
            return 'audio';

        return null;
    }

    function addFile( file )
    {
        const identifier = 'attachment-' + nextAttachmentNumber;
        nextAttachmentNumber = nextAttachmentNumber + 1;

        const kind = kindOf( file );
        const path = actions.pathForFile( file );
        const request = { type: 'attachment.add', attachment_id: identifier, kind, name: file.name, path };
        const reason = localRefusal( kind, path );

        if ( reason === null )
        {
            actions.send( request );

            return;
        }

        const refusal = { type: 'attachment.refused', attachment_id: identifier, reason };

        actions.dispatch( request );
        actions.dispatch( refusal );
    }

    /**
     * What the page can refuse without asking: a file of no kind a model reads, or one that is
     * not on this computer, such as a picture dragged out of a web page.
     */
    function localRefusal( kind, path )
    {
        if ( kind === null )
            return 'Only PNG and JPEG images and WAV audio can be attached.';

        if ( path === null )
            return 'Only files saved on this computer can be attached.';

        return null;
    }

    function addFiles( files )
    {
        if ( attachRefusal !== null )
            return;

        for ( const file of files )
            addFile( file );
    }

    function onAttachClick()
    {
        if ( attachRefusal !== null )
            return;

        picker.click();
    }

    function onPickerChange()
    {
        const files = Array.from( picker.files );

        addFiles( files );
        picker.value = '';
    }

    function onChipClick( event )
    {
        const target = event.target.closest( '[data-action="remove-attachment"]' );

        if ( target === null )
            return;

        const message = { type: 'attachment.remove', attachment_id: target.dataset.identifier };
        actions.send( message );
    }

    function hasFiles( event )
    {
        return event.dataTransfer !== null && event.dataTransfer.types.includes( 'Files' );
    }

    function onDragOver( event )
    {
        if ( !hasFiles( event ) )
            return;

        event.preventDefault();
        dropOverlay.hidden = false;
        dropOverlay.classList.toggle( 'refused', attachRefusal !== null );
        dropLabel.textContent = attachRefusal === null ? 'Drop to attach' : attachRefusal;
    }

    function onDragLeave( event )
    {
        if ( event.relatedTarget === null || !dropZone.contains( event.relatedTarget ) )
            dropOverlay.hidden = true;
    }

    function onDrop( event )
    {
        if ( !hasFiles( event ) )
            return;

        event.preventDefault();
        dropOverlay.hidden = true;

        const files = Array.from( event.dataTransfer.files );
        addFiles( files );
    }

    form.addEventListener( 'submit', onSubmit );
    input.addEventListener( 'keydown', onKeyDown );
    input.addEventListener( 'input', resize );
    attach.addEventListener( 'click', onAttachClick );
    picker.addEventListener( 'change', onPickerChange );
    chips.addEventListener( 'click', onChipClick );
    dropZone.addEventListener( 'dragover', onDragOver );
    dropZone.addEventListener( 'dragleave', onDragLeave );
    dropZone.addEventListener( 'drop', onDrop );
    submit.type = 'submit';

    function buildChip( attachment )
    {
        const chip = element( 'span', 'attachment-chip' );
        const icon = element( 'span', 'attachment-icon' );
        const name = element( 'span', 'attachment-name', attachment.name );
        const removeLabel = 'Remove ' + attachment.name;
        const remove = iconButton( 'quiet-icon chip-remove', 'close', removeLabel, 'remove-attachment' );

        chip.classList.toggle( 'refused', attachment.status === 'refused' );
        chip.classList.toggle( 'pending', attachment.status === 'pending' );
        chip.title = attachment.status === 'refused' ? attachment.reason : attachment.name;
        remove.dataset.identifier = attachment.id;
        setIcon( icon, attachment.kind === 'audio' ? 'audio' : 'image' );
        chip.append( icon, name, remove );

        return chip;
    }

    function renderChips( state )
    {
        if ( state.attachments === lastAttachments )
            return;

        lastAttachments = state.attachments;
        clear( chips );

        for ( const attachment of state.attachments )
        {
            const chip = buildChip( attachment );
            chips.append( chip );
        }

        chips.hidden = state.attachments.length === 0;
    }

    /**
     * Why attach is off for this model, in the reader's words, or null when it is on.
     */
    function refusalFor( session )
    {
        if ( session === null || session.model === null )
            return 'Load a model to attach files.';

        if ( session.modalities.length === 0 )
            return session.model + ' reads text only, so it cannot take images or audio.';

        return null;
    }

    function acceptFor( session )
    {
        const accepted = [];

        if ( session !== null && session.modalities.includes( 'image' ) )
            accepted.push( ...imageTypes );

        if ( session !== null && session.modalities.includes( 'audio' ) )
            accepted.push( ...audioTypes );

        return accepted.join( ',' );
    }

    function render( state )
    {
        const busy = isBusy( state );
        const session = state.session;
        const loaded = session !== null && session.model !== null;

        attachRefusal = refusalFor( session );
        attach.classList.toggle( 'disabled', attachRefusal !== null );
        attach.setAttribute( 'aria-disabled', attachRefusal === null ? 'false' : 'true' );
        attach.title = attachRefusal === null ? 'Attach an image or audio clip' : attachRefusal;
        picker.accept = acceptFor( session );

        input.placeholder = loaded ? 'Message ' + session.model : 'Load a model to start a conversation';
        input.disabled = !loaded;

        setIcon( submit, busy ? 'stop' : 'send' );
        submit.classList.toggle( 'stop', busy );
        submit.setAttribute( 'aria-label', busy ? 'Stop' : 'Send' );
        submit.title = busy ? 'Stop (Esc)' : 'Send (Enter)';
        hint.textContent = busy ? 'Esc stops the reply' : 'Shift+Enter for a new line';

        renderChips( state );
    }

    function focus()
    {
        input.focus();
    }

    /**
     * Type a message and send it, as a person would. Used by the mock's scenario menu.
     */
    function play( text )
    {
        input.value = text;
        resize();
        submitTurn();
    }

    return { render, focus, play, addFiles };
}
