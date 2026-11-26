// All projects and a project (AssistantInterface.md 4.4, 4.5), drawn into the places' pane from the session's records.
// The places view owns the pane, its clicks and the rows' menus; this draws what a project's places show.

import { element, button, iconButton } from '../dom.js';
import { matchingProjects, projectFilters, projectSummaries, currentProject, madeInProject, projectNextAction } from '../state.js';
import { whenLabel, folderName, activityText, dayLabel, clockLabel, plainLine } from '../format.js';
import { head, section, figure, stateDot, stateWords, pulseStrip, countsLine } from './place-parts.js';

const filterWords = { all: 'All', waiting: 'Waiting on you', active: 'Active', quiet: 'Quiet' };
const sortWords = { recent: 'Most recent', name: 'By name' };

const diskNotes = { changed: 'Changed on disk since', missing: 'No longer on disk' };

/**
 * One thing a project's chats made (4.5, 4.9): a file by its name, or a reply kept by how it begins, with its chat and
 * when; it opens its drawer.
 */
export function madeRow( made, now )
{
    const row = button( 'row', null, 'open-creation' );
    const main = element( 'span', 'row-main' );
    const file = made.kind === 'file';
    const title = file ? folderName( made.path ) : plainLine( made.start ?? '' ) || 'A reply you kept';
    const parts = [ file ? ( made.tool === 'edit_file' ? 'Changed by Mila' : 'Written by Mila' ) : 'A reply you kept', 'in ' + made.item.title ];
    const note = file ? diskNotes[ made.disk ] ?? null : null;

    if ( note !== null )
        parts.push( note );

    row.dataset.identifier = made.item.id;

    if ( file )
        row.dataset.path = made.path;
    else
        row.dataset.message = String( made.message );

    if ( file )
        row.title = made.path;

    main.append( element( 'span', 'row-title', title ), element( 'span', note === null ? 'row-sub' : 'row-sub made-changed', parts.join( ' · ' ) ) );
    row.append( main, element( 'span', 'row-when', whenLabel( made.time ?? made.item.updated, now ) ) );

    return row;
}

/** The latest thing that happened in a project, as one line, or when it was made. */
function latestLine( summary, now )
{
    if ( summary.latest !== null )
        return activityText( projectEvent( summary.latest, summary.project ) ) + ' · ' + whenLabel( summary.latest.time, now );

    return 'Made ' + whenLabel( summary.project.created, now );
}

/** One of the session's events as Activity's entries are, its projects named as they are now. */
function projectEvent( recorded, project )
{
    const name = ( id ) => ( id === project.id ? project.name : null );

    return { kind: 'project', recorded, to: name( recorded.project ), from: name( recorded.from ) };
}

function projectRow( summary, now )
{
    const row = button( 'row project-row', null, 'open-project' );
    const main = element( 'span', 'row-main' );
    const purpose = summary.project.purpose ? summary.project.purpose : 'No purpose written';

    row.dataset.project = summary.project.id;
    main.append( element( 'span', 'row-title', summary.project.name ),
        element( 'span', 'row-sub', purpose + ' · ' + latestLine( summary, now ) ) );
    row.append( stateDot( summary.state ), main, element( 'span', 'row-when', countsLine( summary ) ) );

    return row;
}

/** A project as the overview's card (4.3): name, purpose, state, its last fourteen days, its latest event and counts. */
export function projectCard( summary, now )
{
    const card = button( 'project-card', null, 'open-project' );
    const top = element( 'span', 'project-card-top' );

    card.dataset.project = summary.project.id;
    top.append( stateDot( summary.state ), element( 'span', 'project-card-name', summary.project.name ),
        element( 'span', 'project-card-state', stateWords[ summary.state ] ) );
    card.append( top, element( 'span', 'project-card-purpose', summary.project.purpose ? summary.project.purpose : 'No purpose written' ),
        pulseStrip( summary.pulse ), element( 'span', 'project-card-latest', latestLine( summary, now ) ),
        element( 'span', 'project-card-counts', countsLine( summary ) ) );

    return card;
}

/** All projects (4.4): a search, a filter, a sort, and a row each. */
export function renderProjects( root, state, local, now, search )
{
    const tools = element( 'div', 'tools' );

    if ( search.value !== local.projectQuery )
        search.value = local.projectQuery;

    tools.append( search );

    for ( const filter of projectFilters )
    {
        const chip = button( 'filter', filterWords[ filter ], 'project-filter' );
        chip.dataset.filter = filter;
        chip.setAttribute( 'aria-pressed', local.projectFilter === filter ? 'true' : 'false' );
        tools.append( chip );
    }

    for ( const sort of Object.keys( sortWords ) )
    {
        const chip = button( 'filter', sortWords[ sort ], 'project-sort' );
        chip.dataset.sort = sort;
        chip.setAttribute( 'aria-pressed', local.projectSort === sort ? 'true' : 'false' );
        tools.append( chip );
    }

    const create = button( 'button', 'New project', 'new-project' );
    const top = head( 'Workspace', 'Projects' );
    const list = element( 'div', 'rows' );
    const shown = matchingProjects( state, local.projectQuery, local.projectFilter, local.projectSort );

    top.append( create );

    if ( shown.length === 0 )
        list.append( element( 'div', 'empty', state.projects.length === 0 ? 'No projects yet.' : 'No project matches.' ) );
    else
        list.append( ...shown.map( ( summary ) => projectRow( summary, now ) ) );

    root.append( top, tools, list );
}

function editor( kind, value, label, multiline )
{
    const input = element( multiline ? 'textarea' : 'input', 'rename-input project-edit' );

    input.value = value;
    input.dataset.edit = kind;
    input.setAttribute( 'aria-label', label );

    if ( multiline )
        input.rows = 2;

    return input;
}

function projectMenu( project )
{
    const menu = element( 'div', 'row-menu menu project-menu' );

    for ( const [ label, action ] of [ [ 'Rename', 'project-rename' ], [ 'Change what it is for', 'project-purpose' ],
        [ 'Add a folder', 'project-add-folder' ] ] )
    {
        const item = button( 'menu-item', label, action );
        item.dataset.project = project.id;
        menu.append( item );
    }

    const remove = button( 'menu-item danger', 'Delete project', 'project-delete' );
    remove.dataset.project = project.id;
    menu.append( remove );

    return menu;
}

/** Deleting asks (11, carried 6): its chats kept in no project by default, its folders never touched. */
function deleteQuestion( project )
{
    const holder = element( 'div', 'notice-card project-confirm' );
    const choices = element( 'div', 'notice-actions' );
    const keep = button( 'button', 'Delete, keep its chats', 'project-delete-keep' );
    const all = button( 'button danger', 'Delete it and its chats', 'project-delete-all' );

    keep.dataset.project = project.id;
    all.dataset.project = project.id;
    choices.append( keep, all, button( 'button quiet', 'Cancel', 'project-delete-cancel' ) );
    holder.append( element( 'div', 'notice-title', 'Delete “' + project.name + '”?' ),
        element( 'div', 'notice-text', 'Its folders and the files in them are not touched. Kept chats move to no project.' ),
        choices );

    return holder;
}

function folderRow( project, folder )
{
    const row = element( 'div', 'reach-row' );
    const main = element( 'div', 'reach-main' );
    const access = element( 'select', 'folder-access' );
    const remove = button( 'button quiet', 'Remove', 'project-remove-folder' );

    for ( const [ value, label ] of [ [ 'read', 'Read' ], [ 'read_write', 'Read and change' ] ] )
    {
        const option = element( 'option', null, label );
        option.value = value;
        option.selected = folder.access === value;
        access.append( option );
    }

    access.dataset.project = project.id;
    access.dataset.path = folder.path;
    access.setAttribute( 'aria-label', 'What Mila may do in ' + folder.path );
    remove.dataset.project = project.id;
    remove.dataset.path = folder.path;
    remove.setAttribute( 'aria-label', 'Remove ' + folder.path + ' from ' + project.name );
    main.append( element( 'div', 'reach-title', folderName( folder.path ) ), element( 'div', 'row-sub reach-note', folder.path ) );
    main.title = folder.path;
    row.append( main, access, remove );

    return row;
}

/** Where a note came from: the chat it was written in, by its title as it is now, and when; and whether it was edited. */
function noteSource( state, note, now )
{
    const chatId = note.source?.ref?.conversation ?? null;
    const chat = state.conversations.find( ( item ) => item.id === chatId );
    const parts = [ chat === undefined ? 'From a chat no longer stored' : 'From “' + chat.title + '”', whenLabel( note.made, now ) ];

    if ( typeof note.edited === 'string' )
        parts.push( 'edited by you' );

    return parts.join( ' · ' );
}

function noteQuestions( note )
{
    const questions = Array.isArray( note.questions ) ? note.questions : [];

    if ( questions.length === 0 )
        return null;

    const list = element( 'ul', 'note-questions' );
    list.append( ...questions.map( ( question ) => element( 'li', null, question ) ) );

    return list;
}

/** The person's own edit of the note: its summary, its questions one a line, and its next step. */
function noteEditor( note )
{
    const form = element( 'div', 'note-editor' );
    const summary = element( 'textarea', 'project-edit note-field' );
    const questions = element( 'textarea', 'project-edit note-field' );
    const next = element( 'input', 'project-edit note-field' );
    const choices = element( 'div', 'notice-actions' );

    summary.value = note?.summary ?? '';
    summary.rows = 4;
    summary.dataset.field = 'summary';
    summary.setAttribute( 'aria-label', 'What Mila understands' );
    questions.value = Array.isArray( note?.questions ) ? note.questions.join( '\n' ) : '';
    questions.rows = 3;
    questions.dataset.field = 'questions';
    questions.setAttribute( 'aria-label', 'Open questions, one a line' );
    next.value = note?.next ?? '';
    next.dataset.field = 'next';
    next.setAttribute( 'aria-label', 'Next' );
    choices.append( button( 'button primary', 'Save', 'understanding-save' ), button( 'button quiet', 'Cancel', 'understanding-cancel' ) );
    form.append( summary, element( 'label', 'note-label', 'Open questions' ), questions,
        element( 'label', 'note-label', 'Next' ), next, choices );

    return form;
}

/**
 * The summary (4.5): *What Mila understands* beside *Next*, the one action beside it from the app's records; a note
 * waiting for the person in full, with where it came from (5.6); and the person's instructions for the project's chats.
 */
function summaryOf( state, local, project, now )
{
    const holder = section( 'Summary', 'project-summary', null );
    const note = project.understanding ?? null;
    const waiting = project.waiting ?? null;

    if ( waiting !== null )
    {
        const card = element( 'div', 'notice-card waiting note-waiting' );
        const choices = element( 'div', 'notice-actions' );
        const accept = button( 'button primary', 'Accept', 'note-accept' );
        const decline = button( 'button quiet', 'Decline', 'note-decline' );

        for ( const choice of [ accept, decline ] )
        {
            choice.dataset.project = project.id;
            choice.dataset.note = waiting.id;
        }

        choices.append( accept, decline );
        card.append( element( 'div', 'notice-kind', 'Waiting for you' ),
            element( 'div', 'notice-title', 'A note written in a chat that read from outside' ),
            element( 'p', 'note-summary', waiting.summary ?? '' ) );

        const questions = noteQuestions( waiting );

        if ( questions !== null )
            card.append( questions );

        if ( waiting.next )
            card.append( element( 'p', 'note-next', 'Next: ' + waiting.next ) );

        card.append( element( 'div', 'notice-text', noteSource( state, waiting, now ) ), choices );
        holder.append( card );
    }

    const columns = element( 'div', 'project-summary-columns' );
    const understood = element( 'div', 'summary-panel' );
    const next = element( 'div', 'summary-panel' );
    const understoodHead = element( 'div', 'summary-head' );

    understoodHead.append( element( 'h3', 'summary-title', 'What Mila understands' ) );

    if ( note !== null && local.editing !== 'understanding' )
    {
        const edit = button( 'link', 'Edit', 'understanding-edit' );
        const remove = button( 'link', 'Clear', 'understanding-clear' );

        remove.dataset.project = project.id;
        understoodHead.append( edit, remove );
    }

    understood.append( understoodHead );

    if ( local.editing === 'understanding' && note !== null )
        understood.append( noteEditor( note ) );
    else if ( note === null )
        understood.append( element( 'p', 'empty', 'Nothing noted yet.' ) );
    else
    {
        understood.append( element( 'p', 'note-summary', note.summary ?? '' ) );

        const questions = noteQuestions( note );

        if ( questions !== null )
            understood.append( element( 'div', 'note-label', 'Open questions' ), questions );

        understood.append( element( 'div', 'row-sub', noteSource( state, note, now ) ) );
    }

    next.append( element( 'h3', 'summary-title', 'Next' ) );
    next.append( element( 'p', note?.next ? 'note-next' : 'empty', note?.next ? note.next : 'Nothing noted yet.' ) );

    const action = projectNextAction( state, project.id );

    if ( action !== null )
    {
        const act = button( 'button', action.label, action.action );
        act.dataset.identifier = action.identifier;
        next.append( act );
    }

    columns.append( understood, next );
    holder.append( columns );

    const instructionsHead = element( 'div', 'summary-head' );
    const instructions = element( 'div', 'summary-panel project-instructions' );

    instructionsHead.append( element( 'h3', 'summary-title', 'Instructions' ) );

    if ( local.editing !== 'instructions' )
        instructionsHead.append( button( 'link', project.instructions ? 'Edit' : 'Write instructions', 'project-instructions' ) );

    instructions.append( instructionsHead );

    if ( local.editing === 'instructions' )
        instructions.append( editor( 'instructions', project.instructions ?? '', 'Instructions for Mila in this project', true ) );
    else
        instructions.append( element( 'p', project.instructions ? 'project-instructions-text' : 'empty',
            project.instructions ? project.instructions : 'None yet.' ) );

    holder.append( instructions );

    return holder;
}

/** A project (4.5): its head, its figures, its summary, its folders, its chats and its history. */
export function renderProject( root, state, local, now, managedRow )
{
    const project = currentProject( state );

    if ( project === null )
        return;

    const summary = projectSummaries( state, now ).find( ( candidate ) => candidate.project.id === project.id );
    const top = element( 'div', 'place-head project-head' );
    const titleRow = element( 'div', 'project-title-row' );
    const menuButton = iconButton( 'row-menu-button quiet-icon', 'more', 'Project options', 'project-menu' );

    if ( local.editing === 'name' )
        titleRow.append( editor( 'name', project.name, 'Project name', false ) );
    else
        titleRow.append( element( 'h1', 'place-title', project.name ) );

    titleRow.append( menuButton );
    top.append( element( 'div', 'eyebrow', 'Project' ), titleRow );

    if ( local.projectMenu )
        top.append( projectMenu( project ) );

    if ( local.editing === 'purpose' )
        top.append( editor( 'purpose', project.purpose ?? '', 'What the project is for', true ) );
    else
        top.append( element( 'p', 'project-purpose', project.purpose ? project.purpose : 'No purpose written yet.' ) );

    const facts = element( 'div', 'project-facts' );
    facts.append( stateDot( summary.state ), element( 'span', null, stateWords[ summary.state ] ),
        element( 'span', 'project-fact-separator', '·' ), element( 'span', null, latestLine( summary, now ) ) );
    top.append( facts );
    root.append( top );

    if ( local.confirmingProject === project.id )
        root.append( deleteQuestion( project ) );

    const glance = element( 'div', 'glance' );
    const newest = summary.chats.length === 0 ? 'None yet' : 'Newest: ' + summary.chats[ 0 ].title;

    glance.append(
        figure( summary.chats.length, 'Chats', newest, { action: 'show-section', data: { section: 'project-chats' } }, null ),
        figure( summary.made, 'Made', summary.made === 1 ? 'One thing made' : 'Files and replies kept',
            summary.made === 0 ? { action: null, data: {} } : { action: 'show-section', data: { section: 'project-made' } }, summary.made === 0 ? 'still' : null ),
        figure( summary.plans, 'Plans', summary.plans === 1 ? 'In one chat' : 'In its chats', { action: null, data: {} }, 'still' ) );

    const atGlance = section( 'At a glance', null, null );
    atGlance.append( glance );
    root.append( atGlance, summaryOf( state, local, project, now ) );

    const addFolder = button( 'link', 'Add a folder', 'project-add-folder' );
    addFolder.dataset.project = project.id;

    const folders = section( 'Folders', 'project-folders', addFolder );
    const folderRows = element( 'div', 'rows' );
    const given = Array.isArray( project.folders ) ? project.folders : [];

    if ( given.length === 0 )
        folderRows.append( element( 'div', 'empty', 'No folder. A chat here asks for one when it needs files.' ) );

    for ( const folder of given )
        folderRows.append( folderRow( project, folder ) );

    folders.append( folderRows );
    root.append( folders );

    const chats = section( 'Chats', 'project-chats', null );
    const chatRows = element( 'div', 'rows' );

    if ( summary.chats.length === 0 )
        chatRows.append( element( 'div', 'empty', 'No chats yet. Ask Mila below to start one here.' ) );
    else
        chatRows.append( ...summary.chats.map( ( item ) => managedRow( item, now, null ) ) );

    chats.append( chatRows );
    root.append( chats );

    const made = madeInProject( state, project.id );

    if ( made.length > 0 )
    {
        const madeSection = section( 'Made', 'project-made', null );
        const madeRows = element( 'div', 'rows' );

        madeRows.append( ...made.map( ( entry ) => madeRow( entry, now ) ) );
        madeSection.append( madeRows );
        root.append( madeSection );
    }

    const events = state.events.filter( ( event ) => event.project === project.id || event.from === project.id );

    if ( events.length === 0 )
        return;

    const history = section( 'History', 'project-history', null );
    const timeline = element( 'div', 'timeline' );

    for ( const recorded of events )
    {
        const entry = element( 'div', 'event ' + ( recorded.who === 'mila' ? 'mila' : 'you' ) );
        const text = element( 'div', 'event-text' );
        const named = ( id ) => ( id === project.id ? project.name : state.projects.find( ( other ) => other.id === id )?.name ?? null );

        text.append( element( 'div', 'event-line', activityText( { kind: 'project', recorded, to: named( recorded.project ), from: named( recorded.from ) } ) ),
            element( 'div', 'event-where', ( recorded.who === 'mila' ? 'Mila' : 'You' ) + ' · ' + dayLabel( new Date( recorded.time ).setHours( 0, 0, 0, 0 ), now ) ) );
        entry.append( element( 'span', 'event-time', clockLabel( recorded.time ) ), text );
        timeline.append( entry );
    }

    history.append( timeline );
    root.append( history );
}
