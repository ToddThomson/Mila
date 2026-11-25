// The words and numbers the page shows for session facts. No DOM, so they are tested with the
// reducer.

const numberFormat = new Intl.NumberFormat( 'en-US' );

export function formatCount( value )
{
    return numberFormat.format( value );
}

export function formatBytes( bytes )
{
    const gigabytes = bytes / 1e9;

    if ( gigabytes >= 1 )
        return gigabytes.toFixed( 1 ) + ' GB';

    const megabytes = bytes / 1e6;

    if ( megabytes >= 1 )
        return Math.round( megabytes ) + ' MB';

    const rounded = Math.round( bytes / 1e3 );
    const kilobytes = Math.max( 1, rounded );

    return kilobytes + ' KB';
}

/** What installing a published model downloads, for its card; null when the publisher did not say. */
export function downloadLine( model )
{
    return model.size_bytes > 0 ? 'Download: ' + formatBytes( model.size_bytes ) : null;
}

/** How far an install's download has come, beside its bar: "3.1 GB of 15.2 GB". */
export function installAmount( progress )
{
    const sofar = progress.done > 0 ? formatBytes( progress.done ) : '0 KB';

    return sofar + ' of ' + formatBytes( progress.total );
}

/**
 * What a published model can do, in a line for a person choosing one: an agent only where it was measured as one
 * and planned here at that depth, then thinking and what it reads besides text.
 */
export function modelAbilities( model )
{
    const parts = [ model.agentic_here ? 'Works through tasks with tools' : 'Answers questions and holds a conversation' ];

    if ( model.thinking )
        parts.push( 'thinks before it answers' );

    const words = { image: 'images', audio: 'audio' };
    const reads = model.modalities.filter( ( modality ) => words[ modality ] !== undefined );

    if ( reads.length > 0 )
        parts.push( 'reads ' + reads.map( ( modality ) => words[ modality ] ).join( ' and ' ) );

    return parts.join( ', ' ) + '.';
}

/** A model's name for a person to read: its manifest's display name, or its name when it declares none. */
export function modelTitle( model )
{
    return model.display_name ? model.display_name : model.name;
}

/**
 * A model the session names, as a person reads it. The installed list first; then the published one, which
 * already holds a model just installed while the installed list is still being sent; its name when in neither.
 */
export function namedTitle( state, name )
{
    const lists = [ state.installedModels, state.publishedModels ?? [] ];

    for ( const list of lists )
    {
        const model = list.find( ( listed ) => listed.name === name );

        if ( model !== undefined && model.display_name )
            return model.display_name;
    }

    return name;
}

/**
 * Why the suggestion suits this computer, from the planner's answer: room for long tasks only when it plans the
 * model agentic here (AssistantOnboarding.md 5); otherwise the card's label already says it runs here.
 */
export function fitLine( model )
{
    return model.agentic_here ? 'Runs on this computer with room for long tasks.' : null;
}

/** How much of the context the conversation takes, from 0 to 1; null before the session has counted it. */
export function contextShare( session )
{
    if ( session.context_used === null || session.context_used === undefined || !session.context_length )
        return null;

    return Math.min( 1, Math.max( 0, session.context_used / session.context_length ) );
}

/** The context ring's popup: what is used, the window, where compaction starts, and what is left, as label and value. */
export function contextDetails( session )
{
    const length = session.context_length;
    const used = session.context_used;
    const windowText = formatCount( length ) + ' tokens, ' + ( session.context_automatic ? 'chosen for this computer' : 'set in Settings' );
    const rows = [];

    if ( used === null || used === undefined )
    {
        rows.push( [ 'Used', 'Counted after the next reply' ] );
    }
    else
    {
        const percent = Math.round( contextShare( session ) * 100 );
        rows.push( [ 'Used', formatCount( used ) + ' tokens (' + percent + '%)' ] );
    }

    rows.push( [ 'Context window', windowText ] );

    if ( session.compacts_at !== null && session.compacts_at !== undefined )
        rows.push( [ 'Compacts at', formatCount( session.compacts_at ) + ' tokens' ] );

    if ( used !== null && used !== undefined )
        rows.push( [ 'Left', formatCount( Math.max( 0, length - used ) ) + ' tokens' ] );

    return rows;
}

/**
 * The thinking clause is left out for a model without the channel: "off" would read as a
 * setting the user could turn on.
 */
export function thinkingFact( session )
{
    if ( !session.thinking_capable )
        return null;

    return 'Thinking ' + session.thinking;
}

export function thoughtLabel( block, streaming )
{
    if ( block.tokens === null )
    {
        if ( !streaming )
            return 'Thought';

        return block.liveTokens > 0 ? 'Thinking · ' + formatCount( block.liveTokens ) + ' tokens' : 'Thinking';
    }

    const count = formatCount( block.tokens );
    const unit = block.tokens === 1 ? ' token' : ' tokens';

    return 'Thought for ' + count + unit;
}

export function statsLine( summary )
{
    if ( summary === null )
        return '';

    const parts = [];
    const prefill = 'Prefill ' + Math.round( summary.prefillMs ) + ' ms';
    parts.push( prefill );

    if ( summary.tokensPerSecond > 0 )
    {
        const rate = summary.tokensPerSecond.toFixed( 1 ) + ' tok/s';
        parts.push( rate );
    }

    const tokens = formatCount( summary.tokens ) + ( summary.tokens === 1 ? ' token' : ' tokens' );
    parts.push( tokens );

    const rounds = summary.rounds + ( summary.rounds === 1 ? ' round' : ' rounds' );
    parts.push( rounds );

    return parts.join( ' · ' );
}

/**
 * What a reply that did not finish on its own says about it, in the reply itself.
 */
export function finishNotice( finish )
{
    switch ( finish )
    {
        case 'length':
            return 'The reply reached its length limit before it finished.';

        case 'context':
            return 'The reply stopped at the end of the context window.';

        case 'cancelled':
            return 'Stopped.';

        case 'error':
            return 'The reply ended with an error.';

        default:
            return null;
    }
}

/**
 * A tool's arguments or result for reading: indented when it is JSON, as written when not.
 */
export function readableJson( text )
{
    if ( text === null || text === undefined )
        return '';

    try
    {
        const parsed = JSON.parse( text );

        return JSON.stringify( parsed, null, 2 );
    }
    catch
    {
        return text;
    }
}

/**
 * What the empty Mila bar says (AssistantInterface.md 5.1): in a chat it is the reply box; elsewhere it takes
 * anything, and the person's chats are a click away.
 */
export function barPrompt( place, projectName, askingForProject = false )
{
    if ( place === 'chat' )
        return 'Reply to Mila';

    // The sidebar's *New project* asks Mila to start one (4.1).
    if ( askingForProject )
        return 'Tell Mila what the project is for';

    // On a project's page what is asked begins a chat in it (5.3).
    if ( place === 'project' && projectName )
        return 'Ask Mila about ' + projectName;

    return 'Ask Mila anything, or open one of your chats';
}

/**
 * How far a plan has come: steps done or skipped of all of them.
 */
export function planProgress( plan )
{
    const finished = plan.filter( ( step ) => step.status === 'done' || step.status === 'skipped' ).length;

    return finished + ' of ' + plan.length + ' done';
}

const minuteMilliseconds = 60 * 1000;
const hourMilliseconds = 60 * minuteMilliseconds;
const weekdayFormat = new Intl.DateTimeFormat( 'en-US', { weekday: 'long' } );
const dateFormat = new Intl.DateTimeFormat( 'en-US', { month: 'short', day: 'numeric' } );

/**
 * When something happened, as a person says it: minutes and hours today, then yesterday, the weekday within a
 * week, and the date before that. Days are the viewer's local days.
 */
export function whenLabel( updated, now )
{
    const then = new Date( updated );
    const elapsed = now.getTime() - then.getTime();
    const today = new Date( now.getFullYear(), now.getMonth(), now.getDate() ).getTime();
    const day = 24 * hourMilliseconds;

    if ( elapsed < minuteMilliseconds )
        return 'Just now';

    if ( then.getTime() >= today )
    {
        if ( elapsed < hourMilliseconds )
        {
            const minutes = Math.floor( elapsed / minuteMilliseconds );

            return minutes === 1 ? '1 minute ago' : minutes + ' minutes ago';
        }

        const hours = Math.floor( elapsed / hourMilliseconds );

        return hours === 1 ? '1 hour ago' : hours + ' hours ago';
    }

    if ( then.getTime() >= today - day )
        return 'Yesterday';

    if ( then.getTime() >= today - 6 * day )
        return weekdayFormat.format( then );

    return dateFormat.format( then );
}

/**
 * What the home screen says of a row that needs the person: where a plan stopped, or how a reply ended.
 */
export function attentionLine( row )
{
    if ( row.kind === 'plan' )
    {
        const progress = row.item.progress;
        const step = progress.finished + 1;

        return 'Stopped at step ' + Math.min( step, progress.steps ) + ' of ' + progress.steps + ': ' + progress.current;
    }

    if ( row.kind === 'changed' )
        return 'Changed on disk after Mila last wrote it, in ' + row.item.title;

    const endings = {
        length: 'The last reply was cut short',
        context: 'The conversation ran out of room',
        error: 'The last reply ended in an error'
    };

    return endings[ row.item.finish ] ?? '';
}

/**
 * How far a plan came, from a conversation's summary: steps finished of all of them, or finished.
 */
export function progressLine( progress )
{
    if ( progress.current === null )
        return 'Finished, ' + progress.steps + ( progress.steps === 1 ? ' step' : ' steps' );

    return progress.finished + ' of ' + progress.steps + ' steps done';
}

/**
 * The last part of a folder's path, which is how a person names their project; of a file's path, its name.
 */
export function folderName( folder )
{
    const parts = folder.split( /[\\/]/ ).filter( ( part ) => part.length > 0 );

    return parts.length === 0 ? folder : parts[ parts.length - 1 ];
}

/**
 * A reply's start as one line of plain text for a list: the markdown's marks taken out, its words kept.
 */
export function plainLine( text )
{
    if ( !text )
        return '';

    return text
        .replace( /```[^\n]*/g, ' ' )
        .replace( /!?\[([^\]]*)\]\([^)]*\)?/g, '$1' )
        .replace( /(^|\s)#{1,6}\s+/g, '$1' )
        .replace( /(^|\s)>\s+/g, '$1' )
        .replace( /\*\*|__|[*`~]/g, '' )
        .replace( /\|?(\s*:?-{3,}:?\s*\|)+\s*(:?-{3,}:?)?/g, ' ' )
        .replace( /\|/g, ' ' )
        .replace( /\s+/g, ' ' )
        .trim();
}

/** The names of the files a chat wrote, the latest first: two named, then how many more. */
function filesNamed( wrote )
{
    const names = wrote.slice().reverse().map( ( file ) => folderName( file.path ) );

    if ( names.length === 1 )
        return names[ 0 ];

    if ( names.length === 2 )
        return names[ 0 ] + ' and ' + names[ 1 ];

    const more = names.length - 2;

    return names[ 0 ] + ', ' + names[ 1 ] + ' and ' + more + ( more === 1 ? ' more' : ' others' );
}

/**
 * What one of the session's events says (AssistantInterface.md 8): `recorded` is the event as stored, and `to` and
 * `from` the names of the projects it names as they are now, or null for one deleted since.
 */
export function projectEventText( recorded, to, from )
{
    const project = to ?? recorded.name ?? 'a project';
    const folder = typeof recorded.path === 'string' ? folderName( recorded.path ) : 'a folder';

    switch ( recorded.kind )
    {
        case 'project.made': return 'Made the project ' + project;
        case 'project.renamed': return typeof recorded.name === 'string' ? 'Renamed a project to ' + recorded.name : 'Renamed a project';
        case 'project.purpose': return 'Changed what ' + project + ' is for';
        case 'project.instructions': return 'Changed the instructions of ' + project;
        case 'project.folder_added': return 'Gave ' + project + ' the folder ' + folder + ( recorded.access === 'read_write' ? ', to read and change' : ', to read' );
        case 'project.folder_access': return ( recorded.access === 'read_write' ? 'Let ' + project + ' change files in ' : 'Kept ' + project + ' to reading ' ) + folder;
        case 'project.folder_removed': return 'Took the folder ' + folder + ' from ' + project;
        case 'project.deleted': return 'Deleted a project';
        case 'chat.moved': return movedText( recorded, to, from );
        case 'suggestion.made': return to !== null ? 'Suggested moving a chat into ' + to : 'Suggested a project: ' + ( recorded.name ?? '' );
        case 'suggestion.declined': return to !== null ? 'Kept a chat out of ' + to : 'Declined the project ' + ( recorded.name ?? '' );
        case 'project.undone': return 'Undid the project ' + ( recorded.name ?? '' );
        case 'understanding.noted': return 'Noted what Mila understands of ' + project;
        case 'understanding.waiting': return 'Wrote a note on ' + project + ' that waits for you';
        case 'understanding.accepted': return 'Accepted the note waiting on ' + project;
        case 'understanding.declined': return 'Declined the note waiting on ' + project;
        case 'understanding.edited': return 'Changed what Mila understands of ' + project;
        case 'understanding.cleared': return recorded.because === 'chat.deleted'
            ? 'Cleared the note on ' + project + ', since a chat it was drawn from was deleted'
            : 'Cleared what Mila understands of ' + project;
        default: return 'Changed a project';
    }
}

function movedText( recorded, to, from )
{
    const chat = recorded.name ?? 'a chat';

    if ( to !== null && from !== null )
        return 'Moved ' + chat + ' from ' + from + ' to ' + to;

    if ( to !== null )
        return 'Moved ' + chat + ' into ' + to;

    if ( from !== null )
        return 'Moved ' + chat + ' out of ' + from;

    return 'Moved ' + chat;
}

/** A line cut at the last word that fits @p most characters, with an ellipsis when anything was cut. */
function shortLine( line, most )
{
    if ( line.length <= most )
        return line;

    const cut = line.slice( 0, most );
    const space = cut.lastIndexOf( ' ' );

    return ( space > most / 2 ? cut.slice( 0, space ) : cut ).replace( /[\s,.;:]+$/, '' ) + '…';
}

/**
 * The chats Mila's suggestion of a project gathers (AssistantInterface.md 6), each title in quotes. In the chat that made
 * the offer, @p here is its id, and it is *this chat*.
 */
export function gatheredLine( offer, here = null )
{
    const chats = Array.isArray( offer.chats ) ? offer.chats : [];
    const titled = chats.filter( ( chat ) => chat.id !== here && typeof chat.title === 'string' && chat.title.length > 0 )
        .map( ( chat ) => '\u201C' + chat.title + '\u201D' );
    const named = here !== null && chats.some( ( chat ) => chat.id === here ) ? [ 'this chat', ...titled ] : titled;

    if ( named.length === 0 )
        return 'Gathers its chats';

    return 'Gathers ' + ( named.length === 1 ? named[ 0 ] : named.slice( 0, -1 ).join( ', ' ) + ' and ' + named.at( -1 ) );
}

/** What an Activity entry says (AssistantInterface.md 4.8). */
export function activityText( event )
{
    const item = event.item;

    if ( event.kind === 'project' )
        return projectEventText( event.recorded, event.to, event.from );

    if ( event.kind === 'sent' )
        return sentText( event.sent );

    if ( event.kind === 'chat' )
        return 'Started a chat: ' + item.title;

    if ( event.kind === 'file' && event.reply !== undefined )
        return 'Kept a reply' + ( event.reply.start ? ': ' + shortLine( plainLine( event.reply.start ), 80 ) : '' );

    if ( event.kind === 'file' )
        return ( event.file !== undefined && event.file.tool === 'edit_file' ? 'Changed ' : 'Wrote ' ) + filesNamed( event.files ?? item.wrote );

    const progress = item.progress;

    if ( progress.current === null )
        return 'Worked through a plan of ' + progress.steps + ( progress.steps === 1 ? ' step' : ' steps' );

    return 'Stopped at step ' + Math.min( progress.finished + 1, progress.steps ) + ' of ' + progress.steps + ': ' + progress.current;
}

/** One thing that left the computer, exactly as it went (AssistantInterface.md 5.7). */
function sentText( sent )
{
    switch ( sent.kind )
    {
        case 'page': return 'Read ' + sent.what;
        case 'search': return 'Searched for “' + sent.what + '”';
        case 'command': return 'Ran with internet access: ' + sent.what;
        default: return 'Opened ' + sent.what;
    }
}

/**
 * What the sheet before a link says (AssistantInterface.md 5.7): where it goes and everything it carries, never the
 * words the link was shown under, which the model chose. Null for what cannot be opened.
 */
export function linkDescription( url )
{
    let parsed = null;

    try
    {
        parsed = new URL( url );
    }
    catch
    {
        return null;
    }

    if ( parsed.protocol === 'mailto:' )
    {
        const facts = [ [ 'To', decodeURIComponent( parsed.pathname ) ] ];

        for ( const [ name, label ] of [ [ 'cc', 'Copy to' ], [ 'bcc', 'Hidden copy to' ], [ 'subject', 'Subject' ], [ 'body', 'Message' ] ] )
        {
            const value = parsed.searchParams.get( name );

            if ( value !== null && value.length > 0 )
                facts.push( [ label, value ] );
        }

        return { title: 'Start this email?', action: 'Open in Mail', facts, note: 'Your mail program opens with this filled in.' };
    }

    if ( parsed.protocol !== 'https:' && parsed.protocol !== 'http:' )
        return null;

    const note = parsed.protocol === 'http:'
        ? 'This address is not encrypted, so others on the network can see it.'
        : 'Everything in this address goes to this site.';

    return { title: 'Open this page?', action: 'Open', facts: [ [ 'Site', parsed.host ], [ 'Full address', url ] ], note };
}

function argumentsOf( block )
{
    try
    {
        const parsed = JSON.parse( block.arguments );

        return parsed !== null && typeof parsed === 'object' ? parsed : {};
    }
    catch
    {
        return {};
    }
}

/**
 * What a call waiting on the person asks, from its own arguments: what would leave and where, and whether allowing it
 * also lets Mila change files in the chat's folder (AssistantInterface.md 5.5 and 5.7).
 */
export function approvalWords( block )
{
    const given = argumentsOf( block );
    const facts = [];
    const notes = [];
    let question = 'Run this tool?';

    if ( block.name === 'read_web_page' && typeof given.url === 'string' )
    {
        const link = linkDescription( given.url );

        question = 'Read this page?';
        facts.push( [ 'Address', given.url ] );

        if ( link !== null )
            notes.push( link.note );
    }

    if ( block.name === 'web_search' && typeof given.query === 'string' )
    {
        question = 'Send this search?';
        facts.push( [ 'Search words', given.query ] );
        notes.push( 'These words go to your search provider.' );
    }

    if ( block.name === 'run_command' && given.network === true )
        notes.push( 'With internet access, this command can send anything in the folder to any address.' );

    if ( block.changesFolder )
    {
        question = 'Allow this change?';
        notes.push( 'This also lets Mila change files in ' + block.changesFolder + ' for the rest of this chat.' );
    }

    // A chat that has read a page, a file or a command's output may be asking for this because of it (5.6).
    if ( block.hasRead && Array.isArray( block.hasRead.latest ) && block.hasRead.latest.length > 0 )
    {
        const latest = block.hasRead.latest.map( source => source.summary );
        const more = block.hasRead.count - latest.length;

        facts.push( [ 'This chat has read', latest.join( '; ' ) + ( more > 0 ? ' and ' + more + ' more' : '' ) ] );
    }

    if ( typeof block.notInYourWords === 'string' )
        notes.push( block.name === 'web_search'
            ? 'You did not write these search words in this chat.'
            : 'You did not write this address in this chat.' );

    return { question, facts, notes };
}

const toolWords = {
    list_folder: 'Listing folders',
    read_file: 'Reading files',
    find_files: 'Finding files',
    search_text: 'Searching inside files',
    write_file: 'Writing files',
    edit_file: 'Editing files',
    propose_plan: 'Proposing plans'
};

/** A tool allowed for a chat, as the person reads it. */
export function toolWord( name )
{
    return toolWords[ name ] ?? name;
}

const longDateFormat = new Intl.DateTimeFormat( 'en-US', { weekday: 'long', month: 'long', day: 'numeric' } );
const clockFormat = new Intl.DateTimeFormat( 'en-US', { hour: 'numeric', minute: '2-digit' } );

/** The date at the head of the overview: "Monday, November 16". */
export function longDate( date )
{
    return longDateFormat.format( date );
}

/** A day's heading in Activity: Today, Yesterday, then the date. `day` is its local midnight in milliseconds. */
export function dayLabel( day, now )
{
    const today = new Date( now.getFullYear(), now.getMonth(), now.getDate() ).getTime();
    const yesterday = new Date( now.getFullYear(), now.getMonth(), now.getDate() - 1 ).getTime();

    if ( day === today )
        return 'Today';

    if ( day === yesterday )
        return 'Yesterday';

    return longDateFormat.format( new Date( day ) );
}

/** The time of day an entry happened: "3:05 PM". */
export function clockLabel( time )
{
    return clockFormat.format( new Date( time ) );
}

/**
 * Up to two letters for a person's name: the first of its first two words, or of a name written as one, its first
 * letter and the next capital ("ToddT" is TT).
 */
export function initials( name )
{
    const words = name.split( /[\s._-]+/ ).filter( ( word ) => word.length > 0 );

    if ( words.length === 0 )
        return '';

    if ( words.length > 1 )
        return ( words[ 0 ][ 0 ] + words[ 1 ][ 0 ] ).toUpperCase();

    const capital = words[ 0 ].slice( 1 ).match( /\p{Lu}/u );

    return ( words[ 0 ][ 0 ] + ( capital === null ? '' : capital[ 0 ] ) ).toUpperCase();
}

/**
 * The licence as paragraphs a box wraps: its text is broken at eighty columns, which a narrower box would break again
 * mid-line. Blank lines still part its paragraphs, and every word stays as written.
 */
export function licenseParagraphs( text )
{
    const normalized = text.replace( /\r\n/g, '\n' ).trim();
    const blocks = normalized.split( /\n[ \t]*\n/ );
    const joined = blocks.map( ( block ) => block.replace( /[ \t]*\n[ \t]*/g, ' ' ) );

    return joined.join( '\n\n' );
}
