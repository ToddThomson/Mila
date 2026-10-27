/**
 * @file Assistant.FileTools.ixx
 * @brief The Mila app's file tools: reading, listing, finding, searching, writing and editing files in the folder
 *        the person chose (ToolCalling.md section 5).
 */

export module Assistant.FileTools;

import std;
import nlohmann.json;

import Mila;

namespace Mila::Assistant
{
    using Mila::AI::FolderEntry;
    using Mila::AI::GrantFile;
    using Mila::AI::ToolGrant;
    using Mila::AI::ToolNeeds;

    // Not exported, and of module linkage rather than internal, since the exported class's members use them.
    namespace FileToolsDetail
    {
        /// Larger files are refused whole; search_text finds a passage in them.
        inline constexpr std::uint64_t kMostBytesRead = 8u * 1024u * 1024u;

        inline constexpr std::uint64_t kMostBytesSearched = 2u * 1024u * 1024u;
        inline constexpr int kDefaultLinesRead = 400;
        inline constexpr int kMostLinesRead = 2000;
        inline constexpr std::size_t kMostEntriesListed = 500;
        inline constexpr std::size_t kMostFilesFound = 200;
        inline constexpr std::size_t kMostMatches = 100;
        inline constexpr std::size_t kMostFilesVisited = 20000;
        inline constexpr std::size_t kMostMatchCharacters = 200;
        inline constexpr std::size_t kEditContextLines = 3;

        std::filesystem::path pathFromUtf8( std::string_view text )
        {
            return std::filesystem::path( std::u8string( reinterpret_cast<const char8_t*>( text.data() ), text.size() ) );
        }

        std::string utf8FromPath( const std::filesystem::path& path )
        {
            const std::u8string text = path.generic_u8string();

            return std::string( text.begin(), text.end() );
        }

        std::string requiredText( const nlohmann::json& arguments, const char* name )
        {
            return arguments.at( name ).get<std::string>();
        }

        std::optional<std::string> optionalText( const nlohmann::json& arguments, const char* name )
        {
            if ( !arguments.contains( name ) || arguments[ name ].is_null() )
                return std::nullopt;

            return arguments[ name ].get<std::string>();
        }

        std::optional<int> optionalInteger( const nlohmann::json& arguments, const char* name )
        {
            if ( !arguments.contains( name ) || arguments[ name ].is_null() )
                return std::nullopt;

            return arguments[ name ].get<int>();
        }

        /// A NUL among the first bytes is what text never has.
        bool looksBinary( std::string_view bytes )
        {
            return bytes.substr( 0, 8192 ).find( '\0' ) != std::string_view::npos;
        }

        std::vector<std::string_view> splitLines( std::string_view text )
        {
            std::vector<std::string_view> lines;
            std::size_t start = 0;

            while ( start < text.size() )
            {
                std::size_t end = text.find( '\n', start );

                if ( end == std::string_view::npos )
                    end = text.size();

                std::string_view line = text.substr( start, end - start );

                if ( line.ends_with( '\r' ) )
                    line.remove_suffix( 1 );

                lines.push_back( line );
                start = end + 1;
            }

            return lines;
        }

        std::size_t countOccurrences( std::string_view text, std::string_view passage )
        {
            std::size_t count = 0;

            for ( std::size_t at = text.find( passage ); at != std::string_view::npos; at = text.find( passage, at + 1 ) )
            {
                ++count;
            }

            return count;
        }

        std::string replaceAll( std::string text, std::string_view from, std::string_view to )
        {
            for ( std::size_t at = text.find( from ); at != std::string::npos; at = text.find( from, at + to.size() ) )
            {
                text.replace( at, from.size(), to );
            }

            return text;
        }

        char foldCase( char character )
        {
            return static_cast<char>( std::tolower( static_cast<unsigned char>( character ) ) );
        }

        /// One name against one segment of a pattern: `*` any run of characters, `?` one. ASCII case is ignored, as
        /// Windows file names ignore it.
        bool matchesSegment( std::string_view name, std::string_view pattern )
        {
            if ( pattern.empty() )
                return name.empty();

            if ( pattern.front() == '*' )
            {
                for ( std::size_t skipped = 0; skipped <= name.size(); ++skipped )
                {
                    if ( matchesSegment( name.substr( skipped ), pattern.substr( 1 ) ) )
                        return true;
                }

                return false;
            }

            if ( name.empty() )
                return false;

            if ( pattern.front() != '?' && foldCase( pattern.front() ) != foldCase( name.front() ) )
                return false;

            return matchesSegment( name.substr( 1 ), pattern.substr( 1 ) );
        }

        /// A relative path's segments against a pattern's; `**` stands for any number of folders, none included.
        bool matchesPath( std::span<const std::string> names, std::span<const std::string> patterns )
        {
            if ( patterns.empty() )
                return names.empty();

            if ( patterns.front() == "**" )
            {
                for ( std::size_t skipped = 0; skipped <= names.size(); ++skipped )
                {
                    if ( matchesPath( names.subspan( skipped ), patterns.subspan( 1 ) ) )
                        return true;
                }

                return false;
            }

            if ( names.empty() || !matchesSegment( names.front(), patterns.front() ) )
                return false;

            return matchesPath( names.subspan( 1 ), patterns.subspan( 1 ) );
        }

        std::vector<std::string> segments( std::string_view text )
        {
            std::vector<std::string> parts;

            for ( const auto part : std::views::split( text, '/' ) )
            {
                if ( !part.empty() )
                    parts.emplace_back( part.begin(), part.end() );
            }

            return parts;
        }

        std::string sizeText( std::uint64_t bytes )
        {
            if ( bytes < 1024 )
                return std::format( "{} bytes", bytes );

            if ( bytes < 1024u * 1024u )
                return std::format( "{:.1f} KB", static_cast<double>( bytes ) / 1024.0 );

            return std::format( "{:.1f} MB", static_cast<double>( bytes ) / ( 1024.0 * 1024.0 ) );
        }

        /// What a walk could not enter, so the model knows a result may be missing what lies there.
        std::string notEnteredNote( std::size_t not_entered )
        {
            if ( not_entered == 0 )
                return {};

            return std::format( "({} folder{} not searched: {} outside the project folder or could not be opened)\n",
                not_entered, not_entered == 1 ? "" : "s", not_entered == 1 ? "it leads" : "they lead" );
        }

        /// A walk's visit for files only: .git is never entered, and the walk ends at a bound.
        template <typename TVisitFile>
        ToolGrant::Visit filesUnder( TVisitFile&& visit_file )
        {
            return [visit_file = std::forward<TVisitFile>( visit_file ), visited = std::size_t{ 0 }](
                std::string_view relative, const FolderEntry& entry ) mutable
                {
                    if ( entry.is_folder )
                        return entry.name != ".git";

                    if ( ++visited > kMostFilesVisited )
                        return false;

                    visit_file( relative, entry );

                    return true;
                };
        }
    }

    using namespace FileToolsDetail;

    /**
     * @brief The file tools, each run by its name, through the grant its call was given.
     *
     * Every path a tool takes is opened through the grant (ToolCalling.md 3.4), which holds it to the project
     * folder: a path, link or junction leading outside is refused when it is opened. A failure throws, and the
     * model reads its message as the call's error.
     */
    export class FileTools
    {
    public:

        /// Each tool's name, as the files prompt defines it.
        static constexpr std::array<std::string_view, 6> kNames{
            "read_file", "list_folder", "find_files", "search_text", "write_file", "edit_file" };

        /// What a tool of that name touches: the first four read files, the last two write them.
        [[nodiscard]] static ToolNeeds needsOf( std::string_view name )
        {
            ToolNeeds needs;
            needs.reads_files = true;
            needs.writes_files = name == "write_file" || name == "edit_file";

            return needs;
        }

        /// Runs the tool of that name, one of kNames.
        [[nodiscard]] static std::string run( std::string_view name, const std::string& arguments, const ToolGrant& grant )
        {
            if ( name == "read_file" )
                return readFile( arguments, grant );

            if ( name == "list_folder" )
                return listFolder( arguments, grant );

            if ( name == "find_files" )
                return findFiles( arguments, grant );

            if ( name == "search_text" )
                return searchText( arguments, grant );

            if ( name == "write_file" )
                return writeFile( arguments, grant );

            if ( name == "edit_file" )
                return editFile( arguments, grant );

            throw std::invalid_argument( std::format( "there is no file tool named '{}'", name ) );
        }

        /// `path`, and optionally `start_line` (from 1) and `line_count`.
        [[nodiscard]] static std::string readFile( const std::string& arguments_text, const ToolGrant& grant )
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string given = requiredText( arguments, "path" );
            const GrantFile file = grant.openFile( given );
            const std::uint64_t bytes_on_disk = file.size();

            if ( bytes_on_disk > kMostBytesRead )
            {
                throw std::runtime_error( std::format(
                    "'{}' is {}, more than can be read at once; search_text finds a passage in it", given, sizeText( bytes_on_disk ) ) );
            }

            const std::string bytes = file.read();

            if ( looksBinary( bytes ) )
                return std::format( "'{}' is a binary file of {}, not text.", given, sizeText( bytes.size() ) );

            const auto lines = splitLines( bytes );
            const int total = static_cast<int>( lines.size() );
            const int first = std::max( 1, optionalInteger( arguments, "start_line" ).value_or( 1 ) );
            const int count = std::clamp( optionalInteger( arguments, "line_count" ).value_or( kDefaultLinesRead ), 1, kMostLinesRead );

            if ( total == 0 )
                return std::format( "'{}' is empty.", given );

            if ( first > total )
                throw std::runtime_error( std::format( "'{}' has {} lines, so there is no line {}", given, total, first ) );

            const int last = std::min( total, first + count - 1 );
            std::string result = std::format( "{} -- lines {} to {} of {}\n", given, first, last, total );

            for ( int line = first; line <= last; ++line )
            {
                result.append( lines[ static_cast<std::size_t>( line - 1 ) ] );
                result.push_back( '\n' );
            }

            if ( last < total )
                result += std::format( "({} more lines; read on with start_line {})\n", total - last, last + 1 );

            return result;
        }

        /// `path`: the folder to list, `.` for the project folder.
        [[nodiscard]] static std::string listFolder( const std::string& arguments_text, const ToolGrant& grant )
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string given = requiredText( arguments, "path" );
            const std::vector<FolderEntry> entries = grant.list( given );

            std::vector<std::string> folders;
            std::vector<std::string> files;

            for ( const FolderEntry& entry : entries )
            {
                if ( folders.size() + files.size() >= kMostEntriesListed )
                    break;

                if ( entry.is_folder )
                    folders.push_back( entry.name + "/" );
                else
                    files.push_back( std::format( "{}  ({})", entry.name, sizeText( entry.bytes ) ) );
            }

            std::ranges::sort( folders );
            std::ranges::sort( files );

            std::string result = std::format( "{}\n", given == "." ? "The project folder" : given );

            if ( entries.empty() )
                return result + "(empty)\n";

            for ( const auto& line : folders )
                result += line + "\n";

            for ( const auto& line : files )
                result += line + "\n";

            if ( entries.size() > folders.size() + files.size() )
                result += std::format( "({} more not listed)\n", entries.size() - folders.size() - files.size() );

            return result;
        }

        /// `pattern`: `*` and `?` within a name, `**` for any folders; a pattern with no `/` matches a name anywhere.
        [[nodiscard]] static std::string findFiles( const std::string& arguments_text, const ToolGrant& grant )
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string pattern = requiredText( arguments, "pattern" );
            const bool anywhere = pattern.find( '/' ) == std::string::npos;
            const auto patterns = segments( pattern );

            if ( patterns.empty() )
                throw std::runtime_error( "the pattern is empty" );

            std::vector<std::string> found;
            std::size_t matched = 0;

            const std::size_t not_entered = grant.walk( ".", filesUnder( [&]( std::string_view relative, const FolderEntry& entry )
                {
                    const auto names = segments( relative );
                    const bool matches = anywhere
                        ? matchesSegment( entry.name, patterns.front() )
                        : matchesPath( names, patterns );

                    if ( matches )
                    {
                        ++matched;

                        if ( found.size() < kMostFilesFound )
                            found.emplace_back( relative );
                    }
                } ) );

            if ( found.empty() )
                return std::format( "No file matches '{}'.\n", pattern ) + notEnteredNote( not_entered );

            std::ranges::sort( found );

            std::string result;

            for ( const auto& path : found )
                result += path + "\n";

            if ( matched > found.size() )
                result += std::format( "({} more match; a narrower pattern lists them)\n", matched - found.size() );

            return result + notEnteredNote( not_entered );
        }

        /// `text`, found exactly as written, and optionally `path`, a file or folder to search in.
        [[nodiscard]] static std::string searchText( const std::string& arguments_text, const ToolGrant& grant )
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string text = requiredText( arguments, "text" );
            const std::string given = optionalText( arguments, "path" ).value_or( "." );

            if ( text.empty() )
                throw std::runtime_error( "the text to search for is empty" );

            const std::optional<FolderEntry> where = grant.find( given );

            if ( !where )
                throw std::runtime_error( std::format( "there is no file or folder '{}'", given ) );

            std::string result;
            std::size_t matches = 0;

            const auto searchFile = [&]( const std::string& shown )
                {
                    std::string bytes;

                    // A file that cannot be opened through the grant -- a link leading outside, a hard link -- is
                    // passed over, as an unreadable one is.
                    try
                    {
                        const GrantFile file = grant.openFile( shown );

                        if ( file.size() > kMostBytesSearched )
                            return;

                        bytes = file.read();
                    }
                    catch ( const std::runtime_error& )
                    {
                        return;
                    }

                    if ( looksBinary( bytes ) || bytes.find( text ) == std::string::npos )
                        return;

                    const auto lines = splitLines( bytes );

                    for ( std::size_t index = 0; index < lines.size(); ++index )
                    {
                        if ( lines[ index ].find( text ) == std::string_view::npos )
                            continue;

                        ++matches;

                        if ( matches <= kMostMatches )
                        {
                            std::string_view line = lines[ index ];
                            const bool cut = line.size() > kMostMatchCharacters;
                            result += std::format( "{}:{}: {}{}\n", shown, index + 1, line.substr( 0, kMostMatchCharacters ), cut ? " ..." : "" );
                        }
                    }
                };

            std::size_t not_entered = 0;

            if ( !where->is_folder )
            {
                searchFile( given );
            }
            else
            {
                not_entered = grant.walk( given, filesUnder( [&]( std::string_view relative, const FolderEntry& entry )
                    {
                        if ( entry.bytes <= kMostBytesSearched )
                            searchFile( utf8FromPath( ( pathFromUtf8( given ) / pathFromUtf8( relative ) ).lexically_normal() ) );
                    } ) );
            }

            if ( matches == 0 )
                return std::format( "'{}' was not found.\n", text ) + notEnteredNote( not_entered );

            if ( matches > kMostMatches )
                result += std::format( "({} more matches; a narrower path or longer text shows them)\n", matches - kMostMatches );

            return result + notEnteredNote( not_entered );
        }

        /// `path` and `content`: the file is created, or replaced whole, and its folders made as needed.
        [[nodiscard]] static std::string writeFile( const std::string& arguments_text, const ToolGrant& grant )
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string given = requiredText( arguments, "path" );
            const std::string content = requiredText( arguments, "content" );
            const std::optional<FolderEntry> existing = grant.find( given );

            if ( existing && existing->is_folder )
                throw std::runtime_error( std::format( "'{}' is a folder", given ) );

            GrantFile file = grant.openFileToWrite( given, true );
            file.replace( content );

            return std::format( "{} '{}' ({}).", existing ? "Replaced" : "Created", given, sizeText( content.size() ) );
        }

        /// `path`, `old_text` and `new_text`: the one place old_text appears becomes new_text.
        [[nodiscard]] static std::string editFile( const std::string& arguments_text, const ToolGrant& grant )
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string given = requiredText( arguments, "path" );
            std::string old_text = requiredText( arguments, "old_text" );
            std::string new_text = requiredText( arguments, "new_text" );

            if ( old_text.empty() )
                throw std::runtime_error( "old_text is empty; write_file makes a new file" );

            GrantFile file = grant.openFileToWrite( given, false );
            std::string bytes = file.read();

            // A model writes its lines with \n; a file kept with \r\n is matched and edited in its own endings.
            if ( bytes.find( "\r\n" ) != std::string::npos && old_text.find( "\r\n" ) == std::string::npos )
            {
                old_text = replaceAll( old_text, "\n", "\r\n" );
                new_text = replaceAll( new_text, "\n", "\r\n" );
            }

            const std::size_t occurrences = countOccurrences( bytes, old_text );

            if ( occurrences == 0 )
            {
                throw std::runtime_error( std::format(
                    "old_text does not appear in '{}'; read the file again and copy the passage exactly", given ) );
            }

            if ( occurrences > 1 )
            {
                throw std::runtime_error( std::format(
                    "old_text appears {} times in '{}'; include more of the text around it so it appears once", occurrences, given ) );
            }

            if ( old_text == new_text )
                throw std::runtime_error( "old_text and new_text are the same, so there is nothing to change" );

            const std::size_t at = bytes.find( old_text );
            bytes.replace( at, old_text.size(), new_text );
            file.replace( bytes );

            // The lines around the change as they now read, so the model checks its edit without reading the file again.
            const auto lines = splitLines( bytes );
            const std::size_t first_changed = static_cast<std::size_t>( std::count( bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>( at ), '\n' ) );
            const std::size_t last_changed = first_changed + static_cast<std::size_t>( std::count( new_text.begin(), new_text.end(), '\n' ) );
            const std::size_t from = first_changed >= kEditContextLines ? first_changed - kEditContextLines : 0;
            const std::size_t to = std::min( lines.empty() ? 0 : lines.size() - 1, last_changed + kEditContextLines );
            std::string result = std::format( "Edited '{}'. Lines {} to {} now read:\n", given, from + 1, to + 1 );

            for ( std::size_t line = from; line <= to && line < lines.size(); ++line )
            {
                result.append( lines[ line ] );
                result.push_back( '\n' );
            }

            return result;
        }
    };
}
