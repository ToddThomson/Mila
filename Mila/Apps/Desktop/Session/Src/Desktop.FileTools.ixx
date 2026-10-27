/**
 * @file Desktop.FileTools.ixx
 * @brief The desktop app's file tools: reading, listing, finding, searching, writing and editing files in the folder
 *        the person chose (ToolCalling.md section 5).
 */

export module Desktop.FileTools;

import std;
import Chat.Json;

namespace Mila::DesktopApp
{
    // Not exported, and of module linkage rather than internal, since the exported class's template member uses them.
    namespace FileToolsDetail
    {
        /// Larger files are refused whole; search_text finds a passage in them.
        inline constexpr std::uintmax_t kMostBytesRead = 8u * 1024u * 1024u;

        inline constexpr std::uintmax_t kMostBytesSearched = 2u * 1024u * 1024u;
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

        std::string readBytes( const std::filesystem::path& file )
        {
            std::ifstream stream( file, std::ios::binary );

            if ( !stream )
                throw std::runtime_error( "the file could not be opened" );

            return std::string( std::istreambuf_iterator<char>( stream ), std::istreambuf_iterator<char>() );
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

        std::string sizeText( std::uintmax_t bytes )
        {
            if ( bytes < 1024 )
                return std::format( "{} bytes", bytes );

            if ( bytes < 1024u * 1024u )
                return std::format( "{:.1f} KB", static_cast<double>( bytes ) / 1024.0 );

            return std::format( "{:.1f} MB", static_cast<double>( bytes ) / ( 1024.0 * 1024.0 ) );
        }
    }

    using namespace FileToolsDetail;

    /**
     * @brief The file tools over one folder, each run by its name.
     *
     * Every path a tool takes is relative to the folder and passes through locate(), the one place the grant of
     * ToolCalling.md 3.4 replaces. Until then nothing here holds a path to the folder: a tool opens what it is
     * given, and only the person's answer to each call stands in the way (Untriaged.md, "The desktop app's file
     * tools open files directly"). A failure throws, and the model reads its message as the call's error.
     */
    export class FileTools
    {
    public:

        /// Each tool's name, as the files prompt defines it.
        static constexpr std::array<std::string_view, 6> kNames{
            "read_file", "list_folder", "find_files", "search_text", "write_file", "edit_file" };

        /**
         * @throws std::invalid_argument when the folder does not exist or is not a folder.
         */
        explicit FileTools( std::filesystem::path folder )
        {
            std::error_code error;

            if ( !std::filesystem::is_directory( folder, error ) )
                throw std::invalid_argument( std::format( "'{}' is not a folder", utf8FromPath( folder ) ) );

            folder_ = std::filesystem::absolute( folder ).lexically_normal();
        }

        /// The tools over a folder named as the schema carries it, in UTF-8.
        [[nodiscard]] static FileTools inFolder( std::string_view folder )
        {
            return FileTools( pathFromUtf8( folder ) );
        }

        [[nodiscard]] const std::filesystem::path& folder() const noexcept
        {
            return folder_;
        }

        /// The folder in UTF-8, with the system's own separators, as the person's picker gave it.
        [[nodiscard]] std::string folderText() const
        {
            const std::u8string text = folder_.u8string();

            return std::string( text.begin(), text.end() );
        }

        /// Runs the tool of that name, one of kNames.
        [[nodiscard]] std::string run( std::string_view name, const std::string& arguments ) const
        {
            if ( name == "read_file" )
                return readFile( arguments );

            if ( name == "list_folder" )
                return listFolder( arguments );

            if ( name == "find_files" )
                return findFiles( arguments );

            if ( name == "search_text" )
                return searchText( arguments );

            if ( name == "write_file" )
                return writeFile( arguments );

            if ( name == "edit_file" )
                return editFile( arguments );

            throw std::invalid_argument( std::format( "there is no file tool named '{}'", name ) );
        }

        /// `path`, and optionally `start_line` (from 1) and `line_count`.
        [[nodiscard]] std::string readFile( const std::string& arguments_text ) const
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string given = requiredText( arguments, "path" );
            const auto file = locate( given );

            if ( !std::filesystem::is_regular_file( file ) )
                throw std::runtime_error( std::format( "there is no file '{}'", given ) );

            const auto bytes_on_disk = std::filesystem::file_size( file );

            if ( bytes_on_disk > kMostBytesRead )
            {
                throw std::runtime_error( std::format(
                    "'{}' is {}, more than can be read at once; search_text finds a passage in it", given, sizeText( bytes_on_disk ) ) );
            }

            const std::string bytes = readBytes( file );

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
        [[nodiscard]] std::string listFolder( const std::string& arguments_text ) const
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string given = requiredText( arguments, "path" );
            const auto folder = locate( given );

            if ( !std::filesystem::is_directory( folder ) )
                throw std::runtime_error( std::format( "there is no folder '{}'", given ) );

            std::vector<std::string> folders;
            std::vector<std::string> files;
            std::size_t seen = 0;

            for ( const auto& entry : std::filesystem::directory_iterator(
                folder, std::filesystem::directory_options::skip_permission_denied ) )
            {
                ++seen;

                if ( folders.size() + files.size() >= kMostEntriesListed )
                    continue;

                std::error_code error;
                const std::string name = utf8FromPath( entry.path().filename() );

                if ( entry.is_directory( error ) )
                    folders.push_back( name + "/" );
                else
                    files.push_back( std::format( "{}  ({})", name, sizeText( entry.file_size( error ) ) ) );
            }

            std::ranges::sort( folders );
            std::ranges::sort( files );

            std::string result = std::format( "{}\n", given == "." ? "The project folder" : given );

            if ( seen == 0 )
                return result + "(empty)\n";

            for ( const auto& line : folders )
                result += line + "\n";

            for ( const auto& line : files )
                result += line + "\n";

            if ( seen > folders.size() + files.size() )
                result += std::format( "({} more not listed)\n", seen - folders.size() - files.size() );

            return result;
        }

        /// `pattern`: `*` and `?` within a name, `**` for any folders; a pattern with no `/` matches a name anywhere.
        [[nodiscard]] std::string findFiles( const std::string& arguments_text ) const
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string pattern = requiredText( arguments, "pattern" );
            const bool anywhere = pattern.find( '/' ) == std::string::npos;
            const auto patterns = segments( pattern );

            if ( patterns.empty() )
                throw std::runtime_error( "the pattern is empty" );

            std::vector<std::string> found;
            std::size_t matched = 0;

            visitFiles( folder_, [&]( const std::filesystem::path& relative )
                {
                    const auto names = segments( utf8FromPath( relative ) );
                    const bool matches = anywhere
                        ? matchesSegment( names.back(), patterns.front() )
                        : matchesPath( names, patterns );

                    if ( matches )
                    {
                        ++matched;

                        if ( found.size() < kMostFilesFound )
                            found.push_back( utf8FromPath( relative ) );
                    }
                } );

            if ( found.empty() )
                return std::format( "No file matches '{}'.", pattern );

            std::ranges::sort( found );

            std::string result;

            for ( const auto& path : found )
                result += path + "\n";

            if ( matched > found.size() )
                result += std::format( "({} more match; a narrower pattern lists them)\n", matched - found.size() );

            return result;
        }

        /// `text`, found exactly as written, and optionally `path`, a file or folder to search in.
        [[nodiscard]] std::string searchText( const std::string& arguments_text ) const
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string text = requiredText( arguments, "text" );
            const std::string given = optionalText( arguments, "path" ).value_or( "." );
            const auto where = locate( given );

            if ( text.empty() )
                throw std::runtime_error( "the text to search for is empty" );

            if ( !std::filesystem::exists( where ) )
                throw std::runtime_error( std::format( "there is no file or folder '{}'", given ) );

            std::string result;
            std::size_t matches = 0;

            const auto searchFile = [&]( const std::filesystem::path& file, const std::string& shown )
                {
                    std::error_code error;

                    if ( std::filesystem::file_size( file, error ) > kMostBytesSearched || error )
                        return;

                    const std::string bytes = readBytes( file );

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

            if ( std::filesystem::is_regular_file( where ) )
            {
                searchFile( where, given );
            }
            else
            {
                visitFiles( where, [&]( const std::filesystem::path& relative )
                    {
                        const auto shown = utf8FromPath( ( pathFromUtf8( given ) / relative ).lexically_normal() );
                        searchFile( where / relative, shown );
                    } );
            }

            if ( matches == 0 )
                return std::format( "'{}' was not found.", text );

            if ( matches > kMostMatches )
                result += std::format( "({} more matches; a narrower path or longer text shows them)\n", matches - kMostMatches );

            return result;
        }

        /// `path` and `content`: the file is created, or replaced whole, and its folders made as needed.
        [[nodiscard]] std::string writeFile( const std::string& arguments_text ) const
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string given = requiredText( arguments, "path" );
            const std::string content = requiredText( arguments, "content" );
            const auto file = locate( given );

            if ( std::filesystem::is_directory( file ) )
                throw std::runtime_error( std::format( "'{}' is a folder", given ) );

            const bool existed = std::filesystem::exists( file );

            std::filesystem::create_directories( file.parent_path() );

            std::ofstream stream( file, std::ios::binary | std::ios::trunc );

            if ( !stream || !stream.write( content.data(), static_cast<std::streamsize>( content.size() ) ) )
                throw std::runtime_error( std::format( "'{}' could not be written", given ) );

            return std::format( "{} '{}' ({}).", existed ? "Replaced" : "Created", given, sizeText( content.size() ) );
        }

        /// `path`, `old_text` and `new_text`: the one place old_text appears becomes new_text.
        [[nodiscard]] std::string editFile( const std::string& arguments_text ) const
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string given = requiredText( arguments, "path" );
            std::string old_text = requiredText( arguments, "old_text" );
            std::string new_text = requiredText( arguments, "new_text" );
            const auto file = locate( given );

            if ( !std::filesystem::is_regular_file( file ) )
                throw std::runtime_error( std::format( "there is no file '{}'", given ) );

            if ( old_text.empty() )
                throw std::runtime_error( "old_text is empty; write_file makes a new file" );

            std::string bytes = readBytes( file );

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

            std::ofstream stream( file, std::ios::binary | std::ios::trunc );

            if ( !stream || !stream.write( bytes.data(), static_cast<std::streamsize>( bytes.size() ) ) )
                throw std::runtime_error( std::format( "'{}' could not be written", given ) );

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

    private:

        /// Where a path the model gave lies, read against the folder. The grant replaces this one function.
        std::filesystem::path locate( std::string_view given ) const
        {
            const std::filesystem::path path = pathFromUtf8( given );

            if ( path.has_root_name() || path.has_root_directory() )
            {
                throw std::runtime_error( std::format(
                    "'{}' is not relative to the project folder; give paths as list_folder shows them", given ) );
            }

            return ( folder_ / path ).lexically_normal();
        }

        /// Every file under a folder, as a path relative to it; .git is not entered, and the walk ends at a bound.
        template <typename TVisit>
        static void visitFiles( const std::filesystem::path& root, TVisit&& visit )
        {
            std::size_t visited = 0;
            std::error_code error;
            std::filesystem::recursive_directory_iterator walk(
                root, std::filesystem::directory_options::skip_permission_denied, error );

            for ( ; !error && walk != std::filesystem::recursive_directory_iterator(); walk.increment( error ) )
            {
                std::error_code kind_error;

                if ( walk->is_directory( kind_error ) )
                {
                    if ( walk->path().filename() == ".git" )
                        walk.disable_recursion_pending();

                    continue;
                }

                if ( !walk->is_regular_file( kind_error ) )
                    continue;

                if ( ++visited > kMostFilesVisited )
                    return;

                visit( walk->path().lexically_relative( root ) );
            }
        }

        std::filesystem::path folder_;
    };
}
