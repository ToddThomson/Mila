/**
 * @file Assistant.SearchKeyFile.ixx
 * @brief Where the person's web search key is kept between launches: one small file in the app's own folder.
 */

export module Assistant.SearchKeyFile;

import std;
import nlohmann.json;

namespace Mila::Assistant
{
    /**
     * @brief The web search key, in plain text in a file of the person's own (MilaApp.md 3.7).
     *
     * Plain text, as other local AI apps keep theirs, until the operating system's credential store holds it. The
     * file is never read into a conversation, a log or the page, which learns only whether a key is set.
     */
    export class SearchKeyFile
    {
    public:

        explicit SearchKeyFile( std::filesystem::path path )
            : path_( std::move( path ) )
        {
        }

        /// The key, or nothing when none is kept or the file cannot be read as one.
        [[nodiscard]] std::optional<std::string> read() const
        {
            std::ifstream file( path_, std::ios::binary );

            if ( !file )
                return std::nullopt;

            try
            {
                const nlohmann::json stored = nlohmann::json::parse( file );
                const nlohmann::json key = stored.value( "tavily_key", nlohmann::json( nullptr ) );

                if ( key.is_string() && !key.get<std::string>().empty() )
                    return key.get<std::string>();
            }
            catch ( const nlohmann::json::parse_error& )
            {
            }

            return std::nullopt;
        }

        /// @throws std::runtime_error when the file cannot be written.
        void write( const std::string& key ) const
        {
            std::filesystem::create_directories( path_.parent_path() );

            std::ofstream file( path_, std::ios::binary | std::ios::trunc );

            if ( !file || !( file << nlohmann::json{ { "tavily_key", key } }.dump() << '\n' ) )
                throw std::runtime_error( "the search key could not be saved" );
        }

        void remove() const noexcept
        {
            std::error_code ignored;
            std::filesystem::remove( path_, ignored );
        }

    private:

        std::filesystem::path path_;
    };
}
