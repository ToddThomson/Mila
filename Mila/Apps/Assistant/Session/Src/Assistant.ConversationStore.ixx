/**
 * @file Assistant.ConversationStore.ixx
 * @brief The Mila app's past conversations: one JSON file each, in a folder of the person's own (MilaApp.md 6.3).
 */

export module Assistant.ConversationStore;

import std;
import nlohmann.json;

import Mila;

namespace Mila::Assistant
{
    /**
     * @brief One JSON file per conversation, named by its id.
     *
     * A file holds the conversation as the page shows it (`messages`, MilaApp.md 4.3) and as the model read it
     * (`turns`), with `id`, `title`, `model`, `created` and `updated`. Ids arrive from the page, so one is used as a
     * file name only when it is made of the characters this store gives ids; any other names no conversation.
     */
    export class ConversationStore
    {
    public:

        explicit ConversationStore( std::filesystem::path directory )
            : directory_( std::move( directory ) )
        {
        }

        /**
         * @brief Where conversations live when the window names no folder: the app's own folder, Mila, in the
         *        person's local application data.
         *
         * On Windows the model store keeps its own folders there too (ModelStore's resolveStoreRoot), beside this
         * one and never holding it, so the store's work never reaches a conversation.
         */
        static std::filesystem::path defaultDirectory()
        {
            using Mila::Distribution::readEnvironmentVariable;

            if ( const auto local_app_data = readEnvironmentVariable( "LOCALAPPDATA" ) )
            {
                return std::filesystem::path( *local_app_data ) / "Mila" / "Conversations";
            }

            if ( const auto xdg_data = readEnvironmentVariable( "XDG_DATA_HOME" ) )
            {
                return std::filesystem::path( *xdg_data ) / "mila" / "conversations";
            }

            if ( const auto home = readEnvironmentVariable( "HOME" ) )
            {
                return std::filesystem::path( *home ) / ".local" / "share" / "mila" / "conversations";
            }

            return std::filesystem::temp_directory_path() / "mila" / "conversations";
        }

        const std::filesystem::path& directory() const noexcept { return directory_; }

        /// A new id, unique in this folder: the creation time to the second, then random hex.
        std::string newIdentifier() const
        {
            const auto now = std::chrono::floor<std::chrono::seconds>( std::chrono::system_clock::now() );

            std::random_device device;
            std::uniform_int_distribution<std::uint32_t> distribution( 0, 0xFFFFFF );

            return std::format( "conversation-{:%Y%m%d%H%M%S}-{:06x}", now, distribution( device ) );
        }

        /// Every stored conversation as `conversations.list` names it, the most recently updated first.
        nlohmann::json list() const
        {
            nlohmann::json items = nlohmann::json::array();

            std::error_code missing;

            if ( !std::filesystem::is_directory( directory_, missing ) )
            {
                return items;
            }

            for ( const auto& entry : std::filesystem::directory_iterator( directory_, missing ) )
            {
                if ( entry.path().extension() != ".json" )
                {
                    continue;
                }

                const std::optional<nlohmann::json> conversation = read( entry.path() );

                if ( !conversation.has_value() )
                {
                    continue;
                }

                // A conversation stored before activities was asked and explored, with no folder (MilaApp.md 4.1).
                items.push_back( {
                    { "id", conversation->value( "id", std::string{} ) },
                    { "title", conversation->value( "title", std::string{} ) },
                    { "updated", conversation->value( "updated", std::string{} ) },
                    { "model", conversation->value( "model", std::string{} ) },
                    { "activity", conversation->value( "activity", std::string( "ask" ) ) },
                    { "folder", conversation->value( "folder", nlohmann::json( nullptr ) ) } } );
            }

            // ISO 8601 in UTC sorts as text. A JSON array's iterators are not a std::ranges range.
            std::sort( items.begin(), items.end(), []( const nlohmann::json& left, const nlohmann::json& right )
                {
                    return left[ "updated" ].get<std::string>() > right[ "updated" ].get<std::string>();
                } );

            return items;
        }

        std::optional<nlohmann::json> load( const std::string& id ) const
        {
            if ( !isStoreIdentifier( id ) )
            {
                return std::nullopt;
            }

            return read( pathOf( id ) );
        }

        /**
         * @brief Write a conversation whole, replacing what its id held.
         *
         * Written beside its final name and then renamed over it, so a crash mid-write leaves the previous version
         * rather than half a file.
         *
         * @throws std::runtime_error when the id is not one this store gave, or the folder cannot be written.
         */
        void save( const nlohmann::json& conversation ) const
        {
            const std::string id = conversation.value( "id", std::string{} );

            if ( !isStoreIdentifier( id ) )
            {
                throw std::runtime_error( std::format( "'{}' is not a conversation id this store gives", id ) );
            }

            std::filesystem::create_directories( directory_ );

            const std::filesystem::path final_path = pathOf( id );
            std::filesystem::path partial_path = final_path;
            partial_path += ".partial";

            {
                std::ofstream file( partial_path, std::ios::binary | std::ios::trunc );

                if ( !file )
                {
                    throw std::runtime_error( std::format( "cannot write {}", partial_path.string() ) );
                }

                file << conversation.dump( 2 ) << '\n';
            }

            std::filesystem::rename( partial_path, final_path );
        }

        /// @return False when no conversation has that id.
        bool rename( const std::string& id, const std::string& title ) const
        {
            std::optional<nlohmann::json> conversation = load( id );

            if ( !conversation.has_value() )
            {
                return false;
            }

            ( *conversation )[ "title" ] = title;
            save( *conversation );

            return true;
        }

        /// @return False when no conversation has that id.
        bool remove( const std::string& id ) const
        {
            if ( !isStoreIdentifier( id ) )
            {
                return false;
            }

            std::error_code failed;

            return std::filesystem::remove( pathOf( id ), failed );
        }

    private:

        /// The characters newIdentifier() writes, so an id can never reach outside the folder.
        static bool isStoreIdentifier( std::string_view id )
        {
            return !id.empty() && id.size() <= 64
                && std::ranges::all_of( id, []( char c )
                    {
                        return ( c >= 'a' && c <= 'z' ) || ( c >= '0' && c <= '9' ) || c == '-';
                    } );
        }

        std::filesystem::path pathOf( const std::string& id ) const
        {
            return directory_ / ( id + ".json" );
        }

        /// Nothing for a file that is not a conversation, which is skipped rather than allowed to hide the others.
        static std::optional<nlohmann::json> read( const std::filesystem::path& path )
        {
            std::ifstream file( path, std::ios::binary );

            if ( !file )
            {
                return std::nullopt;
            }

            try
            {
                nlohmann::json conversation = nlohmann::json::parse( file );

                if ( !conversation.is_object() || !conversation.contains( "id" ) )
                {
                    return std::nullopt;
                }

                return conversation;
            }
            catch ( const nlohmann::json::parse_error& )
            {
                return std::nullopt;
            }
        }

        std::filesystem::path directory_;
    };
}
