/**
 * @file ConversationStore.cpp
 * @brief The bodies of ConversationStore's members; the interface is Desktop.ConversationStore.ixx.
 */

module Desktop.ConversationStore;

import std;
import nlohmann.json;
import Desktop.ConversationSummary;
import Desktop.StoreFile;

namespace Mila::Desktop
{
    ConversationStore::ConversationStore( std::filesystem::path directory )
        : directory_( std::move( directory ) )
    {
    }

    std::string ConversationStore::newIdentifier() const
    {
        return newStoreIdentifier( "conversation" );
    }

    nlohmann::json ConversationStore::list() const
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

            // A chat's activity, which conversations stored before AssistantInterface.md phase 3 carry, is not read. One
            // stored before projects is in none.
            nlohmann::json item = {
                { "id", conversation->value( "id", std::string{} ) },
                { "title", conversation->value( "title", std::string{} ) },
                { "created", conversation->value( "created", std::string{} ) },
                { "updated", conversation->value( "updated", std::string{} ) },
                { "model", conversation->value( "model", std::string{} ) },
                { "folder", conversation->value( "folder", nlohmann::json( nullptr ) ) },
                { "project", conversation->value( "project", nlohmann::json( nullptr ) ) },
                { "folder_from_project", conversation->value( "folder_from_project", false ) } };

            item.update( summarizeConversation( *conversation ) );
            items.push_back( std::move( item ) );
        }

        // ISO 8601 in UTC sorts as text. A JSON array's iterators are not a std::ranges range.
        std::sort( items.begin(), items.end(), []( const nlohmann::json& left, const nlohmann::json& right )
            {
                return left[ "updated" ].get<std::string>() > right[ "updated" ].get<std::string>();
            } );

        return items;
    }

    std::vector<nlohmann::json> ConversationStore::loadAll() const
    {
        std::vector<nlohmann::json> conversations;
        std::error_code missing;

        if ( !std::filesystem::is_directory( directory_, missing ) )
        {
            return conversations;
        }

        for ( const auto& entry : std::filesystem::directory_iterator( directory_, missing ) )
        {
            if ( entry.path().extension() != ".json" )
            {
                continue;
            }

            if ( std::optional<nlohmann::json> conversation = read( entry.path() ) )
            {
                conversations.push_back( std::move( *conversation ) );
            }
        }

        return conversations;
    }

    std::optional<nlohmann::json> ConversationStore::load( const std::string& id ) const
    {
        if ( !isStoreIdentifier( id ) )
        {
            return std::nullopt;
        }

        return read( pathOf( id ) );
    }

    void ConversationStore::save( const nlohmann::json& conversation ) const
    {
        const std::string id = conversation.value( "id", std::string{} );

        if ( !isStoreIdentifier( id ) )
        {
            throw std::runtime_error( std::format( "'{}' is not a conversation id this store gives", id ) );
        }

        writeStoreFile( pathOf( id ), conversation.dump( 2 ) + "\n" );
    }

    bool ConversationStore::rename( const std::string& id, const std::string& title ) const
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

    bool ConversationStore::remove( const std::string& id ) const
    {
        if ( !isStoreIdentifier( id ) )
        {
            return false;
        }

        std::error_code failed;

        return std::filesystem::remove( pathOf( id ), failed );
    }

    std::filesystem::path ConversationStore::pathOf( const std::string& id ) const
    {
        return directory_ / ( id + ".json" );
    }

    std::optional<nlohmann::json> ConversationStore::read( const std::filesystem::path& path )
    {
        std::optional<nlohmann::json> conversation = readStoreObject( path );

        if ( !conversation.has_value() || !conversation->contains( "id" ) )
        {
            return std::nullopt;
        }

        return conversation;
    }
}
