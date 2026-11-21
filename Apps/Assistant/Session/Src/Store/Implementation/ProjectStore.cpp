/**
 * @file ProjectStore.cpp
 * @brief The bodies of ProjectStore's members; the interface is Assistant.ProjectStore.ixx.
 */

module Assistant.ProjectStore;

import std;
import nlohmann.json;
import Assistant.EventLog;
import Assistant.StoreFile;

namespace Mila::Assistant
{
    namespace
    {
        constexpr std::string_view kRecordName = "project.json";
        constexpr std::string_view kHistoryName = "history.jsonl";
        constexpr std::string_view kUnderstandingName = "understanding.json";

        nlohmann::json noUnderstanding()
        {
            return { { "v", 1 }, { "current", nullptr }, { "waiting", nullptr } };
        }
    }

    ProjectStore::ProjectStore( std::filesystem::path directory )
        : directory_( std::move( directory ) )
    {
    }

    std::string ProjectStore::newIdentifier() const
    {
        return newStoreIdentifier( "project" );
    }

    std::vector<nlohmann::json> ProjectStore::list() const
    {
        std::vector<nlohmann::json> projects;
        std::error_code missing;

        if ( !std::filesystem::is_directory( directory_, missing ) )
        {
            return projects;
        }

        for ( const auto& entry : std::filesystem::directory_iterator( directory_, missing ) )
        {
            const std::string id = entry.path().filename().string();

            if ( !isStoreIdentifier( id ) )
            {
                continue;
            }

            if ( std::optional<nlohmann::json> project = load( id ) )
            {
                projects.push_back( std::move( *project ) );
            }
        }

        // ISO 8601 in UTC sorts as text.
        std::ranges::sort( projects, []( const nlohmann::json& left, const nlohmann::json& right )
            {
                return left.value( "updated", std::string{} ) > right.value( "updated", std::string{} );
            } );

        return projects;
    }

    std::optional<nlohmann::json> ProjectStore::load( const std::string& id ) const
    {
        if ( !isStoreIdentifier( id ) )
        {
            return std::nullopt;
        }

        std::optional<nlohmann::json> project = readStoreObject( folderOf( id ) / kRecordName );

        // A record that names another id is not this project's.
        if ( !project.has_value() || project->value( "id", std::string{} ) != id )
        {
            return std::nullopt;
        }

        return project;
    }

    void ProjectStore::save( const nlohmann::json& project ) const
    {
        const std::string id = project.value( "id", std::string{} );

        if ( !isStoreIdentifier( id ) )
        {
            throw std::runtime_error( std::format( "'{}' is not a project id this store gives", id ) );
        }

        writeStoreFile( folderOf( id ) / kRecordName, project.dump( 2 ) + "\n" );
    }

    bool ProjectStore::remove( const std::string& id ) const
    {
        if ( !isStoreIdentifier( id ) )
        {
            return false;
        }

        std::error_code failed;

        return std::filesystem::remove_all( folderOf( id ), failed ) > 0;
    }

    EventLog ProjectStore::history( const std::string& id ) const
    {
        if ( !isStoreIdentifier( id ) )
        {
            throw std::invalid_argument( std::format( "'{}' is not a project id this store gives", id ) );
        }

        return EventLog( folderOf( id ) / kHistoryName );
    }

    nlohmann::json ProjectStore::understanding( const std::string& id ) const
    {
        if ( !load( id ).has_value() )
        {
            return noUnderstanding();
        }

        std::optional<nlohmann::json> stored = readStoreObject( folderOf( id ) / kUnderstandingName );

        if ( !stored.has_value() )
        {
            return noUnderstanding();
        }

        for ( const char* where : { "current", "waiting" } )
        {
            if ( !stored->contains( where ) || !( *stored )[ where ].is_object() )
            {
                ( *stored )[ where ] = nullptr;
            }
        }

        return *stored;
    }

    void ProjectStore::saveUnderstanding( const std::string& id, const nlohmann::json& understanding ) const
    {
        // A project deleted meanwhile is not brought back by its note.
        if ( !load( id ).has_value() )
        {
            throw std::runtime_error( std::format( "'{}' names no stored project", id ) );
        }

        writeStoreFile( folderOf( id ) / kUnderstandingName, understanding.dump( 2 ) + "\n" );
    }

    std::filesystem::path ProjectStore::folderOf( const std::string& id ) const
    {
        return directory_ / id;
    }
}
