/**
 * @file Desktop.ProjectStore.ixx
 * @brief The workspace's projects: a folder each, named by its id, holding `project.json` and `history.jsonl`
 *        (AssistantInterface.md 8).
 */

export module Desktop.ProjectStore;

import std;
import nlohmann.json;
import Desktop.EventLog;

namespace Mila::Desktop
{
    /**
     * @brief One folder per project. A project's chats are not in it: a chat names its project by one field.
     *
     * `project.json` holds `v`, `id`, `name`, `purpose`, `instructions` (kept, and given to no chat until
     * AssistantInterface.md phase 7), `folders` (`[{path, access}]`, `access` `read` or `read_write`), `created` and
     * `updated`. Ids arrive from the page, so one names a folder only when it is made of the characters this store
     * gives ids; any other names no project.
     */
    export class ProjectStore
    {
    public:

        explicit ProjectStore( std::filesystem::path directory );

        const std::filesystem::path& directory() const noexcept { return directory_; }

        /// A new id, `project-<UTC time>-<hex>`.
        std::string newIdentifier() const;

        /// Every stored project's record, the most recently updated first. A folder that holds none is skipped.
        std::vector<nlohmann::json> list() const;

        std::optional<nlohmann::json> load( const std::string& id ) const;

        /**
         * @brief Write a project's record whole, replacing what its id held.
         *
         * @throws std::runtime_error when the id is not one this store gave, or the folder cannot be written.
         */
        void save( const nlohmann::json& project ) const;

        /// Remove the project's folder and everything in it. @return False when no project has that id.
        bool remove( const std::string& id ) const;

        /// The project's history. @throws std::invalid_argument when the id is not one this store gave.
        EventLog history( const std::string& id ) const;

    private:

        std::filesystem::path folderOf( const std::string& id ) const;

        std::filesystem::path directory_;
    };
}
