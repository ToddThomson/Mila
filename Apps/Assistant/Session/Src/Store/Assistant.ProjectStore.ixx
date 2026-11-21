/**
 * @file Assistant.ProjectStore.ixx
 * @brief The workspace's projects: a folder each, named by its id, holding `project.json`, `understanding.json` and
 *        `history.jsonl` (AssistantInterface.md 8).
 */

export module Assistant.ProjectStore;

import std;
import nlohmann.json;
import Assistant.EventLog;

namespace Mila::Assistant
{
    /**
     * @brief One folder per project. A project's chats are not in it: a chat names its project by one field.
     *
     * `project.json` holds `v`, `id`, `name`, `purpose`, `instructions` (the person's, which end each of its chats'
     * system prompt), `folders` (`[{path, access}]`, `access` `read` or `read_write`), `created` and `updated`.
     * `understanding.json` holds `v`, `current` and `waiting`, each a note or null (Assistant.Understanding). Ids arrive
     * from the page, so one names a folder only when it is made of the characters this store gives ids; any other
     * names no project.
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

        /// The project's understanding, or an empty one when it has none or the id names no project.
        nlohmann::json understanding( const std::string& id ) const;

        /**
         * @brief Write a project's understanding whole.
         *
         * @throws std::runtime_error when the id names no stored project, or the file cannot be written.
         */
        void saveUnderstanding( const std::string& id, const nlohmann::json& understanding ) const;

    private:

        std::filesystem::path folderOf( const std::string& id ) const;

        std::filesystem::path directory_;
    };
}
