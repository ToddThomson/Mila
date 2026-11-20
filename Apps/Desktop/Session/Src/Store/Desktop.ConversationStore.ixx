/**
 * @file Desktop.ConversationStore.ixx
 * @brief Mila Assistant's conversations: one JSON file each, in the workspace's own folder (MilaDesktop.md 6.3).
 */

export module Desktop.ConversationStore;

import std;
import nlohmann.json;

namespace Mila::Desktop
{
    /**
     * @brief One JSON file per conversation, named by its id.
     *
     * A file holds the conversation as the page shows it (`messages`, MilaDesktop.md 4.3) and as the model read it
     * (`turns`), with `id`, `title`, `model`, `project` (an id, or null for a chat in no project), `created` and
     * `updated`. Ids arrive from the page, so one is used as a file name only when it is made of the characters this
     * store gives ids; any other names no conversation.
     */
    export class ConversationStore
    {
    public:

        explicit ConversationStore( std::filesystem::path directory );

        const std::filesystem::path& directory() const noexcept { return directory_; }

        /// A new id, unique in this folder: the creation time to the second, then random hex.
        std::string newIdentifier() const;

        /// Every stored conversation as `conversations.list` names it, the most recently updated first.
        nlohmann::json list() const;

        std::optional<nlohmann::json> load( const std::string& id ) const;

        /// Every stored conversation whole, in no order, for a search through what they say (AssistantInterface.md 5.2).
        std::vector<nlohmann::json> loadAll() const;

        /**
         * @brief Write a conversation whole, replacing what its id held.
         *
         * @throws std::runtime_error when the id is not one this store gave, or the folder cannot be written.
         */
        void save( const nlohmann::json& conversation ) const;

        /// @return False when no conversation has that id.
        bool rename( const std::string& id, const std::string& title ) const;

        /// @return False when no conversation has that id.
        bool remove( const std::string& id ) const;

    private:

        std::filesystem::path pathOf( const std::string& id ) const;

        /// Nothing for a file that is not a conversation, which is skipped rather than allowed to hide the others.
        static std::optional<nlohmann::json> read( const std::filesystem::path& path );

        std::filesystem::path directory_;
    };
}
