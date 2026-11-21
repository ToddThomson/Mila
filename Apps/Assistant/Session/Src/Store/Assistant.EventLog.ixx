/**
 * @file Assistant.EventLog.ixx
 * @brief A file of the workspace's events, one a line: a project's `history.jsonl` or the workspace's
 *        `activity.jsonl` (AssistantInterface.md 8).
 */

export module Assistant.EventLog;

import std;
import nlohmann.json;

namespace Mila::Assistant
{
    /**
     * @brief Events are appended in ordinary work and never edited; a deletion rewrites the file to forget a name.
     *
     * An event is an object with `time` (ISO 8601 UTC to the millisecond), `who` (`you` or `mila`) and `kind`, and
     * any of `project` and `from` (project ids), `conversation` (a conversation id), `name` (what it named when it
     * happened: a project's or a chat's), `path` and `access` (a project's folder).
     */
    export class EventLog
    {
    public:

        explicit EventLog( std::filesystem::path path );

        const std::filesystem::path& path() const noexcept { return path_; }

        /// @throws std::runtime_error when the file cannot be written.
        void append( const nlohmann::json& event ) const;

        /// Every readable event, oldest first.
        std::vector<nlohmann::json> read() const;

        /**
         * @brief Forget a deleted project or conversation in every event that names it.
         *
         * The event keeps its kind and time. It loses the id, and the name and folder it carried when the id was the
         * thing the event was about: a chat's title in an event about that chat, a project's name or folder in an
         * event about that project. The file is rewritten only when an event changed.
         *
         * @throws std::runtime_error when the file cannot be written.
         */
        void forget( std::string_view id ) const;

    private:

        std::filesystem::path path_;
    };
}
