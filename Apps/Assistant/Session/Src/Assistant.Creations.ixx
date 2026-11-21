/**
 * @file Assistant.Creations.ixx
 * @brief What Mila made in a chat and where it came from, read from the chat's stored calls, and the file each write
 *        left on disk (Specifications/Apps/AssistantInterface.md 4.9 and 8).
 */

export module Assistant.Creations;

import std;
import nlohmann.json;

namespace Mila::Assistant
{
    /// Text files larger than this are named in the drawer and not shown.
    export constexpr std::uintmax_t kMostBytesShown = 256 * 1024;

    /// The path a file tool's call wrote, as the call named it, or nothing when it failed, did not run or named none.
    export std::optional<std::string> writtenPath( const nlohmann::json& call );

    /**
     * @brief The file as a write left it: `{location, sha256, size, modified}`, or null when it cannot be read.
     *
     * `location` is the full path in UTF-8. `modified` is the file system's own count for the last write, compared with
     * a later reading of the same file on the same computer and never shown.
     */
    export nlohmann::json fileRecord( const std::filesystem::path& file );

    /**
     * @brief How the file a write left compares with the disk now: `"same"`, `"changed"` or `"missing"`, or null for a
     *        write stored before writes kept a record.
     *
     * The size and modified time are compared first, and the file is hashed only when its size is the same and its
     * time is not, so drawing the overview reads no file that has not been touched.
     */
    export nlohmann::json diskState( const nlohmann::json& written );

    /**
     * @brief One file a chat wrote, with where it came from, or nothing when no call in it wrote @p path.
     *
     * Returns `{kind: "file", conversation_id, title, project, path, location, made, writes, read_before,
     * read_before_total, exposed}`. `writes` holds every write of the file, oldest first, each `{call_id, tool, time,
     * model, step, sha256}`; `made` is the first. `read_before` holds the latest 20 calls that read before the first
     * write, each `{tool, what, time, exposes}`, and `read_before_total` how many there were. `exposed` is whether one of
     * them read from outside. Anything the chat stored before a field existed is null in it.
     */
    export std::optional<nlohmann::json> fileCreation( const nlohmann::json& conversation, std::string_view path );

    /**
     * @brief A reply the person kept, or nothing when message @p index of the chat is not a kept reply.
     *
     * Returns `{kind: "reply", conversation_id, title, project, message, kept, model, text, read_before,
     * read_before_total, exposed}`, `read_before` as for a file: the calls before the reply and in its own turn.
     */
    export std::optional<nlohmann::json> replyCreation( const nlohmann::json& conversation, std::size_t index );

    /**
     * @brief A file's text for the drawer: `{content, content_kind}`, `content_kind` `"markdown"` or `"text"`; both
     *        null for a file that is missing, larger than kMostBytesShown, or not UTF-8 text.
     */
    export nlohmann::json showableContent( const std::filesystem::path& file );
}
