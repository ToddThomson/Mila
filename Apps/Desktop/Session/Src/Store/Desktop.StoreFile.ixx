/**
 * @file Desktop.StoreFile.ixx
 * @brief How the workspace's files are written and read, whole records only, so that sealing them changes one place
 *        (AssistantInterface.md 8, *One layer owns the files*).
 */

export module Desktop.StoreFile;

import std;
import nlohmann.json;

namespace Mila::Desktop
{
    /**
     * @brief Replace a file's contents whole, creating its folder.
     *
     * Written beside its final name and then renamed over it, so a crash mid-write leaves the previous version rather
     * than half a file.
     *
     * @throws std::runtime_error when the file cannot be written.
     */
    export void writeStoreFile( const std::filesystem::path& path, std::string_view text );

    /// The JSON object a file holds, or nothing for a file that is missing, unreadable or not an object.
    export std::optional<nlohmann::json> readStoreObject( const std::filesystem::path& path );

    /**
     * @brief Add one record as a line at the end of a file of records, creating it and its folder.
     *
     * @throws std::runtime_error when the file cannot be written.
     */
    export void appendStoreLine( const std::filesystem::path& path, const nlohmann::json& record );

    /// Each line of a file of records that is a JSON object, in file order. A line cut short by a crash is skipped.
    export std::vector<nlohmann::json> readStoreLines( const std::filesystem::path& path );

    /**
     * @brief Rewrite a file of records whole, one per line, as writeStoreFile writes a file.
     *
     * @throws std::runtime_error when the file cannot be written.
     */
    export void writeStoreLines( const std::filesystem::path& path, const std::vector<nlohmann::json>& records );

    /// A new id for a record of this kind: the kind, the creation time to the second in UTC, then random hex.
    export std::string newStoreIdentifier( std::string_view kind );

    /// True for the characters newStoreIdentifier() writes, so an id from the page can never name a path outside.
    export bool isStoreIdentifier( std::string_view id );
}
