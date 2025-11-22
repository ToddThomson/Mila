/**
 * @file ArchiveMode.ixx
 * @brief Archive open mode for model serializers.
 *
 * Indicates whether an archive is intended for reading or writing; used to
 * guard operations that are valid only in the corresponding mode.
 */

module;
#include <string_view>

export module Serialization.OpenMode;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Mode indicating whether the archive is used for reading or writing.
     *
     * Preconditions:
     *  - The underlying `ModelSerializer` should be opened for the matching
     *    operation (openForRead for `Read`, openForWrite for `Write`) before
     *    calling archive helpers.
     *
     * Semantics:
     *  - `Read`  : permits read operations (readJson, readBlob, file queries).
     *  - `Write` : permits write operations (writeJson, writeBlob, add data).
     *
     * Example:
     * @code
     * auto serializer = std::make_unique<ZipSerializer>( ... );
     * serializer->openForWrite(...);
     * ModelArchive archive( std::move(serializer), ArchiveMode::Write );
     * archive.writeJson("meta.json", j);
     * @endcode
     */
    export enum class OpenMode
    {
        Read,
        Write
    };

    /**
     * @brief Convert ArchiveMode to string
     */
    export constexpr std::string_view archiveModeToString( OpenMode mode )
    {
        switch (mode)
        {
            case OpenMode::Read:
                return "Read";
            case OpenMode::Write:
                return "Write";
            
            default:
                return "Unknown";
        }
    }
}