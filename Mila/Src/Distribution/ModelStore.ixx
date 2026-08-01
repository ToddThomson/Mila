/**
 * @file ModelStore.ixx
 * @brief The local store of installed models: content-addressed blobs plus the records that name them.
 *
 * A blob's path is its digest, so a file present in the store has been verified and an interrupted
 * transfer cannot be mistaken for a complete one. Records are what make the store listable, and what
 * removal is refcounted against. See Specifications/ModelDistribution.md.
 */

module;
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <ios>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

export module Distribution.ModelStore;

import nlohmann.json;
import Distribution.Sha256;
import Distribution.Environment;

namespace Mila::Distribution
{
    /**
     * @brief What a fetch attempt reported back to the store.
     *
     * Deliberately narrower than HttpResult: the store cares about three outcomes, and keeping it
     * decoupled from the HTTP client is what makes it testable offline.
     */
    export enum class FetchOutcome
    {
        Complete,      ///< The requested byte range arrived in full.
        RangeIgnored,  ///< A resume was requested and the server sent the whole file.
        Failed         ///< Anything else; message carries the detail.
    };

    export struct FetchReport
    {
        FetchOutcome outcome{ FetchOutcome::Failed };
        std::string message;
    };

    /**
     * @brief Fetches one blob from resume_from, handing each chunk to sink.
     *
     * The store supplies the sink so it can hash and write in one pass. What is being fetched
     * is the fetcher's business: the store deals in digests, and a URL scheme belongs to
     * whichever hub produced the fetcher.
     */
    export using BlobFetcher = std::function<FetchReport(
        uint64_t resume_from,
        const std::function<bool( const char*, size_t )>& sink )>;

    /**
     * @brief Resolve the store root, first match wins.
     *
     * MILA_CACHE_DIR, then the platform user cache. An explicit override exists because a
     * multi-gigabyte store often does not belong on the same volume as the user profile.
     */
    export inline std::filesystem::path resolveStoreRoot()
    {
        if ( const auto explicit_root = readEnvironmentVariable( "MILA_CACHE_DIR" ) )
        {
            return std::filesystem::path( *explicit_root );
        }

        if ( const auto local_app_data = readEnvironmentVariable( "LOCALAPPDATA" ) )
        {
            return std::filesystem::path( *local_app_data ) / "Mila" / "models";
        }

        if ( const auto xdg_cache = readEnvironmentVariable( "XDG_CACHE_HOME" ) )
        {
            return std::filesystem::path( *xdg_cache ) / "mila" / "models";
        }

        if ( const auto home = readEnvironmentVariable( "HOME" ) )
        {
            return std::filesystem::path( *home ) / ".cache" / "mila" / "models";
        }

        return std::filesystem::temp_directory_path() / "mila" / "models";
    }

    /**
     * @brief One file composing a model, as the manifest declares it.
     *
     * `path` is the name the file has at its origin, kept for diagnostics and for republishing; the
     * bytes live at the digest.
     */
    export struct ModelFile
    {
        std::string role;
        std::string path;
        std::string sha256;
        uint64_t bytes{ 0 };
    };

    /**
     * @brief An installed model: its manifest variant, plus how it came to be here.
     *
     * Owner and repository are implied by the record's location and stored anyway, so a record
     * copied out of the store still says what it describes.
     */
    export struct ModelRecord
    {
        std::string owner;
        std::string repository;
        std::string variant;

        std::string architecture;
        std::string weight_quantization;
        std::string minimum_mila_version;

        /// Empty for a model installed from a local package rather than fetched.
        std::string hub;

        /// The resolved commit, not the ref that was asked for. Empty for a local install.
        std::string revision;

        std::string installed_at;

        std::vector<ModelFile> files;

        std::string coordinate() const
        {
            std::string text = owner + "/" + repository;

            if ( !variant.empty() )
            {
                text += ":" + variant;
            }

            return text;
        }

        const ModelFile* file( std::string_view role ) const
        {
            for ( const auto& entry : files )
            {
                if ( entry.role == role )
                {
                    return &entry;
                }
            }

            return nullptr;
        }
    };

    /**
     * @brief A record together with where its bytes actually are.
     *
     * `bytes_on_disk` sums the model's own files. A blob shared between variants is counted in
     * each, so the per-model figures deliberately do not sum to the store total.
     */
    export struct StoredModel
    {
        ModelRecord record;

        std::filesystem::path weights_path;
        std::filesystem::path tokenizer_path;

        /// False when a declared file is missing from blobs/, which makes the model unloadable.
        bool complete{ false };

        uint64_t bytes_on_disk{ 0 };
    };

    export struct RemovalReport
    {
        /// Records unlinked.
        int records_removed{ 0 };

        /// Blobs no surviving record referenced.
        int blobs_removed{ 0 };

        /// Rejected transfers and abandoned locks, which belong to no record at all.
        int files_removed{ 0 };

        uint64_t bytes_reclaimed{ 0 };

        /// Paths the platform refused to delete, most often a blob still mapped by a live process.
        std::vector<std::string> retained;
    };

    export struct StoreUsage
    {
        uint64_t blob_bytes{ 0 };
        uint64_t reclaimable_bytes{ 0 };
        uint64_t partial_bytes{ 0 };
        int model_count{ 0 };
        int blob_count{ 0 };
    };

    export struct PruneOptions
    {
        /**
         * @brief Also discard in-flight partials.
         *
         * Off by default: a partial left by a failed transfer is good bytes that the next attempt
         * resumes onto, so deleting it silently converts a cheap retry into a full re-download.
         */
        bool discard_partials{ false };
    };

    /**
     * @brief The local store of installed models.
     *
     * Layout:
     *   models/<owner>/<repository>/<variant>.json   the records -- the index
     *   blobs/sha256-<hex>                           the content
     *   tmp/                                         in-flight transfers and their locks
     */
    export class ModelStore
    {
    public:

        explicit ModelStore( std::filesystem::path root = resolveStoreRoot() )
            : root_( std::move( root ) )
        {}

        const std::filesystem::path& root() const noexcept { return root_; }

        std::filesystem::path blobPath( const std::string& sha256_hex ) const
        {
            return root_ / "blobs" / ( "sha256-" + sha256_hex );
        }

        std::filesystem::path recordPath(
            const std::string& owner,
            const std::string& repository,
            const std::string& variant ) const
        {
            return root_ / "models" / owner / repository / ( variant + ".json" );
        }

        bool contains( const std::string& sha256_hex ) const
        {
            std::error_code ignored;

            return std::filesystem::exists( blobPath( sha256_hex ), ignored );
        }

        // -------------------------------------------------------------------
        // Content
        // -------------------------------------------------------------------

        /**
         * @brief Ensure a blob is present, fetching it if it is not.
         *
         * Resumes from whatever a previous attempt left in tmp/, which is why the partial is named
         * after the digest rather than randomly -- a retry must be able to find it. The hash is
         * recomputed over the resumed prefix before appending, because SHA-256 is sequential and
         * cannot be restored from a byte offset alone.
         *
         * A transfer lock arbitrates between processes. Chat and the inference server share one
         * store, and the deterministic partial name that makes resume possible would otherwise let
         * two of them append into a single file and interleave.
         *
         * On a digest mismatch the partial is kept under a rejected name rather than destroyed:
         * the bytes are known bad, but the byte count is the evidence that separates "altered in
         * flight" from "a length bug", and destroying it destroys the diagnosis.
         *
         * @param description What the blob is, for messages only -- typically the file's path in
         *        the repository it came from. The store never interprets it.
         *
         * @return Path to the verified blob.
         * @throws std::runtime_error on fetch failure, digest mismatch, or a lock held elsewhere.
         */
        std::filesystem::path ensureBlob(
            const std::string& description,
            const std::string& expected_sha256_hex,
            const BlobFetcher& fetcher )
        {
            const auto final_path = blobPath( expected_sha256_hex );

            std::error_code ignored;

            if ( std::filesystem::exists( final_path, ignored ) )
            {
                return final_path;
            }

            std::filesystem::create_directories( final_path.parent_path() );
            std::filesystem::create_directories( root_ / "tmp" );

            const auto partial_path =
                root_ / "tmp" / ( "sha256-" + expected_sha256_hex + ".partial" );

            TransferLock lock( root_ / "tmp" / ( "sha256-" + expected_sha256_hex + ".lock" ) );

            if ( !lock.held() )
            {
                throw std::runtime_error( std::format(
                    "ModelStore: another process is already fetching this blob.\n"
                    "  blob {}\n"
                    "  lock {}\n"
                    "If no transfer is running, delete the lock file and retry.",
                    description, lock.path().string() ) );
            }

            // Re-check under the lock: the process that held it may have been fetching exactly
            // this blob and finished while this one waited on the open.
            if ( std::filesystem::exists( final_path, ignored ) )
            {
                return final_path;
            }

            Sha256 hash;
            uint64_t resume_from = 0;

            if ( std::filesystem::exists( partial_path, ignored ) )
            {
                resume_from = replayPartialIntoHash( partial_path, hash );
            }

            // Append so a resumed transfer extends the partial rather than truncating it.
            std::ofstream output( partial_path, std::ios::binary
                | ( resume_from > 0 ? std::ios::app : std::ios::trunc ) );

            if ( !output.is_open() )
            {
                throw std::runtime_error(
                    "ModelStore: cannot open " + partial_path.string() + " for writing" );
            }

            auto sink = [&]( const char* data, size_t length ) -> bool
                {
                    output.write( data, static_cast<std::streamsize>( length ) );

                    if ( !output )
                    {
                        return false;
                    }

                    hash.update( data, length );

                    return true;
                };

            FetchReport report = fetcher( resume_from, sink );

            // Closed before the digest is judged: a buffered tail still in the stream would
            // make the file on disk shorter than the bytes that were hashed.
            output.close();

            if ( report.outcome == FetchOutcome::RangeIgnored )
            {
                // The server sent the whole file despite the Range header, so the partial now
                // holds a prefix followed by a second copy from byte zero. Nothing salvageable.
                std::filesystem::remove( partial_path, ignored );

                throw std::runtime_error( std::format(
                    "ModelStore: server ignored the resume request for {}; "
                    "discarded the partial, retry from the start", description ) );
            }

            if ( report.outcome != FetchOutcome::Complete )
            {
                // Kept, not deleted: the bytes so far are good and a retry can resume onto them.
                throw std::runtime_error( std::format(
                    "ModelStore: fetch of {} failed: {}", description, report.message ) );
            }

            const std::string actual = hash.finish();

            if ( actual != expected_sha256_hex )
            {
                std::error_code size_error;
                const auto received = std::filesystem::file_size( partial_path, size_error );

                const auto rejected_path = partial_path.string() + ".rejected";
                std::filesystem::remove( rejected_path, ignored );
                std::filesystem::rename( partial_path, rejected_path, ignored );

                throw std::runtime_error( std::format(
                    "ModelStore: digest mismatch for {}\n"
                    "  expected sha256 {}\n"
                    "  actual   sha256 {}\n"
                    "  bytes received  {}\n"
                    "  kept for inspection at {}",
                    description, expected_sha256_hex, actual,
                    size_error ? 0 : received, rejected_path ) );
            }

            // The rename is what publishes the blob. Until it lands, no reader can see a path
            // that implies verification.
            std::error_code rename_error;
            std::filesystem::rename( partial_path, final_path, rename_error );

            if ( rename_error )
            {
                // A concurrent fetch of the same digest may have won the race; its bytes are
                // equally verified, so an existing target is success rather than failure.
                if ( std::filesystem::exists( final_path, ignored ) )
                {
                    std::filesystem::remove( partial_path, ignored );

                    return final_path;
                }

                throw std::runtime_error( std::format(
                    "ModelStore: cannot publish blob {}: {}",
                    final_path.string(), rename_error.message() ) );
            }

            return final_path;
        }

        // -------------------------------------------------------------------
        // Records
        // -------------------------------------------------------------------

        /**
         * @brief Write a record, stamping the install time.
         *
         * Written to tmp/ and renamed, because a peer process may be listing the store while this
         * one installs, and a half-written record must never be readable.
         */
        void writeRecord( ModelRecord record )
        {
            if ( record.owner.empty() || record.repository.empty() || record.variant.empty() )
            {
                throw std::runtime_error(
                    "ModelStore: a record needs an owner, a repository and a variant" );
            }

            if ( record.installed_at.empty() )
            {
                record.installed_at = utcTimestamp();
            }

            const auto destination = recordPath( record.owner, record.repository, record.variant );

            std::filesystem::create_directories( destination.parent_path() );
            std::filesystem::create_directories( root_ / "tmp" );

            const std::string text = toJson( record ).dump( 2 );

            const auto staging = root_ / "tmp" / std::format( "record-{}-{}-{}.json",
                record.owner, record.repository, record.variant );

            {
                std::ofstream output( staging, std::ios::binary | std::ios::trunc );

                if ( !output.is_open() )
                {
                    throw std::runtime_error(
                        "ModelStore: cannot open " + staging.string() + " for writing" );
                }

                output.write( text.data(), static_cast<std::streamsize>( text.size() ) );
                output.close();

                if ( !output )
                {
                    throw std::runtime_error(
                        "ModelStore: short write for " + staging.string() );
                }
            }

            std::error_code rename_error;
            std::filesystem::remove( destination, rename_error );
            std::filesystem::rename( staging, destination, rename_error );

            if ( rename_error )
            {
                std::error_code ignored;
                std::filesystem::remove( staging, ignored );

                throw std::runtime_error( std::format(
                    "ModelStore: cannot publish record {}: {}",
                    destination.string(), rename_error.message() ) );
            }
        }

        std::optional<ModelRecord> readRecord(
            const std::string& owner,
            const std::string& repository,
            const std::string& variant ) const
        {
            return readRecordFile( recordPath( owner, repository, variant ) );
        }

        /**
         * @brief Every installed model, in directory order.
         *
         * A record whose blobs have gone missing is reported with `complete` false rather than
         * omitted: a store that silently hides a broken entry cannot be repaired by its owner.
         */
        std::vector<StoredModel> list() const
        {
            std::vector<StoredModel> models;

            const auto models_root = root_ / "models";

            std::error_code ignored;

            if ( !std::filesystem::exists( models_root, ignored ) )
            {
                return models;
            }

            for ( const auto& entry :
                std::filesystem::recursive_directory_iterator( models_root, ignored ) )
            {
                if ( !entry.is_regular_file() || entry.path().extension() != ".json" )
                {
                    continue;
                }

                if ( auto record = readRecordFile( entry.path() ) )
                {
                    models.push_back( describe( *record ) );
                }
            }

            return models;
        }

        /**
         * @brief Where an installed model's files are, or nothing.
         *
         * Never consults a hub and never accepts a path: the store is the only thing a load reads
         * from. A record whose blobs are incomplete resolves to nothing, because a caller that
         * receives a path expects bytes behind it.
         */
        std::optional<StoredModel> locate(
            const std::string& owner,
            const std::string& repository,
            const std::string& variant ) const
        {
            auto record = readRecord( owner, repository, variant );

            if ( !record.has_value() )
            {
                return std::nullopt;
            }

            StoredModel model = describe( *record );

            if ( !model.complete )
            {
                return std::nullopt;
            }

            return model;
        }

        /**
         * @brief Resolve a record against the blob store without requiring it to be complete.
         */
        StoredModel describe( const ModelRecord& record ) const
        {
            StoredModel model;
            model.record = record;
            model.complete = !record.files.empty();

            std::error_code ignored;

            for ( const auto& file : record.files )
            {
                const auto path = blobPath( file.sha256 );

                if ( std::filesystem::exists( path, ignored ) )
                {
                    model.bytes_on_disk += std::filesystem::file_size( path, ignored );
                }
                else
                {
                    model.complete = false;

                    continue;
                }

                if ( file.role == "weights" )
                {
                    model.weights_path = path;
                }
                else if ( file.role == "tokenizer" )
                {
                    model.tokenizer_path = path;
                }
            }

            if ( model.weights_path.empty() )
            {
                model.complete = false;
            }

            return model;
        }

        // -------------------------------------------------------------------
        // Removal
        // -------------------------------------------------------------------

        /**
         * @brief Remove one installed model, then reclaim what nothing else references.
         *
         * The sweep is what makes this safe. Deduplication means a tokenizer blob may back several
         * variants, so removal cannot delete a model's files simply because that model is going.
         */
        RemovalReport remove(
            const std::string& owner,
            const std::string& repository,
            const std::string& variant )
        {
            RemovalReport report;

            const auto record_path = recordPath( owner, repository, variant );

            std::error_code ignored;

            if ( !std::filesystem::exists( record_path, ignored ) )
            {
                return report;
            }

            std::error_code remove_error;
            std::filesystem::remove( record_path, remove_error );

            if ( remove_error )
            {
                report.retained.push_back( record_path.string() );

                return report;
            }

            report.records_removed = 1;

            removeEmptyParents( record_path.parent_path() );

            const RemovalReport swept = prune();

            report.blobs_removed = swept.blobs_removed;
            report.bytes_reclaimed = swept.bytes_reclaimed;
            report.retained.insert(
                report.retained.end(), swept.retained.begin(), swept.retained.end() );

            return report;
        }

        /**
         * @brief Reclaim blobs no record names, rejected transfers, and abandoned locks.
         *
         * Mark-and-sweep over the record tree is exact and costs a directory walk: records are
         * kilobytes, and the alternative -- a reference count maintained by hand -- is a number
         * that can be wrong.
         */
        RemovalReport prune( const PruneOptions& options = {} )
        {
            RemovalReport report;

            std::set<std::string> referenced;

            for ( const auto& model : list() )
            {
                for ( const auto& file : model.record.files )
                {
                    referenced.insert( file.sha256 );
                }
            }

            std::error_code ignored;

            const auto blobs_root = root_ / "blobs";

            if ( std::filesystem::exists( blobs_root, ignored ) )
            {
                for ( const auto& entry :
                    std::filesystem::directory_iterator( blobs_root, ignored ) )
                {
                    if ( !entry.is_regular_file() )
                    {
                        continue;
                    }

                    const std::string name = entry.path().filename().string();

                    if ( !name.starts_with( "sha256-" ) )
                    {
                        continue;
                    }

                    if ( referenced.contains( name.substr( 7 ) ) )
                    {
                        continue;
                    }

                    reclaim( entry.path(), report, report.blobs_removed );
                }
            }

            const auto tmp_root = root_ / "tmp";

            if ( std::filesystem::exists( tmp_root, ignored ) )
            {
                for ( const auto& entry :
                    std::filesystem::directory_iterator( tmp_root, ignored ) )
                {
                    if ( !entry.is_regular_file() )
                    {
                        continue;
                    }

                    const std::string name = entry.path().filename().string();

                    if ( name.ends_with( ".rejected" ) )
                    {
                        reclaim( entry.path(), report, report.files_removed );
                    }
                    else if ( name.ends_with( ".lock" ) && isAbandoned( entry.path() ) )
                    {
                        // A lock outliving any plausible transfer is a crashed process, not a
                        // peer. Reclaimed on a generous clock so a slow but live fetch is safe.
                        reclaim( entry.path(), report, report.files_removed );
                    }
                    else if ( name.ends_with( ".partial" ) && options.discard_partials )
                    {
                        reclaim( entry.path(), report, report.files_removed );
                    }
                }
            }

            return report;
        }

        /**
         * @brief What the store holds and what could be reclaimed.
         */
        StoreUsage usage() const
        {
            StoreUsage totals;

            std::set<std::string> referenced;

            for ( const auto& model : list() )
            {
                ++totals.model_count;

                for ( const auto& file : model.record.files )
                {
                    referenced.insert( file.sha256 );
                }
            }

            std::error_code ignored;

            const auto blobs_root = root_ / "blobs";

            if ( std::filesystem::exists( blobs_root, ignored ) )
            {
                for ( const auto& entry :
                    std::filesystem::directory_iterator( blobs_root, ignored ) )
                {
                    if ( !entry.is_regular_file() )
                    {
                        continue;
                    }

                    const std::string name = entry.path().filename().string();

                    if ( !name.starts_with( "sha256-" ) )
                    {
                        continue;
                    }

                    const uint64_t bytes = std::filesystem::file_size( entry.path(), ignored );

                    ++totals.blob_count;
                    totals.blob_bytes += bytes;

                    if ( !referenced.contains( name.substr( 7 ) ) )
                    {
                        totals.reclaimable_bytes += bytes;
                    }
                }
            }

            const auto tmp_root = root_ / "tmp";

            if ( std::filesystem::exists( tmp_root, ignored ) )
            {
                for ( const auto& entry :
                    std::filesystem::directory_iterator( tmp_root, ignored ) )
                {
                    if ( !entry.is_regular_file() )
                    {
                        continue;
                    }

                    const std::string name = entry.path().filename().string();
                    const uint64_t bytes = std::filesystem::file_size( entry.path(), ignored );

                    if ( name.ends_with( ".partial" ) )
                    {
                        totals.partial_bytes += bytes;
                    }
                    else if ( name.ends_with( ".rejected" ) )
                    {
                        totals.reclaimable_bytes += bytes;
                    }
                }
            }

            return totals;
        }

    private:

        /**
         * @brief An exclusively created lock file, released on destruction.
         *
         * std::ios::noreplace is C++23's open-if-absent: it fails rather than truncates when the
         * file exists, which is the whole mechanism -- exactly one process can be transferring a
         * given digest.
         */
        class TransferLock
        {
        public:

            explicit TransferLock( std::filesystem::path path )
                : path_( std::move( path ) )
            {
                std::ofstream file( path_, std::ios::binary | std::ios::noreplace );

                held_ = file.is_open();
            }

            ~TransferLock()
            {
                if ( held_ )
                {
                    std::error_code ignored;
                    std::filesystem::remove( path_, ignored );
                }
            }

            TransferLock( const TransferLock& ) = delete;
            TransferLock& operator=( const TransferLock& ) = delete;

            bool held() const noexcept { return held_; }
            const std::filesystem::path& path() const noexcept { return path_; }

        private:

            std::filesystem::path path_;
            bool held_{ false };
        };

        static std::string utcTimestamp()
        {
            const auto now = std::chrono::time_point_cast<std::chrono::seconds>(
                std::chrono::system_clock::now() );

            return std::format( "{:%Y-%m-%dT%H:%M:%SZ}", now );
        }

        /**
         * @brief True when a lock is old enough that no live transfer could still hold it.
         */
        static bool isAbandoned( const std::filesystem::path& path )
        {
            std::error_code ignored;

            const auto written = std::filesystem::last_write_time( path, ignored );

            if ( ignored )
            {
                return false;
            }

            return ( std::filesystem::file_time_type::clock::now() - written )
                > std::chrono::hours( 24 );
        }

        static void reclaim(
            const std::filesystem::path& path, RemovalReport& report, int& counter )
        {
            std::error_code size_error;
            const uint64_t bytes = std::filesystem::file_size( path, size_error );

            std::error_code remove_error;
            std::filesystem::remove( path, remove_error );

            if ( remove_error )
            {
                // Windows refuses to delete a mapped file and POSIX does not; reporting the
                // platform's answer beats pretending both behave the same.
                report.retained.push_back( path.string() );

                return;
            }

            ++counter;

            if ( !size_error )
            {
                report.bytes_reclaimed += bytes;
            }
        }

        void removeEmptyParents( std::filesystem::path directory ) const
        {
            const auto models_root = root_ / "models";

            std::error_code ignored;

            while ( directory != models_root && directory.has_relative_path() )
            {
                if ( !std::filesystem::is_empty( directory, ignored ) || ignored )
                {
                    return;
                }

                std::error_code remove_error;
                std::filesystem::remove( directory, remove_error );

                if ( remove_error )
                {
                    return;
                }

                directory = directory.parent_path();
            }
        }

        static std::optional<ModelRecord> readRecordFile( const std::filesystem::path& path )
        {
            std::ifstream input( path, std::ios::binary );

            if ( !input.is_open() )
            {
                return std::nullopt;
            }

            std::string text;
            char buffer[ 4096 ];

            for ( ;; )
            {
                input.read( buffer, sizeof( buffer ) );

                const auto read = static_cast<size_t>( input.gcount() );

                if ( read == 0 )
                {
                    break;
                }

                text.append( buffer, read );
            }

            nlohmann::json json = nlohmann::json::parse( text, nullptr, false );

            if ( json.is_discarded() || !json.is_object() )
            {
                return std::nullopt;
            }

            return fromJson( json );
        }

        static nlohmann::json toJson( const ModelRecord& record )
        {
            nlohmann::json files = nlohmann::json::object();

            for ( const auto& file : record.files )
            {
                files[ file.role ] = {
                    { "path", file.path },
                    { "sha256", file.sha256 },
                    { "bytes", file.bytes } };
            }

            return {
                { "manifest_version", 1 },
                { "architecture", record.architecture },
                { "variant", record.variant },
                { "weight_quantization", record.weight_quantization },
                { "minimum_mila_version", record.minimum_mila_version },
                { "files", files },
                { "installed", {
                    { "owner", record.owner },
                    { "repository", record.repository },
                    { "hub", record.hub },
                    { "revision", record.revision },
                    { "installed_at", record.installed_at } } } };
        }

        static std::optional<ModelRecord> fromJson( const nlohmann::json& json )
        {
            ModelRecord record;

            record.architecture = json.value( "architecture", std::string{} );
            record.variant = json.value( "variant", std::string{} );
            record.weight_quantization = json.value( "weight_quantization", std::string{} );
            record.minimum_mila_version = json.value( "minimum_mila_version", std::string{} );

            if ( json.contains( "installed" ) && json[ "installed" ].is_object() )
            {
                const nlohmann::json& installed = json[ "installed" ];

                record.owner = installed.value( "owner", std::string{} );
                record.repository = installed.value( "repository", std::string{} );
                record.hub = installed.value( "hub", std::string{} );
                record.revision = installed.value( "revision", std::string{} );
                record.installed_at = installed.value( "installed_at", std::string{} );
            }

            if ( json.contains( "files" ) && json[ "files" ].is_object() )
            {
                for ( const auto& entry : json[ "files" ].items() )
                {
                    const nlohmann::json& value = entry.value();

                    if ( !value.is_object() || !value.contains( "sha256" ) )
                    {
                        continue;
                    }

                    ModelFile file;
                    file.role = entry.key();
                    file.path = value.value( "path", std::string{} );
                    file.sha256 = value.value( "sha256", std::string{} );
                    file.bytes = value.value( "bytes", uint64_t{ 0 } );

                    record.files.push_back( std::move( file ) );
                }
            }

            if ( record.owner.empty() || record.repository.empty() || record.variant.empty() )
            {
                return std::nullopt;
            }

            return record;
        }

        /**
         * @brief Feed an existing partial through the hash and return its length.
         *
         * SHA-256 is sequential, so resuming a transfer means re-reading the prefix. That is
         * disk-local and far cheaper than re-downloading it.
         */
        static uint64_t replayPartialIntoHash(
            const std::filesystem::path& partial_path, Sha256& hash )
        {
            std::ifstream input( partial_path, std::ios::binary );

            if ( !input.is_open() )
            {
                return 0;
            }

            std::string buffer( 1u << 20, '\0' );
            uint64_t total = 0;

            for ( ;; )
            {
                input.read( buffer.data(), static_cast<std::streamsize>( buffer.size() ) );

                const auto read = static_cast<size_t>( input.gcount() );

                if ( read == 0 )
                {
                    break;
                }

                hash.update( buffer.data(), read );
                total += read;
            }

            return total;
        }

        std::filesystem::path root_;
    };
}
