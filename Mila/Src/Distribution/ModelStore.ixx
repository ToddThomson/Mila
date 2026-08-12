/**
 * @file ModelStore.ixx
 * @brief The local store of installed models: content-addressed blobs plus the records that name them.
 *
 * A blob's path is its digest, so a file present in the store has been verified and an interrupted
 * transfer cannot be mistaken for a complete one. Records are what make the store listable, and what
 * removal is refcounted against. See Specifications/ModelDistribution.md.
 */

module;
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <ios>
// Required by every importer of nlohmann.json: basic_json::create compares a unique_ptr
// against nullptr at the point of instantiation, and those operators are not reachable
// through the module.
#include <memory>
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
import Distribution.ModelManifest;
import Distribution.ModelPackage;

namespace Mila::Distribution
{
    /**
     * @brief What a fetch attempt reported back to the store.
     *
     * Deliberately narrower than HttpResult: the store cares about four outcomes, and keeping it
     * decoupled from the HTTP client is what makes it testable offline.
     */
    export enum class FetchOutcome
    {
        Complete,      ///< The requested byte range arrived in full.
        RangeIgnored,  ///< A resume was requested and the server sent the whole file.

        /// A resume was requested from at or past the end, so nothing was sent and the partial
        /// already holds every byte there is. Whether they are the right bytes is the digest's
        /// question, not the transport's.
        RangeNotSatisfiable,

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
     *
     * The root holds models/, blobs/ and tmp/, so it does NOT end in "models" -- appending it
     * here as well as in recordPath() is what produced the `Mila\models\models` tree.
     */
    export inline std::filesystem::path resolveStoreRoot()
    {
        if ( const auto explicit_root = readEnvironmentVariable( "MILA_CACHE_DIR" ) )
        {
            return std::filesystem::path( *explicit_root );
        }

        if ( const auto local_app_data = readEnvironmentVariable( "LOCALAPPDATA" ) )
        {
            return std::filesystem::path( *local_app_data ) / "Mila";
        }

        if ( const auto xdg_cache = readEnvironmentVariable( "XDG_CACHE_HOME" ) )
        {
            return std::filesystem::path( *xdg_cache ) / "mila";
        }

        if ( const auto home = readEnvironmentVariable( "HOME" ) )
        {
            return std::filesystem::path( *home ) / ".cache" / "mila";
        }

        return std::filesystem::temp_directory_path() / "mila";
    }

    /**
     * @brief An installed model: what the manifest published, plus how this copy came to be here.
     *
     * The name is the key and the only thing in the path. Everything about origin is a field, so
     * a record copied out of the store still says what it describes and where it came from.
     */
    export struct ModelRecord
    {
        /// The store's key, and the only name a user types. Unique across the store.
        std::string name;

        std::string architecture;

        /// Descriptive. The name carries the precision; this says what the bytes actually are.
        std::string variant;

        std::string weight_quantization;
        std::string minimum_mila_version;

        std::string base_model;
        std::string license;

        /// Instruction-tuned. Decides the prompt template a consumer applies.
        bool instruct{ false };

        /**
         * @brief Where this copy came from. Never published, and never part of the path.
         *
         * A model published from this machine leaves hub empty. Origin is mutable -- a locally
         * published model may be pushed to a hub later -- which is the reason it is a field
         * rather than a directory: changing it must not move files.
         */
        std::string hub;
        std::string owner;
        std::string repository;

        /// The resolved commit, not the ref that was asked for. Empty for a local install.
        std::string revision;

        std::string installed_at;

        std::vector<ModelFile> files;

        /// True when nothing served this: it was published from this machine.
        bool isLocal() const noexcept { return hub.empty(); }

        /// Where it came from, for display. Never a key.
        std::string origin() const
        {
            if ( hub.empty() )
            {
                return "local";
            }

            return owner.empty() ? hub : hub + ":" + owner + "/" + repository;
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

    /**
     * @brief How a package is to be installed.
     *
     * A package carries its own name; these are the overrides. Nothing here names an origin,
     * because installing from a package is what "published from this machine" means.
     */
    export struct InstallOptions
    {
        /// Empty takes the manifest's name, then the package directory's name.
        std::string name;

        /**
         * @brief Replace an existing record of the same name.
         *
         * Off by default. A name is unique across the store, so installing over one is either a
         * refresh the caller meant or a collision they need to know about, and the store cannot
         * tell which.
         */
        bool replace{ false };

        /**
         * @brief Move the package's files into the store rather than copying them.
         *
         * On by default. A move is free on one volume, and it keeps a single integrity model in
         * which the path is the digest -- a copy leaves a second, unmanaged copy of a file that
         * may be several gigabytes. Turn it off to keep the package for a hub upload.
         */
        bool move_files{ true };
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
     * @verbatim
     *   models/<owner>/<repository>/<variant>.json   the records -- the index
     *   blobs/sha256-<hex>                           the content
     *   tmp/                                         in-flight transfers and their locks
     * @endverbatim
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

        /**
         * @brief The case-folded form of a model name, which is what keys the store.
         *
         * A name is a filename, so without folding the store inherits the filesystem's opinion
         * of case: `/install Llama-3.1-8B` resolves on Windows and fails on Linux, and the
         * developer's platform is the forgiving one -- so the failure only ever appears for
         * someone else. Folding makes one name mean one model on both.
         *
         * ASCII by hand rather than std::tolower, which is locale-dependent (a Turkish locale
         * maps 'I' to a dotless form and would key the same model two ways). requireUsableName
         * already restricts names to [A-Za-z0-9._-], so ASCII is the whole domain.
         *
         * The record keeps the name as it was published -- this folds the key, never the label.
         */
        static std::string foldName( std::string_view name )
        {
            std::string folded( name );

            for ( char& character : folded )
            {
                if ( character >= 'A' && character <= 'Z' )
                {
                    character = static_cast<char>( character - 'A' + 'a' );
                }
            }

            return folded;
        }

        std::filesystem::path recordPath( const std::string& name ) const
        {
            return root_ / "models" / ( foldName( name ) + ".json" );
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
         * @param expected_sha256_hex The blob's digest, which is also its name under blobs/.
         * @param fetcher Supplies the bytes when the blob is absent, resuming from any partial.
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

            // RangeNotSatisfiable is not a failure to recover from but a transfer that is already
            // over: nothing was appended because there was nothing left to send. Falling through
            // to the digest is what settles it -- a match publishes, and a partial longer than the
            // file mismatches and is kept with its byte count, which is the evidence either way.
            // Throwing here instead wedged the store forever, since every retry replayed to the
            // same offset and drew the same 416 without the digest ever being consulted.
            if ( report.outcome != FetchOutcome::Complete
                && report.outcome != FetchOutcome::RangeNotSatisfiable )
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

        /**
         * @brief Take a file that is already on disk into the store, verifying it on the way.
         *
         * The counterpart to ensureBlob for bytes that need no transfer. It hashes rather than
         * trusting the caller, because the store's whole guarantee is that a path names its
         * content -- a blob adopted unverified would poison every later cache hit.
         *
         * A move publishes by rename when the package and the store share a volume, which costs
         * nothing whatever the file's size. Across volumes there is no atomic move, so the bytes
         * go through tmp/ and are renamed from there: a partial copy must never occupy a path
         * that implies verification.
         *
         * @return Path to the verified blob.
         * @throws std::runtime_error on a digest mismatch or a lock held by another process.
         */
        std::filesystem::path adoptBlob(
            const std::string& description,
            const std::filesystem::path& source,
            const std::string& expected_sha256_hex,
            bool move_file = true )
        {
            const auto final_path = blobPath( expected_sha256_hex );

            std::error_code ignored;

            if ( std::filesystem::exists( final_path, ignored ) )
            {
                return final_path;
            }

            std::filesystem::create_directories( final_path.parent_path() );
            std::filesystem::create_directories( root_ / "tmp" );

            TransferLock lock( root_ / "tmp" / ( "sha256-" + expected_sha256_hex + ".lock" ) );

            if ( !lock.held() )
            {
                throw std::runtime_error( std::format(
                    "ModelStore: another process is already installing this blob.\n"
                    "  blob {}\n"
                    "  lock {}\n"
                    "If no install is running, delete the lock file and retry.",
                    description, lock.path().string() ) );
            }

            if ( std::filesystem::exists( final_path, ignored ) )
            {
                return final_path;
            }

            const std::string actual = sha256OfFile( source );

            if ( actual != expected_sha256_hex )
            {
                // The source is the caller's file and is left exactly as it was: unlike a
                // download, these bytes are not the store's to quarantine.
                throw std::runtime_error( std::format(
                    "ModelStore: digest mismatch for {}\n"
                    "  file            {}\n"
                    "  expected sha256 {}\n"
                    "  actual   sha256 {}",
                    description, source.string(), expected_sha256_hex, actual ) );
            }

            if ( move_file )
            {
                std::error_code move_error;
                std::filesystem::rename( source, final_path, move_error );

                if ( !move_error )
                {
                    return final_path;
                }
            }

            const auto staging =
                root_ / "tmp" / ( "sha256-" + expected_sha256_hex + ".partial" );

            std::error_code copy_error;
            std::filesystem::copy_file( source, staging,
                std::filesystem::copy_options::overwrite_existing, copy_error );

            if ( copy_error )
            {
                throw std::runtime_error( std::format(
                    "ModelStore: cannot stage {}: {}", source.string(), copy_error.message() ) );
            }

            std::error_code rename_error;
            std::filesystem::rename( staging, final_path, rename_error );

            if ( rename_error )
            {
                if ( !std::filesystem::exists( final_path, ignored ) )
                {
                    std::filesystem::remove( staging, ignored );

                    throw std::runtime_error( std::format(
                        "ModelStore: cannot publish blob {}: {}",
                        final_path.string(), rename_error.message() ) );
                }

                std::filesystem::remove( staging, ignored );
            }

            if ( move_file )
            {
                std::filesystem::remove( source, ignored );
            }

            return final_path;
        }

        // -------------------------------------------------------------------
        // Installation
        // -------------------------------------------------------------------

        /**
         * @brief Install one variant of a package, and record it.
         *
         * Publishing to the local store and publishing to a hub take the same directory: what
         * differs is only where the bytes go. The record is written last, after every file has
         * verified, so a failed install leaves nothing that looks installed.
         *
         * The package is not validated first on purpose. Adoption hashes each file as it takes
         * it, so a separate validate() pass would read every byte a second time -- at 6.8 GB
         * that is not a cost worth paying for a check that already happened.
         *
         * @throws std::runtime_error if the variant does not exist, if it needs a newer Mila, if
         *         a file's digest disagrees with the manifest, or if the names do not form a
         *         coordinate.
         */
        StoredModel install( const ModelPackage& package, const InstallOptions& options = {} )
        {
            const ModelManifest& manifest = package.manifest();
            const std::string source = package.directory().string();

            requireCompatibleMilaVersion( manifest, source );

            ModelRecord record;
            record.name = firstNonEmpty(
                options.name, manifest.name, package.directory().filename().string() );
            record.architecture = manifest.architecture;
            record.variant = manifest.variant;
            record.weight_quantization = manifest.weight_quantization;
            record.minimum_mila_version = manifest.minimum_mila_version;
            record.base_model = manifest.base_model;
            record.license = manifest.license;
            record.instruct = manifest.instruct;

            // hub, owner, repository and revision stay empty: nothing served this, and a record
            // naming a hub it did not come from would make a local build look published.

            const ModelFile* weights = manifest.file( kWeightsRole );

            requireUsableName( record.name );
            requireNameIsFree( record.name, record,
                weights == nullptr ? std::string{} : weights->sha256, options.replace );

            for ( const auto& file : manifest.files )
            {
                adoptBlob(
                    std::format( "{} ({})", file.path, record.name ),
                    package.pathOf( file ),
                    file.sha256,
                    options.move_files );

                record.files.push_back( file );
            }

            return describe( writeRecord( std::move( record ) ) );
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
        /**
         * @brief Persist a record, and hand back what was actually written.
         *
         * Returns the record rather than void because the install time is stamped here: a
         * caller that kept its own copy would hold one that disagrees with the store.
         */
        ModelRecord writeRecord( ModelRecord record )
        {
            requireUsableName( record.name );

            if ( record.installed_at.empty() )
            {
                record.installed_at = utcTimestamp();
            }

            const auto destination = recordPath( record.name );

            std::filesystem::create_directories( destination.parent_path() );
            std::filesystem::create_directories( root_ / "tmp" );

            const std::string text = toJson( record ).dump( 2 );

            // Folded like the destination, so two casings of one name contend for the same
            // staging file rather than both appearing to succeed into the same record.
            const auto staging =
                root_ / "tmp" / std::format( "record-{}.json", foldName( record.name ) );

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

            return record;
        }

        std::optional<ModelRecord> readRecord( const std::string& name ) const
        {
            return readRecordFile( recordPath( name ) );
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

            // One level: the name is the key, so the record tree is flat and a walk is a single
            // directory read.
            for ( const auto& entry :
                std::filesystem::directory_iterator( models_root, ignored ) )
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
        std::optional<StoredModel> locate( const std::string& name ) const
        {
            auto record = readRecord( name );

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

                if ( file.role == kWeightsRole )
                {
                    model.weights_path = path;
                }
                else if ( file.role == kTokenizerRole )
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

        /**
         * @brief Rename an installed model.
         *
         * One record is rewritten and nothing else moves. The blobs are content-addressed, so
         * what a model is called here has no bearing on where its bytes live; and origin is a
         * field rather than a path segment, so a renamed model still says where it came from.
         *
         * The install time is carried over: renaming is not reinstalling.
         *
         * @return False when no model of that name is installed.
         * @throws std::runtime_error if the new name is unusable or already taken.
         */
        bool rename( const std::string& from, const std::string& to )
        {
            requireUsableName( to );

            auto record = readRecord( from );

            if ( !record.has_value() )
            {
                return false;
            }

            if ( from == to )
            {
                return true;
            }

            // A change of case alone is the same record under a new label: it keys to the same
            // file, so it is neither a collision with itself nor something to delete afterwards.
            // Without this the write below would land on recordPath( from ) and the removal
            // would then take the model with it.
            const bool relabel_in_place = ( foldName( from ) == foldName( to ) );

            if ( !relabel_in_place && readRecord( to ).has_value() )
            {
                throw std::runtime_error( std::format(
                    "ModelStore: a model named '{}' is already installed. One name is one model.",
                    to ) );
            }

            record->name = to;

            writeRecord( *record );

            if ( !relabel_in_place )
            {
                std::error_code ignored;
                std::filesystem::remove( recordPath( from ), ignored );
            }

            return true;
        }

        // -------------------------------------------------------------------
        // Removal
        // -------------------------------------------------------------------

        /**
         * @brief Remove one installed model, then reclaim what nothing else references.
         *
         * The sweep is what makes this safe. Deduplication means a tokenizer blob may back several
         * models, so removal cannot delete a model's files simply because that model is going.
         */
        RemovalReport remove( const std::string& name )
        {
            RemovalReport report;

            const auto record_path = recordPath( name );

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

        static const std::string& firstNonEmpty(
            const std::string& first, const std::string& second, const std::string& third )
        {
            if ( !first.empty() )
            {
                return first;
            }

            return second.empty() ? third : second;
        }

        /**
         * @brief Refuse a name that cannot be a filename or cannot be typed back.
         *
         * The name is the record's filename, so anything path-shaped would escape the store, and
         * a name a user cannot retype is not a name. The permitted set matches what a hub allows
         * in a repository, which is what these names come from.
         */
        static void requireUsableName( const std::string& name )
        {
            const bool usable = !name.empty() && std::all_of( name.begin(), name.end(),
                []( char character )
                {
                    return ( character >= 'A' && character <= 'Z' )
                        || ( character >= 'a' && character <= 'z' )
                        || ( character >= '0' && character <= '9' )
                        || character == '.' || character == '_' || character == '-';
                } );

            if ( !usable )
            {
                throw std::runtime_error( std::format(
                    "ModelStore: '{}' is not a usable model name. Letters, digits, '.', '_' and "
                    "'-' only.", name ) );
            }
        }

        /**
         * @brief Enforce that one name means one model.
         *
         * A collision is refused rather than namespaced, because a store where one name means
         * two things is the state this layout exists to make impossible. Silently replacing
         * would be worse than refusing: the displaced model's blobs become unreferenced and the
         * next prune reclaims them.
         *
         * Two cases are not collisions. A hub model reinstalled from the same repository is a
         * refresh, possibly at a newer revision. And identical content under the same name is
         * the same model, which is what keeps a local re-install idempotent -- a locally
         * published model has no origin to compare, so content is the only evidence there is.
         */
        void requireNameIsFree(
            const std::string& name,
            const ModelRecord& incoming,
            const std::string& incoming_weights_digest,
            bool replace ) const
        {
            auto existing = readRecord( name );

            if ( !existing.has_value() || replace )
            {
                return;
            }

            if ( !incoming.hub.empty() && existing->hub == incoming.hub
                && existing->owner == incoming.owner
                && existing->repository == incoming.repository )
            {
                return;
            }

            const ModelFile* existing_weights = existing->file( kWeightsRole );

            if ( existing_weights != nullptr
                && existing_weights->sha256 == incoming_weights_digest )
            {
                return;
            }

            throw std::runtime_error( std::format(
                "ModelStore: a model named '{}' is already installed, from {}. One name is one "
                "model: install this under a different name, remove that one first, or pass "
                "replace to overwrite it.",
                name, existing->origin() ) );
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

            // The published half of the record is the manifest verbatim; `installed` is the half
            // the store writes and never publishes.
            return {
                { "manifest_version", 1 },
                { "name", record.name },
                { "architecture", record.architecture },
                { "variant", record.variant },
                { "weight_quantization", record.weight_quantization },
                { "minimum_mila_version", record.minimum_mila_version },
                { "base_model", record.base_model },
                { "license", record.license },
                { "instruct", record.instruct },
                { "files", files },
                { "installed", {
                    { "hub", record.hub },
                    { "owner", record.owner },
                    { "repository", record.repository },
                    { "revision", record.revision },
                    { "installed_at", record.installed_at } } } };
        }

        static std::optional<ModelRecord> fromJson( const nlohmann::json& json )
        {
            ModelRecord record;

            record.name = json.value( "name", std::string{} );
            record.architecture = json.value( "architecture", std::string{} );
            record.variant = json.value( "variant", std::string{} );
            record.weight_quantization = json.value( "weight_quantization", std::string{} );
            record.minimum_mila_version = json.value( "minimum_mila_version", std::string{} );
            record.base_model = json.value( "base_model", std::string{} );
            record.license = json.value( "license", std::string{} );
            record.instruct = json.value( "instruct", false );

            if ( json.contains( "installed" ) && json[ "installed" ].is_object() )
            {
                const nlohmann::json& installed = json[ "installed" ];

                record.hub = installed.value( "hub", std::string{} );
                record.owner = installed.value( "owner", std::string{} );
                record.repository = installed.value( "repository", std::string{} );
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

            if ( record.name.empty() )
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
