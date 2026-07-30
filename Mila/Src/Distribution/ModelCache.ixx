/**
 * @file ModelCache.ixx
 * @brief Content-addressed blob cache for downloaded model files.
 *
 * A blob's path is its digest, so a file present in the store has been verified and an
 * interrupted transfer cannot be mistaken for a complete one. Fetching is injected rather
 * than called directly, which is what lets resume, range-rejection and digest-mismatch
 * handling be tested without a network. See Specifications/ModelDistribution.md.
 */

module;
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>

export module Distribution.ModelCache;

import Distribution.Sha256;

namespace Mila::Distribution
{
    /**
     * @brief What a fetch attempt reported back to the cache.
     *
     * Deliberately narrower than HttpResult: the cache cares about three outcomes, and
     * keeping it decoupled from the HTTP client is what makes it testable offline.
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
     * @brief Fetches url from resume_from, handing each chunk to sink.
     *
     * The cache supplies the sink so it can hash and write in one pass.
     */
    export using BlobFetcher = std::function<FetchReport(
        const std::string& url,
        uint64_t resume_from,
        const std::function<bool( const char*, size_t )>& sink )>;

    /**
     * @brief Resolve the cache root, first match wins.
     *
     * MILA_CACHE_DIR, then the platform user cache. An explicit override exists because a
     * multi-gigabyte store often does not belong on the same volume as the user profile.
     */
    export inline std::filesystem::path resolveCacheRoot()
    {
        if ( const char* explicit_root = std::getenv( "MILA_CACHE_DIR" ) )
        {
            if ( *explicit_root != '\0' )
            {
                return std::filesystem::path( explicit_root );
            }
        }

        if ( const char* local_app_data = std::getenv( "LOCALAPPDATA" ) )
        {
            if ( *local_app_data != '\0' )
            {
                return std::filesystem::path( local_app_data ) / "Mila" / "models";
            }
        }

        if ( const char* xdg_cache = std::getenv( "XDG_CACHE_HOME" ) )
        {
            if ( *xdg_cache != '\0' )
            {
                return std::filesystem::path( xdg_cache ) / "mila" / "models";
            }
        }

        if ( const char* home = std::getenv( "HOME" ) )
        {
            if ( *home != '\0' )
            {
                return std::filesystem::path( home ) / ".cache" / "mila" / "models";
            }
        }

        return std::filesystem::temp_directory_path() / "mila" / "models";
    }

    /**
     * @brief Content-addressed store of model blobs.
     *
     * Layout:
     *   blobs/sha256-<hex>   verified files
     *   tmp/<unique>         in-flight transfers only
     *   manifests/...        owned by the resolver, not this class
     */
    export class ModelCache
    {
    public:

        explicit ModelCache( std::filesystem::path root = resolveCacheRoot() )
            : root_( std::move( root ) )
        {}

        const std::filesystem::path& root() const noexcept { return root_; }

        std::filesystem::path blobPath( const std::string& sha256_hex ) const
        {
            return root_ / "blobs" / ( "sha256-" + sha256_hex );
        }

        std::filesystem::path manifestPath(
            const std::string& organization,
            const std::string& repository,
            const std::string& variant ) const
        {
            return root_ / "manifests" / organization / repository / ( variant + ".json" );
        }

        bool contains( const std::string& sha256_hex ) const
        {
            std::error_code ignored;

            return std::filesystem::exists( blobPath( sha256_hex ), ignored );
        }

        /**
         * @brief Ensure a blob is present, fetching it if it is not.
         *
         * Resumes from whatever a previous attempt left in tmp/, which is why the partial is
         * named after the digest rather than randomly -- a retry must be able to find it. The
         * hash is recomputed over the resumed prefix before appending, because SHA-256 is
         * sequential and cannot be restored from a byte offset alone.
         *
         * On a digest mismatch the partial is destroyed rather than kept: the bytes are known
         * bad, and leaving them would make the next attempt resume onto corruption.
         *
         * @return Path to the verified blob.
         * @throws std::runtime_error on fetch failure or digest mismatch.
         */
        std::filesystem::path ensureBlob(
            const std::string& url,
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

            const auto partial_path = root_ / "tmp" / ( "sha256-" + expected_sha256_hex + ".partial" );

            Sha256 hash;
            uint64_t resume_from = 0;

            if ( std::filesystem::exists( partial_path, ignored ) )
            {
                resume_from = replayPartialIntoHash( partial_path, hash );
            }

            // Append mode so a resumed transfer extends the partial rather than truncating it.
            FilePointer output( std::fopen( partial_path.string().c_str(),
                resume_from > 0 ? "ab" : "wb" ), &std::fclose );

            if ( output == nullptr )
            {
                throw std::runtime_error(
                    "ModelCache: cannot open " + partial_path.string() + " for writing" );
            }

            auto sink = [&]( const char* data, size_t length ) -> bool
                {
                    if ( std::fwrite( data, 1, length, output.get() ) != length )
                    {
                        return false;
                    }

                    hash.update( data, length );

                    return true;
                };

            FetchReport report = fetcher( url, resume_from, sink );

            output.reset();

            if ( report.outcome == FetchOutcome::RangeIgnored )
            {
                // The server sent the whole file despite the Range header, so the partial now
                // holds a prefix followed by a second copy from byte zero. Nothing salvageable.
                std::filesystem::remove( partial_path, ignored );

                throw std::runtime_error( std::format(
                    "ModelCache: server ignored the resume request for {}; "
                    "discarded the partial, retry from the start", url ) );
            }

            if ( report.outcome != FetchOutcome::Complete )
            {
                // Kept, not deleted: the bytes so far are good and a retry can resume onto them.
                throw std::runtime_error( std::format(
                    "ModelCache: fetch of {} failed: {}", url, report.message ) );
            }

            const std::string actual = hash.finish();

            if ( actual != expected_sha256_hex )
            {
                // Report the byte count too. A digest alone cannot distinguish "the right
                // bytes arrived and the manifest is stale" from "the wrong number of bytes
                // arrived", and those need completely different fixes. Keep the partial for
                // inspection under a rejected name rather than destroying the evidence --
                // it is excluded from resume by not being the .partial path.
                std::error_code size_error;
                const auto received = std::filesystem::file_size( partial_path, size_error );

                const auto rejected_path = partial_path.string() + ".rejected";
                std::filesystem::remove( rejected_path, ignored );
                std::filesystem::rename( partial_path, rejected_path, ignored );

                throw std::runtime_error( std::format(
                    "ModelCache: digest mismatch for {}\n"
                    "  expected sha256 {}\n"
                    "  actual   sha256 {}\n"
                    "  bytes received  {}\n"
                    "  kept for inspection at {}",
                    url, expected_sha256_hex, actual,
                    size_error ? 0 : received, rejected_path ) );
            }

            // The rename is what publishes the blob. Until it lands, no reader can see a
            // path that implies verification.
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
                    "ModelCache: cannot publish blob {}: {}",
                    final_path.string(), rename_error.message() ) );
            }

            return final_path;
        }

    private:

        using FilePointer = std::unique_ptr<std::FILE, int( * )( std::FILE* )>;

        /**
         * @brief Feed an existing partial through the hash and return its length.
         *
         * SHA-256 is sequential, so resuming a transfer means re-reading the prefix. That is
         * disk-local and far cheaper than re-downloading it.
         */
        static uint64_t replayPartialIntoHash(
            const std::filesystem::path& partial_path, Sha256& hash )
        {
            FilePointer input( std::fopen( partial_path.string().c_str(), "rb" ), &std::fclose );

            if ( input == nullptr )
            {
                return 0;
            }

            std::string buffer( 1u << 20, '\0' );
            uint64_t total = 0;

            for ( ;; )
            {
                const size_t read = std::fread( buffer.data(), 1, buffer.size(), input.get() );

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
