/**
 * @file ModelCache.Cpu.cpp
 * @brief SHA-256 known-answer tests, and the cache's fetch/resume/verify contract.
 *
 * The cache takes its fetcher as a callable, so every case here runs offline and
 * deterministically -- including the two that matter most and would be awkward to provoke
 * against a real server: a resumed transfer, and a server that ignores the Range header and
 * replies with the whole file.
 *
 * C stdio rather than fstream throughout: MSVC C++23 raises C2079 on basic_istream::sentry
 * when stream I/O meets `import Mila;` in a .cpp.
 *
 * CPU only, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

import Mila;

namespace Mila::Tests::Distribution
{
    using namespace Mila::Distribution;

    namespace
    {
        using FilePointer = std::unique_ptr<std::FILE, int( * )( std::FILE* )>;

        FilePointer openFile( const std::filesystem::path& path, const char* mode )
        {
            return FilePointer( std::fopen( path.string().c_str(), mode ), &std::fclose );
        }

        std::string readWholeFile( const std::filesystem::path& path )
        {
            auto file = openFile( path, "rb" );

            if ( file == nullptr )
            {
                return {};
            }

            std::string contents;
            char buffer[ 4096 ];

            for ( ;; )
            {
                const size_t read = std::fread( buffer, 1, sizeof( buffer ), file.get() );

                if ( read == 0 )
                {
                    break;
                }

                contents.append( buffer, read );
            }

            return contents;
        }

        void writeWholeFile( const std::filesystem::path& path, std::string_view contents )
        {
            std::filesystem::create_directories( path.parent_path() );

            auto file = openFile( path, "wb" );
            ASSERT_NE( file.get(), nullptr );
            std::fwrite( contents.data(), 1, contents.size(), file.get() );
        }

        /**
         * @brief A cache root under the temp directory, removed on destruction.
         */
        class ScratchCacheRoot
        {
        public:

            ScratchCacheRoot()
            {
                static int counter = 0;

                path_ = std::filesystem::temp_directory_path()
                    / std::format( "mila_cache_test_{}", counter++ );

                std::error_code ignored;
                std::filesystem::remove_all( path_, ignored );
            }

            ~ScratchCacheRoot()
            {
                std::error_code ignored;
                std::filesystem::remove_all( path_, ignored );
            }

            const std::filesystem::path& path() const { return path_; }

        private:

            std::filesystem::path path_;
        };

        /// A fetcher that serves `payload`, honouring resume_from.
        BlobFetcher servingFetcher( std::string payload, int* call_count = nullptr )
        {
            return [payload = std::move( payload ), call_count](
                const std::string&, uint64_t resume_from,
                const std::function<bool( const char*, size_t )>& sink ) -> FetchReport
                {
                    if ( call_count != nullptr )
                    {
                        ++( *call_count );
                    }

                    if ( resume_from > payload.size() )
                    {
                        return { FetchOutcome::Failed, "resume past end" };
                    }

                    const std::string remainder = payload.substr( static_cast<size_t>( resume_from ) );

                    if ( !sink( remainder.data(), remainder.size() ) )
                    {
                        return { FetchOutcome::Failed, "sink refused" };
                    }

                    return { FetchOutcome::Complete, {} };
                };
        }
    }

    // ================================================================
    // SHA-256 known answers
    // ================================================================

    TEST( Sha256, MatchesTheNistVectors )
    {
        EXPECT_EQ( sha256Hex( "", 0 ),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" );

        EXPECT_EQ( sha256Hex( "abc", 3 ),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" );

        // 56 bytes: exercises the padding path where the length does not fit the final block.
        const std::string fifty_six =
            "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        EXPECT_EQ( sha256Hex( fifty_six.data(), fifty_six.size() ),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" );
    }

    TEST( Sha256, IsIndifferentToChunkBoundaries )
    {
        const std::string message( 1000, 'x' );
        const std::string expected = sha256Hex( message.data(), message.size() );

        // Chunk sizes chosen to straddle the 64-byte block boundary in different phases.
        for ( size_t chunk : { size_t( 1 ), size_t( 7 ), size_t( 63 ), size_t( 64 ), size_t( 65 ), size_t( 999 ) } )
        {
            Sha256 hash;

            for ( size_t offset = 0; offset < message.size(); offset += chunk )
            {
                hash.update( message.data() + offset, std::min( chunk, message.size() - offset ) );
            }

            EXPECT_EQ( hash.finish(), expected ) << "chunk size " << chunk;
        }
    }

    // ================================================================
    // Cache root resolution
    // ================================================================

    TEST( ModelCache, HonoursAnExplicitCacheRoot )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );

        EXPECT_EQ( cache.root(), scratch.path() );
        EXPECT_EQ( cache.blobPath( "abcd" ).filename().string(), "sha256-abcd" );
        EXPECT_EQ( cache.manifestPath( "mila-llm", "gemma-4-12b-it", "fp4" ),
            scratch.path() / "manifests" / "mila-llm" / "gemma-4-12b-it" / "fp4.json" );
    }

    // ================================================================
    // Fetch, verify, publish
    // ================================================================

    TEST( ModelCache, FetchesVerifiesAndPublishesABlob )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );

        const std::string payload = "the quick brown fox jumps over the lazy dog";
        const std::string digest = sha256Hex( payload.data(), payload.size() );

        const auto path = cache.ensureBlob( "https://example/blob", digest, servingFetcher( payload ) );

        EXPECT_TRUE( std::filesystem::exists( path ) );
        EXPECT_EQ( path, cache.blobPath( digest ) );
        EXPECT_EQ( readWholeFile( path ), payload );
        EXPECT_TRUE( cache.contains( digest ) );
    }

    TEST( ModelCache, SecondRequestHitsTheCacheAndDoesNotFetch )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );

        const std::string payload = "cached payload";
        const std::string digest = sha256Hex( payload.data(), payload.size() );

        int calls = 0;
        auto fetcher = servingFetcher( payload, &calls );

        cache.ensureBlob( "https://example/blob", digest, fetcher );
        ASSERT_EQ( calls, 1 );

        cache.ensureBlob( "https://example/blob", digest, fetcher );

        // The whole point of content-addressing: a present blob is a verified blob.
        EXPECT_EQ( calls, 1 );
    }

    TEST( ModelCache, ResumesFromAnExistingPartial )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );

        const std::string payload = "0123456789abcdefghijklmnopqrstuvwxyz";
        const std::string digest = sha256Hex( payload.data(), payload.size() );

        // Leave a prefix where a previous attempt would have left it.
        const auto partial = scratch.path() / "tmp" / ( "sha256-" + digest + ".partial" );
        writeWholeFile( partial, payload.substr( 0, 10 ) );

        uint64_t observed_offset = 0;

        auto fetcher = [&]( const std::string&, uint64_t resume_from,
            const std::function<bool( const char*, size_t )>& sink ) -> FetchReport
            {
                observed_offset = resume_from;
                const std::string remainder = payload.substr( static_cast<size_t>( resume_from ) );
                sink( remainder.data(), remainder.size() );

                return { FetchOutcome::Complete, {} };
            };

        const auto path = cache.ensureBlob( "https://example/blob", digest, fetcher );

        // The prefix was replayed through the hash, not re-downloaded.
        EXPECT_EQ( observed_offset, 10u );
        EXPECT_EQ( readWholeFile( path ), payload );
    }

    TEST( ModelCache, DiscardsThePartialWhenTheServerIgnoresTheRange )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );

        const std::string payload = "0123456789abcdef";
        const std::string digest = sha256Hex( payload.data(), payload.size() );

        const auto partial = scratch.path() / "tmp" / ( "sha256-" + digest + ".partial" );
        writeWholeFile( partial, payload.substr( 0, 8 ) );

        auto fetcher = []( const std::string&, uint64_t,
            const std::function<bool( const char*, size_t )>& ) -> FetchReport
            {
                return { FetchOutcome::RangeIgnored, "sent the whole file" };
            };

        // Appending a from-zero response onto a partial would concatenate. The partial must
        // be destroyed rather than left for the next attempt to resume onto.
        EXPECT_THROW(
            cache.ensureBlob( "https://example/blob", digest, fetcher ),
            std::runtime_error );

        EXPECT_FALSE( std::filesystem::exists( partial ) );
        EXPECT_FALSE( cache.contains( digest ) );
    }

    TEST( ModelCache, RejectsAndDestroysABlobWhoseDigestDoesNotMatch )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );

        const std::string claimed = sha256Hex( "what was promised", 17 );

        auto fetcher = servingFetcher( "something else entirely" );

        EXPECT_THROW(
            cache.ensureBlob( "https://example/blob", claimed, fetcher ),
            std::runtime_error );

        // Nothing published, and the bad bytes are gone -- keeping them would make the next
        // attempt resume onto corruption.
        EXPECT_FALSE( cache.contains( claimed ) );

        const auto partial = scratch.path() / "tmp" / ( "sha256-" + claimed + ".partial" );
        EXPECT_FALSE( std::filesystem::exists( partial ) );
    }

    TEST( ModelCache, KeepsThePartialWhenAFetchFailsMidTransfer )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );

        const std::string payload = "0123456789abcdefghij";
        const std::string digest = sha256Hex( payload.data(), payload.size() );

        auto failing = [&]( const std::string&, uint64_t,
            const std::function<bool( const char*, size_t )>& sink ) -> FetchReport
            {
                sink( payload.data(), 6 );

                return { FetchOutcome::Failed, "connection reset" };
            };

        EXPECT_THROW(
            cache.ensureBlob( "https://example/blob", digest, failing ),
            std::runtime_error );

        // Kept on purpose: those six bytes are good, and the next attempt resumes onto them.
        const auto partial = scratch.path() / "tmp" / ( "sha256-" + digest + ".partial" );
        ASSERT_TRUE( std::filesystem::exists( partial ) );
        EXPECT_EQ( readWholeFile( partial ), payload.substr( 0, 6 ) );

        // And the retry completes.
        const auto path = cache.ensureBlob( "https://example/blob", digest, servingFetcher( payload ) );
        EXPECT_EQ( readWholeFile( path ), payload );
    }

    TEST( ModelCache, PublishesNothingUntilTheDigestIsVerified )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );

        const std::string payload = "atomicity matters";
        const std::string digest = sha256Hex( payload.data(), payload.size() );
        const auto final_path = cache.blobPath( digest );

        auto observing = [&]( const std::string&, uint64_t,
            const std::function<bool( const char*, size_t )>& sink ) -> FetchReport
            {
                sink( payload.data(), payload.size() );

                // Mid-transfer the final path must not exist: a reader that sees it is
                // entitled to assume the bytes are complete and verified.
                EXPECT_FALSE( std::filesystem::exists( final_path ) );

                return { FetchOutcome::Complete, {} };
            };

        cache.ensureBlob( "https://example/blob", digest, observing );

        EXPECT_TRUE( std::filesystem::exists( final_path ) );
    }
}
