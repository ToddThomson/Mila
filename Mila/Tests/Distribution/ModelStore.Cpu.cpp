/**
 * @file ModelStore.Cpu.cpp
 * @brief SHA-256 known answers, the store's fetch/resume/verify contract, and its records.
 *
 * The store takes its fetcher as a callable, so every case here runs offline and
 * deterministically -- including the ones that would be awkward to provoke against a real
 * server: a resumed transfer, a server that ignores the Range header, and a peer process
 * already transferring the same blob.
 *
 * C stdio rather than fstream throughout: MSVC C++23 raises C2079 on basic_istream::sentry
 * when stream I/O meets `import Mila;` in a .cpp.
 *
 * CPU only, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <algorithm>
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
         * @brief A store root under the temp directory, removed on destruction.
         */
        class ScratchStoreRoot
        {
        public:

            ScratchStoreRoot()
            {
                static int counter = 0;

                path_ = std::filesystem::temp_directory_path()
                    / std::format( "mila_store_test_{}", counter++ );

                std::error_code ignored;
                std::filesystem::remove_all( path_, ignored );
            }

            ~ScratchStoreRoot()
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
                uint64_t resume_from,
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

        /// Put `payload` in the store and return its digest, bypassing the fetch path.
        std::string seedBlob( ModelStore& store, const std::string& payload )
        {
            const std::string digest = sha256Hex( payload.data(), payload.size() );
            writeWholeFile( store.blobPath( digest ), payload );

            return digest;
        }

        ModelFile makeFile(
            std::string role, std::string path, std::string digest, uint64_t bytes )
        {
            ModelFile file;
            file.role = std::move( role );
            file.path = std::move( path );
            file.sha256 = std::move( digest );
            file.bytes = bytes;

            return file;
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
    // Layout
    // ================================================================

    TEST( ModelStore, HonoursAnExplicitStoreRoot )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        EXPECT_EQ( store.root(), scratch.path() );
        EXPECT_EQ( store.blobPath( "abcd" ).filename().string(), "sha256-abcd" );

        // One level, and the name is the key. The root holds models/, blobs/ and tmp/, so it
        // must not itself end in "models" -- that produced the Mila\models\models tree.
        EXPECT_EQ( store.recordPath( "gemma-4-12b-it-fp4" ),
            scratch.path() / "models" / "gemma-4-12b-it-fp4.json" );
    }

    TEST( ModelStore, ResolvesARootThatDoesNotDoubleTheModelsSegment )
    {
        // The default root holds models/ rather than being it.
        const auto root = resolveStoreRoot();

        EXPECT_NE( root.filename().string(), "models" );
    }

    // ================================================================
    // Fetch, verify, publish
    // ================================================================

    TEST( ModelStore, FetchesVerifiesAndPublishesABlob )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string payload = "the quick brown fox jumps over the lazy dog";
        const std::string digest = sha256Hex( payload.data(), payload.size() );

        const auto path = store.ensureBlob( "model.safetensors", digest, servingFetcher( payload ) );

        EXPECT_TRUE( std::filesystem::exists( path ) );
        EXPECT_EQ( path, store.blobPath( digest ) );
        EXPECT_EQ( readWholeFile( path ), payload );
        EXPECT_TRUE( store.contains( digest ) );
    }

    TEST( ModelStore, SecondRequestHitsTheStoreAndDoesNotFetch )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string payload = "cached payload";
        const std::string digest = sha256Hex( payload.data(), payload.size() );

        int calls = 0;
        auto fetcher = servingFetcher( payload, &calls );

        store.ensureBlob( "model.safetensors", digest, fetcher );
        ASSERT_EQ( calls, 1 );

        store.ensureBlob( "model.safetensors", digest, fetcher );

        // The whole point of content-addressing: a present blob is a verified blob.
        EXPECT_EQ( calls, 1 );
    }

    TEST( ModelStore, ResumesFromAnExistingPartial )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string payload = "0123456789abcdefghijklmnopqrstuvwxyz";
        const std::string digest = sha256Hex( payload.data(), payload.size() );

        // Leave a prefix where a previous attempt would have left it.
        const auto partial = scratch.path() / "tmp" / ( "sha256-" + digest + ".partial" );
        writeWholeFile( partial, payload.substr( 0, 10 ) );

        uint64_t observed_offset = 0;

        auto fetcher = [&]( uint64_t resume_from,
            const std::function<bool( const char*, size_t )>& sink ) -> FetchReport
            {
                observed_offset = resume_from;
                const std::string remainder = payload.substr( static_cast<size_t>( resume_from ) );
                sink( remainder.data(), remainder.size() );

                return { FetchOutcome::Complete, {} };
            };

        const auto path = store.ensureBlob( "model.safetensors", digest, fetcher );

        // The prefix was replayed through the hash, not re-downloaded.
        EXPECT_EQ( observed_offset, 10u );
        EXPECT_EQ( readWholeFile( path ), payload );
    }

    TEST( ModelStore, PublishesACompletePartialWhenTheRangeIsNotSatisfiable )
    {
        // A process killed between the last write and the rename leaves a partial holding every
        // byte. Resuming from its end draws a 416, and treating that as a failure wedged the
        // store forever: the retry replayed to the same offset and never consulted the digest.
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string payload = "0123456789abcdef";
        const std::string digest = sha256Hex( payload.data(), payload.size() );

        const auto partial = scratch.path() / "tmp" / ( "sha256-" + digest + ".partial" );
        writeWholeFile( partial, payload );

        uint64_t observed_offset = 0;

        auto fetcher = [&observed_offset]( uint64_t resume_from,
            const std::function<bool( const char*, size_t )>& ) -> FetchReport
            {
                observed_offset = resume_from;

                return { FetchOutcome::RangeNotSatisfiable, "offset is at or past the end" };
            };

        const auto path = store.ensureBlob( "model.safetensors", digest, fetcher );

        EXPECT_EQ( observed_offset, payload.size() );
        EXPECT_TRUE( store.contains( digest ) );
        EXPECT_EQ( readWholeFile( path ), payload );
        EXPECT_FALSE( std::filesystem::exists( partial ) );
    }

    TEST( ModelStore, RejectsAnOverlongPartialWhenTheRangeIsNotSatisfiable )
    {
        // The other half of 416: the offset can be past the end because the partial holds more
        // bytes than the file does. Falling through to the digest is what catches that, and the
        // rejected copy keeps the byte count that says which of the two happened.
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string payload = "0123456789abcdef";
        const std::string digest = sha256Hex( payload.data(), payload.size() );

        const auto partial = scratch.path() / "tmp" / ( "sha256-" + digest + ".partial" );
        writeWholeFile( partial, payload + "trailing garbage" );

        auto fetcher = []( uint64_t,
            const std::function<bool( const char*, size_t )>& ) -> FetchReport
            {
                return { FetchOutcome::RangeNotSatisfiable, "offset is at or past the end" };
            };

        EXPECT_THROW(
            store.ensureBlob( "model.safetensors", digest, fetcher ),
            std::runtime_error );

        EXPECT_FALSE( store.contains( digest ) );
        EXPECT_TRUE( std::filesystem::exists( partial.string() + ".rejected" ) );
    }

    TEST( ModelStore, DiscardsThePartialWhenTheServerIgnoresTheRange )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string payload = "0123456789abcdef";
        const std::string digest = sha256Hex( payload.data(), payload.size() );

        const auto partial = scratch.path() / "tmp" / ( "sha256-" + digest + ".partial" );
        writeWholeFile( partial, payload.substr( 0, 8 ) );

        auto fetcher = []( uint64_t,
            const std::function<bool( const char*, size_t )>& ) -> FetchReport
            {
                return { FetchOutcome::RangeIgnored, "sent the whole file" };
            };

        // Appending a from-zero response onto a partial would concatenate. The partial must
        // be destroyed rather than left for the next attempt to resume onto.
        EXPECT_THROW(
            store.ensureBlob( "model.safetensors", digest, fetcher ),
            std::runtime_error );

        EXPECT_FALSE( std::filesystem::exists( partial ) );
        EXPECT_FALSE( store.contains( digest ) );
    }

    TEST( ModelStore, KeepsAMismatchedBlobForInspection )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string claimed = sha256Hex( "what was promised", 17 );
        const std::string served = "something else entirely";

        EXPECT_THROW(
            store.ensureBlob( "model.safetensors", claimed, servingFetcher( served ) ),
            std::runtime_error );

        EXPECT_FALSE( store.contains( claimed ) );

        // Not resumable -- it no longer occupies the .partial path -- but kept, because the
        // byte count is what separates "altered in flight" from "a length bug".
        const auto partial = scratch.path() / "tmp" / ( "sha256-" + claimed + ".partial" );
        EXPECT_FALSE( std::filesystem::exists( partial ) );

        const auto rejected = std::filesystem::path( partial.string() + ".rejected" );
        ASSERT_TRUE( std::filesystem::exists( rejected ) );
        EXPECT_EQ( readWholeFile( rejected ), served );
    }

    TEST( ModelStore, KeepsThePartialWhenAFetchFailsMidTransfer )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string payload = "0123456789abcdefghij";
        const std::string digest = sha256Hex( payload.data(), payload.size() );

        auto failing = [&]( uint64_t,
            const std::function<bool( const char*, size_t )>& sink ) -> FetchReport
            {
                sink( payload.data(), 6 );

                return { FetchOutcome::Failed, "connection reset" };
            };

        EXPECT_THROW(
            store.ensureBlob( "model.safetensors", digest, failing ),
            std::runtime_error );

        // Kept on purpose: those six bytes are good, and the next attempt resumes onto them.
        const auto partial = scratch.path() / "tmp" / ( "sha256-" + digest + ".partial" );
        ASSERT_TRUE( std::filesystem::exists( partial ) );
        EXPECT_EQ( readWholeFile( partial ), payload.substr( 0, 6 ) );

        // And the retry completes.
        const auto path = store.ensureBlob( "model.safetensors", digest, servingFetcher( payload ) );
        EXPECT_EQ( readWholeFile( path ), payload );
    }

    TEST( ModelStore, PublishesNothingUntilTheDigestIsVerified )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string payload = "atomicity matters";
        const std::string digest = sha256Hex( payload.data(), payload.size() );
        const auto final_path = store.blobPath( digest );

        auto observing = [&]( uint64_t,
            const std::function<bool( const char*, size_t )>& sink ) -> FetchReport
            {
                sink( payload.data(), payload.size() );

                // Mid-transfer the final path must not exist: a reader that sees it is
                // entitled to assume the bytes are complete and verified.
                EXPECT_FALSE( std::filesystem::exists( final_path ) );

                return { FetchOutcome::Complete, {} };
            };

        store.ensureBlob( "model.safetensors", digest, observing );

        EXPECT_TRUE( std::filesystem::exists( final_path ) );
    }

    TEST( ModelStore, RefusesToTransferABlobAnotherProcessHasLocked )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string payload = "one writer at a time";
        const std::string digest = sha256Hex( payload.data(), payload.size() );

        // Stand in for a peer process mid-transfer. Without the lock both would append into
        // one deterministic .partial and interleave.
        const auto lock = scratch.path() / "tmp" / ( "sha256-" + digest + ".lock" );
        writeWholeFile( lock, "" );

        int calls = 0;

        EXPECT_THROW(
            store.ensureBlob( "model.safetensors", digest, servingFetcher( payload, &calls ) ),
            std::runtime_error );

        EXPECT_EQ( calls, 0 );
        EXPECT_FALSE( store.contains( digest ) );

        // Once the peer is gone the transfer proceeds, and the lock does not outlive it.
        std::filesystem::remove( lock );

        store.ensureBlob( "model.safetensors", digest, servingFetcher( payload ) );

        EXPECT_TRUE( store.contains( digest ) );
        EXPECT_FALSE( std::filesystem::exists( lock ) );
    }

    // ================================================================
    // Records -- the index that makes a store a store
    // ================================================================

    TEST( ModelStore, WritesAndReadsBackARecord )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        ModelRecord record;
        record.name = "gemma-4-12b-it-fp4";
        record.architecture = "gemma";
        record.weight_quantization = "per_group_fp4_128";
        record.minimum_mila_version = "0.20.0";
        record.hub = "huggingface";
        record.revision = "570dbe0e";
        record.files.push_back( makeFile( "weights", "gemma.safetensors", "aaaa", 42 ) );

        store.writeRecord( record );

        const auto read_back = store.readRecord( "gemma-4-12b-it-fp4" );

        ASSERT_TRUE( read_back.has_value() );
        EXPECT_EQ( read_back->name, "gemma-4-12b-it-fp4" );
        EXPECT_EQ( read_back->architecture, "gemma" );
        EXPECT_EQ( read_back->weight_quantization, "per_group_fp4_128" );
        EXPECT_EQ( read_back->revision, "570dbe0e" );
        ASSERT_EQ( read_back->files.size(), 1u );
        EXPECT_EQ( read_back->files[ 0 ].sha256, "aaaa" );
        EXPECT_EQ( read_back->files[ 0 ].bytes, 42u );

        // Stamped by the store, not by the caller -- a record must say when it arrived.
        EXPECT_FALSE( read_back->installed_at.empty() );
    }

    TEST( ModelStore, ListsEveryInstalledModel )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string weights_digest = seedBlob( store, "gemma weights" );
        const std::string tokenizer_digest = seedBlob( store, "gemma tokenizer" );

        for ( const char* variant : { "fp4", "fp8" } )
        {
            ModelRecord record;
            record.name = std::string( "gemma-4-12b-it-" ) + variant;
            record.hub = "huggingface";
            record.owner = "mila-llm";
            record.repository = record.name;
            record.files.push_back( makeFile( "weights", "w.safetensors", weights_digest, 13 ) );
            record.files.push_back( makeFile( "tokenizer", "t.bin", tokenizer_digest, 15 ) );

            store.writeRecord( record );
        }

        ModelRecord local;
        local.name = "llama-3.2-3b-instruct-bf16";
        local.files.push_back( makeFile( "weights", "l.safetensors", weights_digest, 13 ) );

        store.writeRecord( local );

        const auto models = store.list();

        ASSERT_EQ( models.size(), 3u );

        // A locally published model is listed exactly like a fetched one. Origin is a field
        // on the record, never a namespace in the path.
        const bool has_local = std::any_of( models.begin(), models.end(),
            []( const StoredModel& model ) { return model.record.isLocal(); } );

        EXPECT_TRUE( has_local );

        for ( const auto& model : models )
        {
            EXPECT_TRUE( model.complete ) << model.record.name;
        }
    }

    TEST( ModelStore, LocatesAnInstalledModelAndNothingElse )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string weights_digest = seedBlob( store, "the weights" );
        const std::string tokenizer_digest = seedBlob( store, "the tokenizer" );

        ModelRecord record;
        record.name = "gemma-4-12b-it-fp4";
        record.files.push_back( makeFile( "weights", "w.safetensors", weights_digest, 11 ) );
        record.files.push_back( makeFile( "tokenizer", "t.bin", tokenizer_digest, 13 ) );

        store.writeRecord( record );

        const auto located = store.locate( "gemma-4-12b-it-fp4" );

        ASSERT_TRUE( located.has_value() );
        EXPECT_EQ( located->weights_path, store.blobPath( weights_digest ) );
        EXPECT_EQ( located->tokenizer_path, store.blobPath( tokenizer_digest ) );
        EXPECT_TRUE( located->complete );

        EXPECT_FALSE( store.locate( "gemma-4-12b-it-fp8" ).has_value() );
        EXPECT_FALSE( store.locate( "not-a-model" ).has_value() );
    }

    TEST( ModelStore, RefusesToLocateAModelWhoseBlobsAreMissing )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        ModelRecord record;
        record.name = "gemma-4-12b-it-fp4";
        record.files.push_back( makeFile( "weights", "w.safetensors", "deadbeef", 11 ) );

        store.writeRecord( record );

        // A caller receiving a path is entitled to bytes behind it.
        EXPECT_FALSE( store.locate( "gemma-4-12b-it-fp4" ).has_value() );

        // But the broken record is still listed, so its owner can see and repair it.
        const auto models = store.list();
        ASSERT_EQ( models.size(), 1u );
        EXPECT_FALSE( models[ 0 ].complete );
    }

    // ================================================================
    // Removal -- refcounted, because deduplication is not free
    // ================================================================

    TEST( ModelStore, RemovingOneModelKeepsTheTokenizerAnotherShares )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string fp4_digest = seedBlob( store, "fp4 weights" );
        const std::string fp8_digest = seedBlob( store, "fp8 weights" );
        const std::string tokenizer_digest = seedBlob( store, "the shared tokenizer" );

        for ( const auto& [variant, digest] : std::vector<std::pair<std::string, std::string>>{
            { "fp4", fp4_digest }, { "fp8", fp8_digest } } )
        {
            ModelRecord record;
            record.name = std::string( "gemma-4-12b-it-" ) + variant;
            record.files.push_back( makeFile( "weights", "w.safetensors", digest, 11 ) );
            record.files.push_back( makeFile( "tokenizer", "t.bin", tokenizer_digest, 20 ) );

            store.writeRecord( record );
        }

        const RemovalReport report = store.remove( "gemma-4-12b-it-fp4" );

        EXPECT_EQ( report.records_removed, 1 );
        EXPECT_EQ( report.blobs_removed, 1 );

        // The removed model's own weights are gone...
        EXPECT_FALSE( store.contains( fp4_digest ) );

        // ...and the tokenizer is not, because fp8 still names it. This is the failure the
        // sweep exists to prevent, and it is silent if it is ever got wrong.
        EXPECT_TRUE( store.contains( tokenizer_digest ) );
        EXPECT_TRUE( store.contains( fp8_digest ) );
        EXPECT_TRUE( store.locate( "gemma-4-12b-it-fp8" ).has_value() );
    }

    TEST( ModelStore, RemovingTheLastModelReclaimsEverything )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string weights_digest = seedBlob( store, "only weights" );
        const std::string tokenizer_digest = seedBlob( store, "only tokenizer" );

        ModelRecord record;
        record.name = "my-model-bf16";
        record.files.push_back( makeFile( "weights", "w.safetensors", weights_digest, 12 ) );
        record.files.push_back( makeFile( "tokenizer", "t.bin", tokenizer_digest, 14 ) );

        store.writeRecord( record );

        const RemovalReport report = store.remove( "my-model-bf16" );

        EXPECT_EQ( report.records_removed, 1 );
        EXPECT_EQ( report.blobs_removed, 2 );
        EXPECT_GT( report.bytes_reclaimed, 0u );
        EXPECT_FALSE( store.contains( weights_digest ) );
        EXPECT_FALSE( store.contains( tokenizer_digest ) );
        EXPECT_TRUE( store.list().empty() );
    }

    TEST( ModelStore, RemovingAModelThatIsNotInstalledIsANoOp )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string digest = seedBlob( store, "unreferenced but not ours to judge" );

        ModelRecord record;
        record.name = "kept-bf16";
        record.files.push_back( makeFile( "weights", "w.safetensors", digest, 34 ) );

        store.writeRecord( record );

        const RemovalReport report = store.remove( "absent" );

        EXPECT_EQ( report.records_removed, 0 );
        EXPECT_EQ( report.blobs_removed, 0 );

        // A missing target must not trigger a sweep that reclaims a live model's blobs.
        EXPECT_TRUE( store.contains( digest ) );
    }

    TEST( ModelStore, PruneReclaimsUnreferencedBlobsAndRejectedTransfers )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string kept_digest = seedBlob( store, "referenced by a record" );
        const std::string orphan_digest = seedBlob( store, "referenced by nothing" );

        ModelRecord record;
        record.name = "kept-bf16";
        record.files.push_back( makeFile( "weights", "w.safetensors", kept_digest, 22 ) );

        store.writeRecord( record );

        const auto rejected = scratch.path() / "tmp" / "sha256-whatever.partial.rejected";
        writeWholeFile( rejected, "bytes that failed their digest" );

        const auto partial = scratch.path() / "tmp" / "sha256-inflight.partial";
        writeWholeFile( partial, "a resumable prefix" );

        const RemovalReport report = store.prune();

        EXPECT_EQ( report.blobs_removed, 1 );
        EXPECT_EQ( report.files_removed, 1 );
        EXPECT_TRUE( store.contains( kept_digest ) );
        EXPECT_FALSE( store.contains( orphan_digest ) );
        EXPECT_FALSE( std::filesystem::exists( rejected ) );

        // A partial is good bytes a retry resumes onto, so pruning must not silently turn a
        // cheap retry into a full re-download.
        EXPECT_TRUE( std::filesystem::exists( partial ) );

        PruneOptions discard;
        discard.discard_partials = true;

        store.prune( discard );

        EXPECT_FALSE( std::filesystem::exists( partial ) );
    }

    TEST( ModelStore, ReportsWhatItHoldsAndWhatCanBeReclaimed )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const std::string kept_digest = seedBlob( store, "referenced" );
        seedBlob( store, "orphaned" );

        ModelRecord record;
        record.name = "kept-bf16";
        record.files.push_back( makeFile( "weights", "w.safetensors", kept_digest, 10 ) );

        store.writeRecord( record );

        writeWholeFile( scratch.path() / "tmp" / "sha256-x.partial", "partial bytes" );

        const StoreUsage usage = store.usage();

        EXPECT_EQ( usage.model_count, 1 );
        EXPECT_EQ( usage.blob_count, 2 );
        EXPECT_EQ( usage.blob_bytes, std::string( "referenced" ).size()
            + std::string( "orphaned" ).size() );
        EXPECT_EQ( usage.reclaimable_bytes, std::string( "orphaned" ).size() );
        EXPECT_EQ( usage.partial_bytes, std::string( "partial bytes" ).size() );
    }
}
