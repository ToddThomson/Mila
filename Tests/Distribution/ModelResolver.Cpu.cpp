/**
 * @file ModelResolver.Cpu.cpp
 * @brief Coordinate parsing, manifest validation, version skew, and the record a pull leaves.
 *
 * Remote access is injected, so every case runs offline: the manifest is a string this file
 * supplies and the blob fetcher serves bytes from memory. No network in the suite.
 *
 * CPU only, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <cstdio>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

import Mila;

namespace Mila::Tests::Distribution
{
    using namespace Mila::Distribution;

    namespace
    {
        class ScratchStoreRoot
        {
        public:

            ScratchStoreRoot()
            {
                static int counter = 0;

                path_ = std::filesystem::temp_directory_path()
                    / std::format( "mila_resolver_test_{}", counter++ );

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

        const std::string kWeightsPayload = "pretend this is a safetensors artifact";
        const std::string kTokenizerPayload = "pretend this is a tokenizer";

        std::string manifestJson( const std::string& minimum_version = "0.1.0" )
        {
            return std::format( R"({{
              "manifest_version": 1,
              "name": "gemma-4-12b-it-fp4",
              "architecture": "gemma",
              "variant": "fp4",
              "weight_quantization": "per_group_fp4_128",
              "minimum_mila_version": "{}",
              "base_model": "google/gemma-4-12b-it",
              "license": "apache-2.0",
              "files": {{
                "weights":   {{ "path": "model_fp4.safetensors", "sha256": "{}" }},
                "tokenizer": {{ "path": "tokenizer.bin",         "sha256": "{}" }}
              }}
            }})",
                minimum_version,
                sha256Hex( kWeightsPayload.data(), kWeightsPayload.size() ),
                sha256Hex( kTokenizerPayload.data(), kTokenizerPayload.size() ) );
        }

        /// A model with no tokenizer -- the weights are the only required role.
        std::string weightsOnlyManifestJson()
        {
            return std::format( R"({{
              "name": "gemma-4-12b-it-fp8",
              "architecture": "gemma",
              "variant": "fp8",
              "weight_quantization": "per_channel_fp8_e4m3",
              "files": {{
                "weights": {{ "path": "model_fp8.safetensors", "sha256": "{}" }}
              }}
            }})", sha256Hex( kWeightsPayload.data(), kWeightsPayload.size() ) );
        }

        /**
         * @brief A hub that serves a manifest this file supplies and payloads from memory.
         *
         * The point of IModelHub being an interface rather than a set of callbacks: the whole
         * resolver suite runs with no network and no URL, and a case that would be awkward to
         * provoke against a live endpoint is a one-line override here.
         */
        class FakeHub : public IModelHub
        {
        public:

            explicit FakeHub( std::string manifest )
                : manifest_( std::move( manifest ) )
            {}

            std::string name() const override { return "fake"; }

            std::vector<HubModel> listModels( const std::string& ) const override { return {}; }

            std::string fetchManifest( const ModelCoordinate& ) const override
            {
                ++manifest_fetches;

                return manifest_;
            }

            FetchReport fetchFile(
                const ModelCoordinate&,
                const std::string& path,
                uint64_t resume_from,
                const std::function<bool( const char*, size_t )>& sink ) const override
            {
                ++file_fetches;

                if ( fail_files )
                {
                    return { FetchOutcome::Failed, "connection reset" };
                }

                const std::string& payload =
                    path.ends_with( "tokenizer.bin" ) ? kTokenizerPayload : kWeightsPayload;

                const std::string remainder =
                    payload.substr( static_cast<size_t>( resume_from ) );

                sink( remainder.data(), remainder.size() );

                return { FetchOutcome::Complete, {} };
            }

            mutable int manifest_fetches{ 0 };
            mutable int file_fetches{ 0 };
            bool fail_files{ false };

        private:

            std::string manifest_;
        };

        /// A hub that fails the test if it is touched at all.
        class ExplodingHub : public IModelHub
        {
        public:

            std::string name() const override { return "exploding"; }

            std::vector<HubModel> listModels( const std::string& ) const override
            {
                ADD_FAILURE() << "resolver listed a hub it should not have reached";

                return {};
            }

            std::string fetchManifest( const ModelCoordinate& coordinate ) const override
            {
                ADD_FAILURE() << "resolver reached the hub for " << coordinate.toString();

                return {};
            }

            FetchReport fetchFile(
                const ModelCoordinate&, const std::string&, uint64_t,
                const std::function<bool( const char*, size_t )>& ) const override
            {
                ADD_FAILURE() << "resolver fetched a file it should not have";

                return { FetchOutcome::Failed, {} };
            }
        };
    }

    // ================================================================
    // Coordinate grammar
    // ================================================================

    TEST( ModelCoordinateParsing, AcceptsTheFullForm )
    {
        const auto coordinate = parseCoordinate( "mila-llm/gemma-4-12b-it@v2" );

        ASSERT_TRUE( coordinate.has_value() );
        EXPECT_EQ( coordinate->organization, "mila-llm" );
        EXPECT_EQ( coordinate->repository, "gemma-4-12b-it" );
        EXPECT_EQ( coordinate->revision, "v2" );
    }

    TEST( ModelCoordinateParsing, DefaultsRevisionAndLeavesVariantEmpty )
    {
        const auto coordinate = parseCoordinate( "mila-llm/gemma-4-12b-it" );

        ASSERT_TRUE( coordinate.has_value() );
        EXPECT_EQ( coordinate->revision, "main" );
    }

    TEST( ModelCoordinateParsing, StripsTheExplicitPrefix )
    {
        const auto coordinate = parseCoordinate( "hf:mila-llm/gemma-4-12b-it" );

        ASSERT_TRUE( coordinate.has_value() );
        EXPECT_EQ( coordinate->organization, "mila-llm" );
    }

    TEST( ModelCoordinateParsing, RejectsThingsThatAreReallyPaths )
    {
        // A second separator, a drive letter, a backslash or no separator at all: none of
        // these are coordinates, and misreading one would turn a typo'd path into a network
        // request against a nonexistent repository.
        EXPECT_FALSE( parseCoordinate( "Data/Models/Gemma/model.safetensors" ).has_value() );
        EXPECT_FALSE( parseCoordinate( "D:/Repos/model.safetensors" ).has_value() );
        EXPECT_FALSE( parseCoordinate( R"(C:\models\model.safetensors)" ).has_value() );
        EXPECT_FALSE( parseCoordinate( "model.safetensors" ).has_value() );
        EXPECT_FALSE( parseCoordinate( "" ).has_value() );
        EXPECT_FALSE( parseCoordinate( "mila-llm/" ).has_value() );
        EXPECT_FALSE( parseCoordinate( "/gemma" ).has_value() );
        EXPECT_FALSE( parseCoordinate( "mila-llm/gemma:" ).has_value() );
    }

    // ================================================================
    // A path is an input to installation, never to loading
    // ================================================================

    TEST( ModelResolverTests, RefusesAPathAndSaysWhy )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        // A path must not reach the hub either. Any remote call here is a failure.
        const ExplodingHub exploding;

        const auto local = scratch.path() / "already_have_this.safetensors";
        std::filesystem::create_directories( local.parent_path() );
        {
            std::unique_ptr<std::FILE, int( * )( std::FILE* )> file(
                std::fopen( local.string().c_str(), "wb" ), &std::fclose );
            ASSERT_NE( file.get(), nullptr );
            std::fwrite( "x", 1, 1, file.get() );
        }

        ModelResolver resolver( store, exploding );

        try
        {
            resolver.pull( local.string(), "mila-llm" );
            FAIL() << "expected a throw";
        }
        catch ( const std::runtime_error& error )
        {
            // An existing file is a different mistake from a typo, and the message has to
            // name the operation that does take a path rather than say "not a coordinate".
            const std::string message = error.what();

            EXPECT_NE( message.find( "install" ), std::string::npos ) << message;
        }
    }

    TEST( ModelResolverTests, RefusesASpecThatIsNotACoordinate )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );
        const FakeHub hub( manifestJson() );
        ModelResolver resolver( store, hub );

        EXPECT_THROW( resolver.pull( "not/a/real/thing.safetensors", "mila-llm" ), std::runtime_error );
    }

    // ================================================================
    // Coordinate resolution
    // ================================================================

    TEST( ModelResolverTests, PullsACoordinateAndStoresBothFiles )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );
        const FakeHub hub( manifestJson() );
        ModelResolver resolver( store, hub );

        const auto pulled = resolver.pull( "gemma-4-12b-it-fp4", "mila-llm" );

        EXPECT_EQ( pulled.record.name, "gemma-4-12b-it-fp4" );
        EXPECT_EQ( pulled.record.architecture, "gemma" );
        EXPECT_EQ( pulled.record.weight_quantization, "per_group_fp4_128" );
        EXPECT_TRUE( pulled.complete );

        EXPECT_TRUE( std::filesystem::exists( pulled.weights_path ) );
        EXPECT_TRUE( std::filesystem::exists( pulled.tokenizer_path ) );

        // Content-addressed: the filename is the digest, not the repository path.
        EXPECT_EQ( pulled.weights_path,
            store.blobPath( sha256Hex( kWeightsPayload.data(), kWeightsPayload.size() ) ) );
    }

    TEST( ModelResolverTests, APullLeavesARecordThatSurvivesTheResolver )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const FakeHub hub( manifestJson() );

        {
            ModelResolver resolver( store, hub );
            resolver.pull( "gemma-4-12b-it-fp4", "mila-llm" );
        }

        // The record is the whole point: pull and load are separate verbs, in separate
        // processes, so what the hub knew has to outlive the object that fetched it.
        const auto located = store.locate( "gemma-4-12b-it-fp4" );

        ASSERT_TRUE( located.has_value() );
        EXPECT_EQ( located->record.architecture, "gemma" );
        EXPECT_EQ( located->record.revision, "main" );

        // Against the hub's own name, not a literal: a record has to say which hub served it,
        // and a second hub that forgot to identify itself has to fail here rather than pass
        // because the only implementation happened to be HuggingFace.
        EXPECT_EQ( located->record.hub, hub.name() );
        EXPECT_TRUE( std::filesystem::exists( located->weights_path ) );

        ASSERT_EQ( store.list().size(), 1u );
    }

    TEST( ModelResolverTests, AFailedPullLeavesNoRecord )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        FakeHub failing( manifestJson() );
        failing.fail_files = true;

        ModelResolver resolver( store, failing );

        EXPECT_THROW( resolver.pull( "gemma-4-12b-it-fp4", "mila-llm" ), std::runtime_error );

        // A record naming a blob that never arrived would make a broken model look installed.
        EXPECT_TRUE( store.list().empty() );
        EXPECT_FALSE( store.locate( "gemma-4-12b-it-fp4" ).has_value() );
    }

    TEST( ModelResolverTests, PullsAModelWithNoTokenizer )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );
        const FakeHub hub( weightsOnlyManifestJson() );
        ModelResolver resolver( store, hub );

        const auto pulled = resolver.pull( "gemma-4-12b-it-fp8", "mila-llm" );

        EXPECT_EQ( pulled.record.name, "gemma-4-12b-it-fp8" );
        EXPECT_TRUE( pulled.complete );
        EXPECT_TRUE( std::filesystem::exists( pulled.weights_path ) );
        EXPECT_TRUE( pulled.tokenizer_path.empty() );
    }

    TEST( ModelResolverTests, RefusesAnArtifactRequiringANewerMila )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );
        const FakeHub hub( manifestJson( "99.0.0" ) );
        ModelResolver resolver( store, hub );

        // Better a version comparison than a parse error deep inside the tensor index, which
        // is what a future format change would otherwise look like.
        EXPECT_THROW( resolver.pull( "gemma-4-12b-it-fp4", "mila-llm" ), std::runtime_error );
    }

    TEST( ModelResolverTests, RejectsAMalformedManifest )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const FakeHub not_json_hub( "this is not json" );
        ModelResolver not_json( store, not_json_hub );
        EXPECT_THROW( not_json.pull( "gemma-4-12b-it-fp4", "mila-llm" ), std::runtime_error );

        const FakeHub no_files_hub( R"({"architecture":"gemma"})" );
        ModelResolver no_variants( store, no_files_hub );
        EXPECT_THROW( no_variants.pull( "gemma-4-12b-it-fp4", "mila-llm" ), std::runtime_error );

        const FakeHub no_digest_hub( R"({"files":{"weights":{"path":"m.safetensors"}}})" );
        ModelResolver no_digest( store, no_digest_hub );
        EXPECT_THROW( no_digest.pull( "gemma-4-12b-it-fp4", "mila-llm" ), std::runtime_error );
    }

    TEST( ModelResolverTests, SecondPullReusesTheStoredBlobs )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        const FakeHub hub( manifestJson() );
        ModelResolver resolver( store, hub );

        const auto first = resolver.pull( "gemma-4-12b-it-fp4", "mila-llm" );
        const auto second = resolver.pull( "gemma-4-12b-it-fp4", "mila-llm" );

        EXPECT_EQ( first.weights_path, second.weights_path );

        // The manifest is still fetched each time -- an open decision in the spec. The blobs
        // are not, which is the part that matters at 6.33 GB: two files on the first pull,
        // none on the second.
        EXPECT_EQ( hub.manifest_fetches, 2 );
        EXPECT_EQ( hub.file_fetches, 2 );
    }

    TEST( ModelResolverTests, RefusesANameThatIsNotAName )
    {
        ScratchStoreRoot scratch;
        ModelStore store( scratch.path() );

        // Caught before it becomes a URL: a path-shaped name would otherwise 404 against a
        // repository that cannot exist, and say nothing about the actual mistake.
        const ExplodingHub exploding;
        ModelResolver resolver( store, exploding );

        EXPECT_THROW( resolver.pull( "not a name", "mila-llm" ), std::runtime_error );
        EXPECT_THROW( resolver.pull( "some/path", "mila-llm" ), std::runtime_error );
    }
}
