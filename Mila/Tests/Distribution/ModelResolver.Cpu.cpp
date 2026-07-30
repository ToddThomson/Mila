/**
 * @file ModelResolver.Cpu.cpp
 * @brief Coordinate parsing, manifest validation, version skew and path passthrough.
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
        class ScratchCacheRoot
        {
        public:

            ScratchCacheRoot()
            {
                static int counter = 0;

                path_ = std::filesystem::temp_directory_path()
                    / std::format( "mila_resolver_test_{}", counter++ );

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

        const std::string kWeightsPayload = "pretend this is a safetensors artifact";
        const std::string kTokenizerPayload = "pretend this is a tokenizer";

        std::string manifestJson( const std::string& minimum_version = "0.1.0" )
        {
            return std::format( R"({{
              "manifest_version": 1,
              "architecture": "gemma",
              "default_variant": "fp4",
              "variants": {{
                "fp4": {{
                  "minimum_mila_version": "{}",
                  "weight_quantization": "per_group_fp4_128",
                  "files": {{
                    "weights":   {{ "path": "model_fp4.safetensors", "sha256": "{}" }},
                    "tokenizer": {{ "path": "tokenizer.bin",         "sha256": "{}" }}
                  }}
                }},
                "fp8": {{
                  "weight_quantization": "per_channel_fp8_e4m3",
                  "files": {{
                    "weights": {{ "path": "model_fp8.safetensors", "sha256": "{}" }}
                  }}
                }}
              }}
            }})",
                minimum_version,
                sha256Hex( kWeightsPayload.data(), kWeightsPayload.size() ),
                sha256Hex( kTokenizerPayload.data(), kTokenizerPayload.size() ),
                sha256Hex( kWeightsPayload.data(), kWeightsPayload.size() ) );
        }

        /// Serves the given manifest, and payloads keyed off the URL's filename.
        RemoteAccess memoryAccess( std::string manifest, int* manifest_fetches = nullptr )
        {
            RemoteAccess access;

            access.fetch_text = [manifest = std::move( manifest ), manifest_fetches](
                const std::string& url ) -> std::string
                {
                    if ( url.ends_with( "mila.json" ) )
                    {
                        if ( manifest_fetches != nullptr )
                        {
                            ++( *manifest_fetches );
                        }

                        return manifest;
                    }

                    throw std::runtime_error( "unexpected text fetch: " + url );
                };

            access.fetch_blob = []( const std::string& url, uint64_t resume_from,
                const std::function<bool( const char*, size_t )>& sink ) -> FetchReport
                {
                    const std::string& payload = url.ends_with( "tokenizer.bin" )
                        ? kTokenizerPayload
                        : kWeightsPayload;

                    const std::string remainder = payload.substr( static_cast<size_t>( resume_from ) );
                    sink( remainder.data(), remainder.size() );

                    return { FetchOutcome::Complete, {} };
                };

            return access;
        }
    }

    // ================================================================
    // Coordinate grammar
    // ================================================================

    TEST( ModelCoordinateParsing, AcceptsTheFullForm )
    {
        const auto coordinate = parseCoordinate( "mila-llm/gemma-4-12b-it:fp4@v2" );

        ASSERT_TRUE( coordinate.has_value() );
        EXPECT_EQ( coordinate->organization, "mila-llm" );
        EXPECT_EQ( coordinate->repository, "gemma-4-12b-it" );
        EXPECT_EQ( coordinate->variant, "fp4" );
        EXPECT_EQ( coordinate->revision, "v2" );
    }

    TEST( ModelCoordinateParsing, DefaultsRevisionAndLeavesVariantEmpty )
    {
        const auto coordinate = parseCoordinate( "mila-llm/gemma-4-12b-it" );

        ASSERT_TRUE( coordinate.has_value() );
        EXPECT_EQ( coordinate->variant, "" );
        EXPECT_EQ( coordinate->revision, "main" );
    }

    TEST( ModelCoordinateParsing, StripsTheExplicitPrefix )
    {
        const auto coordinate = parseCoordinate( "hf:mila-llm/gemma-4-12b-it:fp4" );

        ASSERT_TRUE( coordinate.has_value() );
        EXPECT_EQ( coordinate->organization, "mila-llm" );
        EXPECT_EQ( coordinate->variant, "fp4" );
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
    // Local path passthrough
    // ================================================================

    TEST( ModelResolverTests, ReturnsAnExistingPathWithoutTouchingTheNetwork )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );

        // A local file must not be re-downloaded. Any remote call here is a failure.
        RemoteAccess exploding;
        exploding.fetch_text = []( const std::string& url ) -> std::string
            {
                ADD_FAILURE() << "resolver reached the network for a local path: " << url;

                return {};
            };
        exploding.fetch_blob = []( const std::string&, uint64_t,
            const std::function<bool( const char*, size_t )>& ) -> FetchReport
            {
                ADD_FAILURE() << "resolver fetched a blob for a local path";

                return { FetchOutcome::Failed, {} };
            };

        const auto local = scratch.path() / "already_have_this.safetensors";
        std::filesystem::create_directories( local.parent_path() );
        {
            std::unique_ptr<std::FILE, int( * )( std::FILE* )> file(
                std::fopen( local.string().c_str(), "wb" ), &std::fclose );
            ASSERT_NE( file.get(), nullptr );
            std::fwrite( "x", 1, 1, file.get() );
        }

        ModelResolver resolver( cache, exploding );
        const auto resolved = resolver.resolve( local.string() );

        EXPECT_TRUE( resolved.from_local_path );
        EXPECT_EQ( resolved.weights_path, local );
        EXPECT_TRUE( resolved.tokenizer_path.empty() );
    }

    TEST( ModelResolverTests, RefusesASpecThatIsNeitherPathNorCoordinate )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );
        ModelResolver resolver( cache, memoryAccess( manifestJson() ) );

        EXPECT_THROW( resolver.resolve( "not/a/real/thing.safetensors" ), std::runtime_error );
    }

    // ================================================================
    // Coordinate resolution
    // ================================================================

    TEST( ModelResolverTests, ResolvesACoordinateAndCachesBothFiles )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );
        ModelResolver resolver( cache, memoryAccess( manifestJson() ) );

        const auto resolved = resolver.resolve( "mila-llm/gemma-4-12b-it:fp4" );

        EXPECT_FALSE( resolved.from_local_path );
        EXPECT_EQ( resolved.variant, "fp4" );
        EXPECT_EQ( resolved.architecture, "gemma" );
        EXPECT_EQ( resolved.weight_quantization, "per_group_fp4_128" );

        EXPECT_TRUE( std::filesystem::exists( resolved.weights_path ) );
        EXPECT_TRUE( std::filesystem::exists( resolved.tokenizer_path ) );

        // Content-addressed: the filename is the digest, not the repository path.
        EXPECT_EQ( resolved.weights_path,
            cache.blobPath( sha256Hex( kWeightsPayload.data(), kWeightsPayload.size() ) ) );
    }

    TEST( ModelResolverTests, UsesTheDefaultVariantWhenNoneIsGiven )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );
        ModelResolver resolver( cache, memoryAccess( manifestJson() ) );

        const auto resolved = resolver.resolve( "mila-llm/gemma-4-12b-it" );

        EXPECT_EQ( resolved.variant, "fp4" );
    }

    TEST( ModelResolverTests, ResolvesAVariantWithNoTokenizer )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );
        ModelResolver resolver( cache, memoryAccess( manifestJson() ) );

        const auto resolved = resolver.resolve( "mila-llm/gemma-4-12b-it:fp8" );

        EXPECT_EQ( resolved.variant, "fp8" );
        EXPECT_TRUE( std::filesystem::exists( resolved.weights_path ) );
        EXPECT_TRUE( resolved.tokenizer_path.empty() );
    }

    TEST( ModelResolverTests, NamesTheAvailableVariantsWhenOneIsMissing )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );
        ModelResolver resolver( cache, memoryAccess( manifestJson() ) );

        try
        {
            resolver.resolve( "mila-llm/gemma-4-12b-it:int2" );
            FAIL() << "expected a throw";
        }
        catch ( const std::runtime_error& error )
        {
            const std::string message = error.what();

            // Listing what is there turns a typo into a one-line fix.
            EXPECT_NE( message.find( "int2" ), std::string::npos );
            EXPECT_NE( message.find( "fp4" ), std::string::npos );
            EXPECT_NE( message.find( "fp8" ), std::string::npos );
        }
    }

    TEST( ModelResolverTests, RefusesAnArtifactRequiringANewerMila )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );
        ModelResolver resolver( cache, memoryAccess( manifestJson( "99.0.0" ) ) );

        // Better a version comparison than a parse error deep inside the tensor index, which
        // is what a future format change would otherwise look like.
        EXPECT_THROW( resolver.resolve( "mila-llm/gemma-4-12b-it:fp4" ), std::runtime_error );
    }

    TEST( ModelResolverTests, RejectsAMalformedManifest )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );

        ModelResolver not_json( cache, memoryAccess( "this is not json" ) );
        EXPECT_THROW( not_json.resolve( "mila-llm/gemma-4-12b-it:fp4" ), std::runtime_error );

        ModelResolver no_variants( cache, memoryAccess( R"({"architecture":"gemma"})" ) );
        EXPECT_THROW( no_variants.resolve( "mila-llm/gemma-4-12b-it:fp4" ), std::runtime_error );

        ModelResolver no_digest( cache, memoryAccess(
            R"({"variants":{"fp4":{"files":{"weights":{"path":"m.safetensors"}}}}})" ) );
        EXPECT_THROW( no_digest.resolve( "mila-llm/gemma-4-12b-it:fp4" ), std::runtime_error );
    }

    TEST( ModelResolverTests, SecondResolutionReusesTheCachedBlobs )
    {
        ScratchCacheRoot scratch;
        ModelCache cache( scratch.path() );

        int manifest_fetches = 0;
        ModelResolver resolver( cache, memoryAccess( manifestJson(), &manifest_fetches ) );

        const auto first = resolver.resolve( "mila-llm/gemma-4-12b-it:fp4" );
        const auto second = resolver.resolve( "mila-llm/gemma-4-12b-it:fp4" );

        EXPECT_EQ( first.weights_path, second.weights_path );

        // The manifest is still fetched each time -- an open decision in the spec. The blobs
        // are not, which is the part that matters at 6.33 GB.
        EXPECT_EQ( manifest_fetches, 2 );
    }
}
