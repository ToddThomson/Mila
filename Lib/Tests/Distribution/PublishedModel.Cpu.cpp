/**
 * @file PublishedModel.Cpu.cpp
 * @brief A published model's listing, the start of its weights read into the store to plan it, why one would not
 *        open, and a folder of packages served as a hub; against a hub that serves from memory, and a scratch folder.
 *
 * Planning itself needs a device and real weights, so it is checked against the publisher on a machine, not here
 * (Specifications/Apps/AssistantOnboarding.md).
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

import Mila;

namespace Mila::Tests::Distribution
{
    using namespace Mila::Distribution;
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Serialization;

    namespace
    {
        class ScratchDirectory
        {
        public:

            ScratchDirectory()
            {
                static int counter = 0;

                path_ = std::filesystem::temp_directory_path() / std::format( "mila_published_test_{}", counter++ );

                std::error_code ignored;
                std::filesystem::remove_all( path_, ignored );
                std::filesystem::create_directories( path_ );
            }

            ~ScratchDirectory()
            {
                std::error_code ignored;
                std::filesystem::remove_all( path_, ignored );
            }

            const std::filesystem::path& path() const { return path_; }

        private:

            std::filesystem::path path_;
        };

        /// A small safetensors file's bytes, written by the library's own writer.
        std::string weightsBytes( const std::filesystem::path& directory )
        {
            const std::filesystem::path path = directory / "weights.safetensors";
            const std::vector<float> values{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };

            {
                SafeTensorsWriter writer( path );
                writer.declareTensor( "w", TensorDataType::FP32, shape_t{ 8 } );
                writer.setMetadata( kMilaConfigMetadataKey, R"({"architecture":"gemma","model_name":"tiny","num_layers":3})" );
                writer.beginData();
                writer.writeTensorData( "w", values.data(), values.size() * sizeof( float ) );
                writer.close();
            }

            std::string bytes( static_cast<std::size_t>( std::filesystem::file_size( path ) ), '\0' );
            std::unique_ptr<std::FILE, int( * )( std::FILE* )> file( std::fopen( path.string().c_str(), "rb" ), &std::fclose );
            std::fread( bytes.data(), 1, bytes.size(), file.get() );

            return bytes;
        }

        std::string manifestFor( const std::string& name, const std::string& architecture, const std::string& weights )
        {
            return std::format( R"({{
              "manifest_version": 1,
              "name": "{}",
              "architecture": "{}",
              "weight_quantization": "none",
              "license": "apache-2.0",
              "capabilities": {{ "instruct": true, "agentic": true }},
              "files": {{
                "weights": {{ "path": "model.safetensors", "sha256": "{}", "bytes": {} }}
              }}
            }})", name, architecture, sha256Hex( weights.data(), weights.size() ), weights.size() );
        }

        /// Serves a listing, manifests by repository and one weights file; counts what it is asked.
        class MemoryHub : public IModelHub
        {
        public:

            std::string name() const override { return "memory"; }

            std::vector<HubModel> listModels( const std::string& owner ) const override
            {
                std::vector<HubModel> listed;

                for ( const auto& [repository, files] : repositories )
                {
                    HubModel model;
                    model.owner = owner;
                    model.repository = repository;
                    model.revision = "abc123";
                    model.files = files;
                    listed.push_back( model );
                }

                return listed;
            }

            std::string fetchManifest( const ModelCoordinate& coordinate ) const override
            {
                const auto found = manifests.find( coordinate.repository );

                if ( found == manifests.end() )
                {
                    throw std::runtime_error( "404" );
                }

                return found->second;
            }

            FetchReport fetchFile( const ModelCoordinate&, const std::string&, uint64_t,
                const std::function<bool( const char*, size_t )>& ) const override
            {
                ADD_FAILURE() << "planning downloaded a whole file";

                return { FetchOutcome::Failed, {} };
            }

            std::string fetchFilePart( const ModelCoordinate& coordinate, const std::string&, uint64_t first,
                uint64_t count ) const override
            {
                ++part_reads;
                revisions_read.push_back( coordinate.revision );

                return weights.substr( static_cast<size_t>( first ), static_cast<size_t>( count ) );
            }

            std::map<std::string, std::vector<std::string>> repositories;
            std::map<std::string, std::string> manifests;
            std::string weights;
            mutable int part_reads{ 0 };
            mutable std::vector<std::string> revisions_read;
        };
    }

    TEST( PublishedModel, ListsEachModelWithAManifestAndKeepsOneUnreadByName )
    {
        ScratchDirectory scratch;
        MemoryHub hub;
        hub.weights = weightsBytes( scratch.path() );
        hub.repositories[ "gemma-tiny" ] = { "mila.json", "model.safetensors" };
        hub.repositories[ "unreadable" ] = { "mila.json" };
        hub.repositories[ "not-for-mila" ] = { "README.md" };
        hub.manifests[ "gemma-tiny" ] = manifestFor( "gemma-tiny", "gemma", hub.weights );

        const std::vector<PublishedModel> listed = PublishedModel::list( hub, "owner" );

        ASSERT_EQ( listed.size(), 2u ) << "a repository without a manifest is not a model";

        const auto& readable = listed[ 0 ].name() == "gemma-tiny" ? listed[ 0 ] : listed[ 1 ];
        const auto& unread = listed[ 0 ].name() == "gemma-tiny" ? listed[ 1 ] : listed[ 0 ];

        ASSERT_TRUE( readable.manifest().has_value() );
        EXPECT_TRUE( readable.manifest()->capabilities.agentic );
        EXPECT_EQ( readable.downloadBytes(), hub.weights.size() );
        EXPECT_EQ( readable.coordinate().revision, "abc123" );
        EXPECT_FALSE( unread.manifest().has_value() );
        EXPECT_EQ( unread.downloadBytes(), 0u );
    }

    TEST( PublishedModel, ReadsTheStartOfItsWeightsIntoTheStoreOnce )
    {
        ScratchDirectory scratch;
        MemoryHub hub;
        hub.weights = weightsBytes( scratch.path() );

        const ModelStore store( scratch.path() / "store" );
        const PublishedModel model( ModelCoordinate{ "owner", "gemma-tiny", "abc123" },
            parseModelManifest( manifestFor( "gemma-tiny", "gemma", hub.weights ), "test" ) );

        const std::filesystem::path header = model.weightsHeader( hub, store );

        EXPECT_EQ( hub.part_reads, 2 ) << "the length, then the header";
        EXPECT_EQ( hub.revisions_read.front(), "abc123" );
        EXPECT_LT( std::filesystem::file_size( header ), hub.weights.size() );

        WeightsReader reader( header );
        EXPECT_TRUE( reader.holdsHeaderOnly() );
        EXPECT_EQ( reader.getWeightsMetadata().num_layers, 3u );
        EXPECT_EQ( reader.getTensorSizeBytes( "w" ), 8 * sizeof( float ) );

        EXPECT_EQ( model.weightsHeader( hub, store ), header );
        EXPECT_EQ( hub.part_reads, 2 ) << "a header already read is not read again";
    }

    TEST( PublishedModel, RefusesWeightsThatDoNotBeginWithAHeader )
    {
        ScratchDirectory scratch;
        MemoryHub hub;
        hub.weights = std::string( 64, '\xff' );

        const ModelStore store( scratch.path() / "store" );
        const PublishedModel model( ModelCoordinate{ "owner", "broken" },
            parseModelManifest( manifestFor( "broken", "gemma", hub.weights ), "test" ) );

        EXPECT_THROW( (void)model.weightsHeader( hub, store ), std::runtime_error );
        EXPECT_FALSE( std::filesystem::exists( scratch.path() / "store" / "headers" ) ) << "nothing is kept of it";
    }

    TEST( PublishedModel, SaysWhyOneWouldNotOpenBeforeReadingItsWeights )
    {
        ScratchDirectory scratch;
        MemoryHub hub;
        hub.weights = weightsBytes( scratch.path() );

        const PublishedModel unread( ModelCoordinate{ "owner", "unread" }, std::nullopt );
        const PublishedModel foreign( ModelCoordinate{ "owner", "gpt2-tiny" },
            parseModelManifest( manifestFor( "gpt2-tiny", "gpt2", hub.weights ), "test" ) );

        const auto unread_refusal = Mila::Deployment::ModelHandle::check( unread );
        ASSERT_TRUE( unread_refusal.has_value() );
        EXPECT_EQ( unread_refusal->reason, Mila::Deployment::OpenRefusal::Reason::ManifestUnread );

        const auto foreign_refusal = Mila::Deployment::ModelHandle::check( foreign );
        ASSERT_TRUE( foreign_refusal.has_value() );
        EXPECT_EQ( foreign_refusal->reason, Mila::Deployment::OpenRefusal::Reason::ArchitectureNotDeployed );

        const ModelStore store( scratch.path() / "store" );
        EXPECT_THROW( (void)Mila::Deployment::ModelHandle::planFor( foreign, {}, hub, store ),
            Mila::Deployment::OpenRefusedError );
        EXPECT_EQ( hub.part_reads, 0 ) << "a model that would not open is refused before its weights are read";
    }

    namespace
    {
        void writeFile( const std::filesystem::path& path, const std::string& bytes )
        {
            std::filesystem::create_directories( path.parent_path() );
            std::unique_ptr<std::FILE, int( * )( std::FILE* )> file( std::fopen( path.string().c_str(), "wb" ), &std::fclose );
            ASSERT_NE( file.get(), nullptr );
            std::fwrite( bytes.data(), 1, bytes.size(), file.get() );
        }

        /// A folder holding one package, laid out as ExportArtifact --package writes one, and a folder that is not one.
        std::string writePackageFolder( const std::filesystem::path& root, const std::filesystem::path& scratch )
        {
            const std::string weights = weightsBytes( scratch );

            writeFile( root / "gemma-tiny" / "model.safetensors", weights );
            writeFile( root / "gemma-tiny" / "mila.json", manifestFor( "gemma-tiny", "gemma", weights ) );
            writeFile( root / "notes" / "README.md", "not a model" );

            return weights;
        }
    }

    TEST( LocalDirectoryHub, ListsEachPackageInTheFolderAndServesItsFiles )
    {
        ScratchDirectory scratch;
        const std::filesystem::path root = scratch.path() / "hub";
        const std::string weights = writePackageFolder( root, scratch.path() );

        std::vector<std::uint64_t> progress;
        const LocalDirectoryHub hub( root, [&progress]( std::uint64_t received, std::uint64_t ) { progress.push_back( received ); return true; } );

        const auto listed = hub.listModels( "anyone" );
        ASSERT_EQ( listed.size(), 1u ) << "a folder without a manifest is not a model";
        EXPECT_EQ( listed[ 0 ].repository, "gemma-tiny" );
        EXPECT_TRUE( listed[ 0 ].hasManifest() );

        const ModelCoordinate coordinate{ "anyone", "gemma-tiny" };
        EXPECT_EQ( parseModelManifest( hub.fetchManifest( coordinate ), "test" ).name, "gemma-tiny" );
        EXPECT_EQ( hub.fetchFilePart( coordinate, "model.safetensors", 8, 16 ), weights.substr( 8, 16 ) );

        std::string resumed;
        const FetchReport report = hub.fetchFile( coordinate, "model.safetensors", 10,
            [&resumed]( const char* data, size_t length ) { resumed.append( data, length ); return true; } );

        EXPECT_EQ( report.outcome, FetchOutcome::Complete );
        EXPECT_EQ( resumed, weights.substr( 10 ) );
        ASSERT_FALSE( progress.empty() );
        EXPECT_EQ( progress.back(), weights.size() );
    }

    TEST( LocalDirectoryHub, RefusesAPathThatLeavesItsPackage )
    {
        ScratchDirectory scratch;
        const std::filesystem::path root = scratch.path() / "hub";
        writePackageFolder( root, scratch.path() );
        writeFile( scratch.path() / "secret.txt", "outside" );

        const LocalDirectoryHub hub( root );

        EXPECT_THROW( (void)hub.fetchFilePart( ModelCoordinate{ "o", "gemma-tiny" }, "../../secret.txt", 0, 7 ), std::runtime_error );
        EXPECT_THROW( (void)hub.fetchManifest( ModelCoordinate{ "o", ".." } ), std::runtime_error );
        EXPECT_THROW( (void)hub.fetchManifest( ModelCoordinate{ "o", "absent" } ), std::runtime_error );
    }

    TEST( LocalDirectoryHub, InstallsAndPlansAsTheNetworkHubDoes )
    {
        ScratchDirectory scratch;
        const std::filesystem::path root = scratch.path() / "hub";
        const std::string weights = writePackageFolder( root, scratch.path() );
        const LocalDirectoryHub hub( root );

        ModelStore store( scratch.path() / "store" );
        ModelResolver resolver( store, hub );
        const auto pulled = resolver.pull( "gemma-tiny", "anyone" );

        EXPECT_EQ( pulled.record.name, "gemma-tiny" );
        EXPECT_GE( pulled.bytes_on_disk, weights.size() );

        const auto listed = PublishedModel::list( hub, "anyone" );
        ASSERT_EQ( listed.size(), 1u );
        EXPECT_TRUE( WeightsReader( listed[ 0 ].weightsHeader( hub, store ) ).holdsHeaderOnly() );
    }

    TEST( PublishedModel, RanksTheModelsNoneOfWhichOpensByName )
    {
        ScratchDirectory scratch;
        MemoryHub hub;
        hub.weights = weightsBytes( scratch.path() );
        hub.repositories[ "zeta-gpt2" ] = { "mila.json" };
        hub.repositories[ "alpha-gpt2" ] = { "mila.json" };
        hub.manifests[ "zeta-gpt2" ] = manifestFor( "zeta-gpt2", "gpt2", hub.weights );
        hub.manifests[ "alpha-gpt2" ] = manifestFor( "alpha-gpt2", "gpt2", hub.weights );

        const ModelStore store( scratch.path() / "store" );
        const auto fits = Mila::Deployment::planPublishedModels( hub, "owner", {}, store );

        ASSERT_EQ( fits.size(), 2u );
        EXPECT_EQ( fits[ 0 ].model.name(), "alpha-gpt2" );
        EXPECT_EQ( fits[ 1 ].model.name(), "zeta-gpt2" );

        for ( const auto& fit : fits )
        {
            EXPECT_TRUE( fit.refusal.has_value() );
            EXPECT_FALSE( fit.suggestable() );
            EXPECT_FALSE( fit.agenticHere() );
        }
    }
}
