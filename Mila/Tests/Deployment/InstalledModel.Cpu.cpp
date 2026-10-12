/**
 * @file InstalledModel.Cpu.cpp
 * @brief An installed model by name: what the library answers before anything loads, and the manifest fields that
 *        answer comes from.
 *
 * Specifications/Deployment.md 8.1 and ModelDistribution.md, Capabilities and modalities. Runs offline against a
 * scratch store; nothing here reads weights.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

import Mila;

namespace Mila::Tests::Deployment
{
    using namespace Mila::Dnn;
    using namespace Mila::Deployment;
    using namespace Mila::Distribution;

    namespace
    {
        /// A store root under the temp directory, removed on destruction.
        class ScratchStore
        {
        public:

            ScratchStore()
            {
                static int counter = 0;

                root_ = std::filesystem::temp_directory_path() / std::format( "mila_installed_model_test_{}", counter++ );

                std::error_code ignored;
                std::filesystem::remove_all( root_, ignored );
            }

            ~ScratchStore()
            {
                std::error_code ignored;
                std::filesystem::remove_all( root_, ignored );
            }

            ModelStore store() const { return ModelStore( root_ ); }

            /// Record a model whose files are all present, each a few bytes under its own digest.
            void install( ModelRecord record, bool with_drafter = false ) const
            {
                ModelStore target = store();

                for ( const char* role : { "weights", "tokenizer", "drafter" } )
                {
                    if ( std::string( role ) == "drafter" && !with_drafter )
                        continue;

                    const std::string payload = record.name + role;
                    const std::string digest = sha256Hex( payload.data(), payload.size() );

                    std::filesystem::create_directories( target.blobPath( digest ).parent_path() );

                    {
                        std::FILE* file = std::fopen( target.blobPath( digest ).string().c_str(), "wb" );
                        std::fwrite( payload.data(), 1, payload.size(), file );
                        std::fclose( file );
                    }

                    ModelFile entry;
                    entry.role = role;
                    entry.path = std::string( role ) + ".bin";
                    entry.sha256 = digest;
                    entry.bytes = payload.size();

                    record.files.push_back( entry );
                }

                target.writeRecord( record );
            }

        private:

            std::filesystem::path root_;
        };

        ModelRecord recordOf( std::string name, std::string architecture, std::string weight_quantization )
        {
            ModelRecord record;
            record.name = std::move( name );
            record.architecture = std::move( architecture );
            record.weight_quantization = std::move( weight_quantization );
            record.capabilities.instruct = true;

            return record;
        }

        const char* kMinimalFiles = R"("files": { "weights": { "path": "w.safetensors", "sha256": "aa", "bytes": 1 } })";
    }

    // ================================================================
    // The weight format a recorded name declares
    // ================================================================

    TEST( WeightQuantizationNames, ParseIsTheInverseOfTheName )
    {
        for ( const WeightQuantization format : { WeightQuantization::None, WeightQuantization::FP8,
                  WeightQuantization::FP4, WeightQuantization::Q4_0, WeightQuantization::Plan } )
        {
            EXPECT_EQ( parseWeightQuantization( weightQuantizationName( format ) ), format )
                << weightQuantizationName( format );
        }
    }

    TEST( WeightQuantizationNames, ReadsEveryFp4GroupAsFp4AndRefusesAnUnknownScheme )
    {
        // The group is the model's geometry, not a choice, so both groups are one format.
        EXPECT_EQ( parseWeightQuantization( "per_group_fp4_64" ), WeightQuantization::FP4 );
        EXPECT_EQ( parseWeightQuantization( "" ), WeightQuantization::None );
        EXPECT_FALSE( parseWeightQuantization( "per_group_int3_16" ).has_value() );
    }

    // ================================================================
    // Capabilities in the manifest
    // ================================================================

    TEST( ModelCapabilitiesManifest, RoundTripsEveryCapabilityAndModality )
    {
        ModelManifest manifest;
        manifest.name = "gemma-4-12b-it-q4_0";
        manifest.architecture = "gemma";
        manifest.capabilities.instruct = true;
        manifest.capabilities.thinking = true;
        manifest.capabilities.tools = true;
        manifest.capabilities.agentic = true;
        manifest.capabilities.modalities = { "image", "audio" };

        ModelFile weights;
        weights.role = "weights";
        weights.path = "w.safetensors";
        weights.sha256 = "aa";
        manifest.files.push_back( weights );

        const ModelManifest read_back = parseModelManifest( toJsonText( manifest ), "test" );

        EXPECT_TRUE( read_back.capabilities.instruct );
        EXPECT_TRUE( read_back.capabilities.thinking );
        EXPECT_TRUE( read_back.capabilities.tools );
        EXPECT_TRUE( read_back.capabilities.agentic );

        // Text is every model's, so it is written for a reader and never read back as a modality.
        EXPECT_EQ( read_back.capabilities.modalities, ( std::vector<std::string>{ "image", "audio" } ) );
    }

    TEST( ModelCapabilitiesManifest, KeepsTheTopLevelInstructAnOlderMilaReads )
    {
        ModelManifest manifest;
        manifest.name = "llama";
        manifest.architecture = "llama";
        manifest.capabilities.instruct = true;

        const std::string text = toJsonText( manifest );

        EXPECT_NE( text.find( "\"instruct\": true,\n  \"capabilities\"" ), std::string::npos ) << text;
    }

    TEST( ModelCapabilitiesManifest, ReadsAManifestWrittenBeforeCapabilitiesExisted )
    {
        const std::string text = std::format( R"({{ "name": "old", "architecture": "llama", "instruct": true, {} }})",
            kMinimalFiles );

        const ModelManifest manifest = parseModelManifest( text, "test" );

        EXPECT_TRUE( manifest.capabilities.instruct );
        EXPECT_FALSE( manifest.capabilities.thinking );
        EXPECT_FALSE( manifest.capabilities.tools );
        EXPECT_FALSE( manifest.capabilities.agentic );
        EXPECT_TRUE( manifest.capabilities.modalities.empty() );
    }

    // ================================================================
    // An installed model, by name
    // ================================================================

    TEST( InstalledModel, FindsAModelWhateverCaseItIsTypedInAndAnswersWithTheStoresSpelling )
    {
        ScratchStore scratch;
        scratch.install( recordOf( "Llama-3.1-8B-Instruct-fp4", "llama", "per_group_fp4_128" ) );

        const auto model = InstalledModel::find( "LLAMA-3.1-8B-INSTRUCT-FP4", scratch.store() );

        ASSERT_TRUE( model.has_value() );
        EXPECT_EQ( model->name(), "Llama-3.1-8B-Instruct-fp4" );
        EXPECT_EQ( model->family(), ModelType::Llama );
        EXPECT_EQ( model->storedWeightQuantization(), WeightQuantization::FP4 );
        EXPECT_TRUE( model->isComplete() );
        EXPECT_EQ( model->attribution(), "" );

        EXPECT_FALSE( InstalledModel::find( "not-installed", scratch.store() ).has_value() );
    }

    TEST( InstalledModel, ReturnsAModelWhoseFilesAreMissingAndSaysSo )
    {
        ScratchStore scratch;
        ModelStore store = scratch.store();

        ModelRecord record = recordOf( "gemma-4-12b-it-fp4", "gemma", "per_group_fp4_128" );
        ModelFile weights;
        weights.role = "weights";
        weights.path = "w.safetensors";
        weights.sha256 = "0000";
        record.files.push_back( weights );
        store.writeRecord( record );

        const auto model = InstalledModel::find( "gemma-4-12b-it-fp4", store );

        ASSERT_TRUE( model.has_value() );
        EXPECT_FALSE( model->isComplete() );
        EXPECT_THROW( requirePlannable( *model ), std::runtime_error );
    }

    TEST( InstalledModel, ReportsADrafterOnlyWhereThisBuildDraftsForTheFamily )
    {
        ScratchStore scratch;
        scratch.install( recordOf( "gemma-4-12b-it-q4_0", "gemma", "q4_0" ), true );
        scratch.install( recordOf( "llama-with-drafter", "llama", "none" ), true );
        scratch.install( recordOf( "gemma-without", "gemma", "q4_0" ) );

        EXPECT_TRUE( InstalledModel::find( "gemma-4-12b-it-q4_0", scratch.store() )->drafter().has_value() );
        EXPECT_FALSE( InstalledModel::find( "llama-with-drafter", scratch.store() )->drafter().has_value() );
        EXPECT_FALSE( InstalledModel::find( "gemma-without", scratch.store() )->drafter().has_value() );
    }

    TEST( InstalledModel, ReportsOnlyTheModalitiesThisBuildRuns )
    {
        ScratchStore scratch;

        ModelRecord record = recordOf( "gemma-4-12b-it-q4_0", "gemma", "q4_0" );
        record.capabilities.thinking = true;
        record.capabilities.modalities = { "image", "audio", "smell" };
        scratch.install( record );

        const ModelCapabilities capabilities =
            InstalledModel::find( "gemma-4-12b-it-q4_0", scratch.store() )->capabilities();

        EXPECT_TRUE( capabilities.thinking );

        for ( const auto& modality : capabilities.modalities )
        {
            EXPECT_TRUE( isSupportedModality( modality ) ) << modality;
        }

        EXPECT_EQ( std::ranges::count( capabilities.modalities, std::string( "smell" ) ), 0 );
    }

    TEST( InstalledModel, RefusesToPlanAFamilyThisBuildCannotDeploy )
    {
        ScratchStore scratch;
        scratch.install( recordOf( "gpt2-small", "gpt2", "none" ) );

        const auto model = InstalledModel::find( "gpt2-small", scratch.store() );

        ASSERT_TRUE( model.has_value() );
        EXPECT_EQ( model->family(), ModelType::Gpt2 );
        EXPECT_THROW( requirePlannable( *model ), std::runtime_error );
    }

    TEST( InstalledModel, NamesTheAttributionALlamaLicenceRequires )
    {
        ScratchStore scratch;

        ModelRecord record = recordOf( "Llama-3.2-3B-Instruct-fp4", "llama", "per_group_fp4_128" );
        record.license = "llama3.2";
        scratch.install( record );

        EXPECT_EQ( InstalledModel::find( "Llama-3.2-3B-Instruct-fp4", scratch.store() )->attribution(),
            "Built with Llama" );
    }

    // ================================================================
    // A request that names no weight format
    // ================================================================

    TEST( DeploymentRequestWeightFormat, TakesTheStoredFormatWhenTheRequestNamesNone )
    {
        const DeploymentRequest resolved = DeploymentRequest{}.resolveWeightQuantization( "per_group_fp4_128" );

        EXPECT_EQ( resolved.getWeightQuantization(), WeightQuantization::FP4 );
        EXPECT_EQ( DeploymentRequest{}.resolveWeightQuantization( "" ).getWeightQuantization(),
            WeightQuantization::None );
    }

    TEST( DeploymentRequestWeightFormat, KeepsAFormatTheRequestNamesEvenReferenceWeights )
    {
        // Named, so not the package's: the load checks it against the stored one.
        EXPECT_EQ( DeploymentRequest{}.withFP8Quantization().resolveWeightQuantization( "" ).getWeightQuantization(),
            WeightQuantization::FP8 );
        EXPECT_EQ( DeploymentRequest{}.withFullPrecision().resolveWeightQuantization( "q4_0" ).getWeightQuantization(),
            WeightQuantization::None );
    }

    TEST( DeploymentRequestWeightFormat, RefusesAStoredSchemeThisBuildDoesNotRead )
    {
        EXPECT_THROW( (void)DeploymentRequest{}.resolveWeightQuantization( "per_group_int3_16" ), std::runtime_error );
    }
}
