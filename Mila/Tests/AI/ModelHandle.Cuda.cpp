/**
 * @file ModelHandle.Cuda.cpp
 * @brief A model opened by name runs as its concrete type does, and a name the list cannot open is refused before any
 *        load (Specifications/ModelHandle.md section 6: H1, N1, N2).
 *
 * H1 reads installed models from the model store, read only, and skips a family with none installed.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

import Mila;

namespace Mila::Tests::AI
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;
    using namespace Mila::Distribution;

    using Handle = Mila::AI::ModelHandle<DeviceType::Cuda, TensorDataType::BF16>;
    using CudaLanguageModel = LanguageModel<DeviceType::Cuda, TensorDataType::BF16>;

    namespace
    {
        constexpr dim_t kContextLength = 4096;

        /// A store root under the temp directory, removed on destruction.
        class ScratchStore
        {
        public:

            ScratchStore()
            {
                static int counter = 0;

                root_ = std::filesystem::temp_directory_path() / std::format( "mila_model_handle_test_{}", counter++ );

                std::error_code ignored;
                std::filesystem::remove_all( root_, ignored );
            }

            ~ScratchStore()
            {
                std::error_code ignored;
                std::filesystem::remove_all( root_, ignored );
            }

            ModelStore store() const { return ModelStore( root_ ); }

            /**
             * @brief Record an instruct model whose weights and tokenizer are each a few bytes, enough to be complete.
             *
             * @param weight_quantization The scheme the record declares; empty for reference weights.
             * @param complete            False leaves the weights' file out of the store.
             */
            void install( std::string name, std::string architecture, std::string architecture_version = {},
                std::string weight_quantization = {}, bool complete = true ) const
            {
                ModelStore target = store();

                ModelRecord record;
                record.name = std::move( name );
                record.architecture = std::move( architecture );
                record.architecture_version = std::move( architecture_version );
                record.weight_quantization = std::move( weight_quantization );
                record.capabilities.instruct = true;

                for ( const char* role : { "weights", "tokenizer" } )
                {
                    const std::string payload = record.name + role;
                    const std::string digest = sha256Hex( payload.data(), payload.size() );

                    std::filesystem::create_directories( target.blobPath( digest ).parent_path() );

                    if ( complete || std::string_view( role ) != "weights" )
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

        DeploymentRequest requestOn( int device_index )
        {
            DeploymentRequest request;
            request.withDevice( DeviceId{ DeviceType::Cuda, device_index } );
            request.withContextLength( kContextLength );

            return request;
        }

        std::vector<std::int32_t> greedy( CudaLanguageModel& model, const std::vector<std::int32_t>& prompt )
        {
            GenerateParams params;
            params.max_new_tokens = 24;
            params.sampling.temperature = 0.0f;
            params.sampling.top_k = 0;

            std::vector<std::int32_t> tokens;

            (void)model.generate( prompt, [&]( std::int32_t token ) { tokens.push_back( token ); }, params );

            return tokens;
        }

        /// The first of these names the store has, or nothing.
        std::optional<std::string> firstInstalled( std::initializer_list<const char*> names )
        {
            for ( const char* name : names )
            {
                if ( InstalledModel::find( name ).has_value() )
                {
                    return std::string( name );
                }
            }

            return std::nullopt;
        }

        /**
         * @brief Greedy tokens through the handle against the typed load of the same request, on every CUDA device
         *        that holds the model; the typed model's reach through as<>() checked on the way.
         *
         * @return Why nothing was compared, or nothing when every holding device compared.
         */
        template<typename TModel, typename TOtherFamily>
        std::optional<std::string> compareOnEveryDevice( std::initializer_list<const char*> names )
        {
            const std::optional<std::string> name = firstInstalled( names );

            if ( !name )
            {
                return std::format( "none of this family's models is installed in {}", ModelStore().root().string() );
            }

            const InstalledModel installed = *InstalledModel::find( *name );

            if ( installed.architecture() == "llama" && installed.stored().record.architecture_version.empty() )
            {
                return std::format( "'{}' was installed before architecture_version existed; reinstall it", *name );
            }

            std::string compared;

            for ( int device = 0; device < getDeviceCount( DeviceType::Cuda ); ++device )
            {
                const DeploymentRequest request = requestOn( device );

                if ( !Handle::planFor( *name, request ).has_value() )
                {
                    continue;
                }

                std::vector<std::int32_t> prompt;
                std::vector<std::int32_t> through_handle;

                // Released before the typed load, so the two never both hold the device.
                {
                    Handle handle = Handle::open( *name, request );

                    EXPECT_NE( handle.as<TModel>(), nullptr ) << *name;
                    EXPECT_EQ( handle.as<TOtherFamily>(), nullptr ) << *name << ": as<>() reached another family's type";
                    EXPECT_EQ( handle.plan().contextLength(), kContextLength ) << *name;
                    EXPECT_EQ( handle.name(), *name );

                    if ( handle.protocol() == nullptr )
                    {
                        ADD_FAILURE() << *name << " is an instruct model and opened with no chat template";

                        return std::nullopt;
                    }

                    const std::vector<Dnn::Conversation::Turn> history{
                        { Dnn::Conversation::Role::User, "Name three rivers in Europe." } };

                    const auto ids = handle.tokenizer().encode( handle.protocol()->formatPrompt( history, {} ).text );
                    prompt.assign( ids.begin(), ids.end() );

                    through_handle = greedy( handle.model(), prompt );
                }

                const auto typed = TModel::load( installed.weightsPath(), request );

                EXPECT_EQ( greedy( *typed, prompt ), through_handle ) << *name << " on CUDA device " << device;
                EXPECT_EQ( through_handle.size(), 24u ) << *name;

                compared += std::format( "{}{}", compared.empty() ? "" : ",", device );
            }

            if ( compared.empty() )
            {
                return std::format( "no CUDA device holds '{}' at context {}", *name, kContextLength );
            }

            ::testing::Test::RecordProperty( "compared_on_cuda_devices", compared );

            return std::nullopt;
        }
    }

    // ================================================================
    // H1, N2: the handle runs what the typed model runs, and reaches only its own type
    // ================================================================

    TEST( ModelHandleCuda, GemmaGeneratesThroughTheHandleAsThroughItsType )
    {
        using Gemma = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;
        using Llama = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;

        if ( const auto skipped = compareOnEveryDevice<Gemma, Llama>( { "gemma-4-12b-it-fp4", "gemma-4-12b-it-q4_0" } ) )
        {
            GTEST_SKIP() << *skipped;
        }
    }

    TEST( ModelHandleCuda, LlamaGeneratesThroughTheHandleAsThroughItsType )
    {
        using Llama = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;
        using Qwen = QwenModel<DeviceType::Cuda, TensorDataType::BF16>;

        if ( const auto skipped = compareOnEveryDevice<Llama, Qwen>(
                 { "Llama-3.2-3B-Instruct-fp4", "Llama-3.1-8B-Instruct-fp4" } ) )
        {
            GTEST_SKIP() << *skipped;
        }
    }

    TEST( ModelHandleCuda, QwenGeneratesThroughTheHandleAsThroughItsType )
    {
        using Qwen = QwenModel<DeviceType::Cuda, TensorDataType::BF16>;
        using Gemma = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;

        if ( const auto skipped = compareOnEveryDevice<Qwen, Gemma>( { "Qwen3.8-27B-fp4", "Qwen3.8-27B-fp4-cb3" } ) )
        {
            GTEST_SKIP() << *skipped;
        }
    }

    // ================================================================
    // N1: what the list cannot open is refused before anything loads
    // ================================================================

    TEST( ModelHandle, RefusesAnArchitectureNotInTheListAndNamesTheList )
    {
        ScratchStore scratch;
        scratch.install( "gpt2-small", "gpt2" );
        scratch.install( "mistral-7b", "mistral" );

        for ( const char* name : { "gpt2-small", "mistral-7b" } )
        {
            try
            {
                (void)Handle::open( name, requestOn( 0 ), scratch.store() );
                ADD_FAILURE() << name << " opened";
            }
            catch ( const std::runtime_error& error )
            {
                EXPECT_NE( std::string( error.what() ).find( "gemma, llama, qwen" ), std::string::npos ) << error.what();
            }

            EXPECT_THROW( (void)Handle::planFor( name, requestOn( 0 ), scratch.store() ), std::runtime_error ) << name;
        }
    }

    TEST( ModelHandle, RefusesANameNotInstalled )
    {
        ScratchStore scratch;

        EXPECT_THROW( (void)Handle::open( "not-installed", requestOn( 0 ), scratch.store() ), std::runtime_error );
    }

    TEST( ModelHandle, RefusesAnInstructModelWithNoChatTemplateBeforeReadingItsFiles )
    {
        // The files are a few bytes each: reading either would fail differently from the template's refusal.
        ScratchStore scratch;
        scratch.install( "llama-undeclared", "llama" );
        scratch.install( "qwen-4", "qwen", "4" );

        const auto refusedFor = [&scratch]( const char* name )
            {
                try
                {
                    (void)Handle::open( name, requestOn( 0 ), scratch.store() );
                }
                catch ( const Mila::AI::OpenRefusedError& error )
                {
                    return std::optional( error.refusal().reason );
                }

                return std::optional<Mila::AI::OpenRefusal::Reason>();
            };

        EXPECT_EQ( refusedFor( "llama-undeclared" ), Mila::AI::OpenRefusal::Reason::ChatTemplateNotRecorded );
        EXPECT_EQ( refusedFor( "qwen-4" ), Mila::AI::OpenRefusal::Reason::NoChatTemplate );
    }

    // ================================================================
    // The check: every refusal open would give, with nothing loaded
    // ================================================================

    namespace
    {
        std::optional<Mila::AI::OpenRefusal::Reason> checked(
            const ScratchStore& scratch, const char* name, const DeploymentRequest& request = requestOn( 0 ) )
        {
            return Handle::check( name, request, scratch.store() )
                .transform( []( const Mila::AI::OpenRefusal& refusal ) { return refusal.reason; } );
        }
    }

    TEST( ModelHandle, ChecksEachRefusalBesideAModelThatOpens )
    {
        using Reason = Mila::AI::OpenRefusal::Reason;

        ScratchStore scratch;
        scratch.install( "gemma-chat", "gemma" );
        scratch.install( "gemma-incomplete", "gemma", {}, {}, false );
        scratch.install( "gpt2-small", "gpt2" );
        scratch.install( "llama-undeclared", "llama" );
        scratch.install( "llama-31", "llama", "3.1" );
        scratch.install( "qwen-4", "qwen", "4" );
        scratch.install( "gemma-unknown-format", "gemma", {}, "per_tensor_int3" );

        EXPECT_EQ( checked( scratch, "gemma-chat" ), std::nullopt );
        EXPECT_EQ( checked( scratch, "llama-31" ), std::nullopt );

        EXPECT_EQ( checked( scratch, "not-installed" ), Reason::NotInstalled );
        EXPECT_EQ( checked( scratch, "gemma-incomplete" ), Reason::FilesMissing );
        EXPECT_EQ( checked( scratch, "gpt2-small" ), Reason::ArchitectureNotDeployed );
        EXPECT_EQ( checked( scratch, "llama-undeclared" ), Reason::ChatTemplateNotRecorded );
        EXPECT_EQ( checked( scratch, "qwen-4" ), Reason::NoChatTemplate );
        EXPECT_EQ( checked( scratch, "gemma-unknown-format" ), Reason::WeightFormatUnknown );
    }

    TEST( ModelHandle, ChecksAWeightFormatAskedForOnLoadAgainstTheStoredOne )
    {
        using Reason = Mila::AI::OpenRefusal::Reason;

        ScratchStore scratch;
        scratch.install( "gemma-reference", "gemma" );
        scratch.install( "gemma-q4_0", "gemma", {}, std::string( weightQuantizationName( WeightQuantization::Q4_0 ) ) );

        DeploymentRequest fp4 = requestOn( 0 );
        fp4.withWeightQuantization( WeightQuantization::FP4 );

        DeploymentRequest q4_0 = requestOn( 0 );
        q4_0.withWeightQuantization( WeightQuantization::Q4_0 );

        // Reference weights quantize to FP4 as they load; packed Q4_0 weights are not FP4, and are what they are.
        EXPECT_EQ( checked( scratch, "gemma-reference", fp4 ), std::nullopt );
        EXPECT_EQ( checked( scratch, "gemma-q4_0", fp4 ), Reason::WeightFormatNotDerivable );
        EXPECT_EQ( checked( scratch, "gemma-q4_0", q4_0 ), std::nullopt );
    }

    TEST( ModelHandle, OpenRefusesWhatTheCheckRefusesWithTheSameReason )
    {
        ScratchStore scratch;
        scratch.install( "gemma-incomplete", "gemma", {}, {}, false );

        try
        {
            (void)Handle::open( "gemma-incomplete", requestOn( 0 ), scratch.store() );
            ADD_FAILURE() << "an incomplete model opened";
        }
        catch ( const Mila::AI::OpenRefusedError& error )
        {
            EXPECT_EQ( error.refusal().reason, Mila::AI::OpenRefusal::Reason::FilesMissing );
            EXPECT_EQ( error.refusal().name, "gemma-incomplete" );
        }
    }
}
