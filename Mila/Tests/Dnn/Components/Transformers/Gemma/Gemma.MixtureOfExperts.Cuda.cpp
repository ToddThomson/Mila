/**
 * @file Gemma.MixtureOfExperts.Cuda.cpp
 * @brief GemmaTransformer with kMixtureOfExperts against a tiny HuggingFace Gemma 4 MoE model.
 *
 * The converted weights and the reference logits come from
 * Mila/Tools/Converters/Gemma/gemma_4_26b_moe/hf_gemma_moe_model_reference.py, which converts its own
 * checkpoint with Gemma/convert_weights.py -- so one capture holds the converter's names and the block's
 * wiring together. Tolerances are the ones Gemma4MoE.md Phase 8 fixed before the first run; the gates
 * that read the capture skip without it.
 *
 * CUDA device tests -- skipped when no CUDA device is present.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::Transformers::Gemma
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        namespace fs = std::filesystem;

        using NoKvCompression = Mila::Dnn::Quant::KvCache::NoKvCompression;
        using NoWeightQuant = Mila::Dnn::Quant::Weight::NoWeightQuant;
        using HostTokens = Tensor<TensorDataType::INT32, CpuMemoryResource>;
        using DeviceTokens = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>;

        template<TensorDataType TPrecision>
        using RoutedNetwork = GemmaTransformer<DeviceType::Cuda, TPrecision, NoWeightQuant, NoKvCompression, true, true>;

        template<TensorDataType TPrecision>
        using DenseNetwork = GemmaTransformer<DeviceType::Cuda, TPrecision>;

        constexpr int64_t kBatch = 1;
        constexpr int64_t kContext = 16;
        constexpr int64_t kModelDim = 128;
        constexpr int64_t kLayers = 2;
        constexpr int64_t kExperts = 8;
        constexpr int64_t kTopK = 2;
        constexpr int64_t kExpertHidden = 64;

        // Gemma4MoE.md Phase 8, "Wiring gate". The BF16 figure replaced the original 1e-2 after
        // HuggingFace's own bfloat16 run measured 1.4e-2 to 4.1e-2 from its float32 run on this model.
        constexpr double kFp32LogitTolerance = 1e-4;
        constexpr double kBf16RelativeL2 = 1e-1;

        fs::path captureDirectory()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_moe_tiny";
        }

        fs::path weightsPath( TensorDataType precision )
        {
            return captureDirectory()
                / ( precision == TensorDataType::BF16 ? "gemma4_moe_tiny_bf16.bin" : "gemma4_moe_tiny_fp32.bin" );
        }

        fs::path referencePath()
        {
            return captureDirectory() / "gemma4_moe_tiny_reference.safetensors";
        }

        bool captureExists()
        {
            return fs::exists( referencePath() )
                && fs::exists( weightsPath( TensorDataType::FP32 ) )
                && fs::exists( weightsPath( TensorDataType::BF16 ) );
        }

        // The capture's geometry, for the gates that need no capture.
        GemmaConfig routedConfig()
        {
            return GemmaConfig( kModelDim, kLayers )
                .withVocabularyLength( 128 )
                .withNumHeads( 4 )
                .withNumKVHeads( 2 )
                .withHeadDim( 64 )
                .withGlobalHeadDim( 128 )
                .withNumGlobalKVHeads( 1 )
                .withKeyEqualsValue( true )
                .withHiddenDimension( 128 )
                .withMaxSequenceLength( 32 )
                .withRMSNormEpsilon( 1e-6f )
                .withWindow( 8 )
                .withSlidingWindowPattern( 2 )
                .withGlobalRotaryDim( 32 )
                .withRoPETheta( 10000.0f )
                .withGlobalRoPETheta( 250000.0f )
                .withFinalLogitSoftcapping( 30.0f )
                .withMixtureOfExperts( kExperts, kTopK, kExpertHidden );
        }

        // The mapping GemmaModel applies to the same metadata; the network is not reachable through it.
        GemmaConfig configFromMetadata( const Serialization::WeightsMetadata& metadata )
        {
            GemmaConfig config( static_cast<dim_t>( metadata.embedding_dim ), static_cast<dim_t>( metadata.num_layers ) );

            config.withVocabularyLength( static_cast<dim_t>( metadata.vocab_size ) )
                .withMaxSequenceLength( static_cast<dim_t>( metadata.max_seq_length ) )
                .withNumHeads( static_cast<dim_t>( metadata.num_heads ) )
                .withNumKVHeads( static_cast<dim_t>( metadata.num_kv_heads ) )
                .withHeadDim( static_cast<dim_t>( metadata.head_dim ) )
                .withGlobalHeadDim( static_cast<dim_t>( metadata.global_head_dim ) )
                .withNumGlobalKVHeads( static_cast<dim_t>( metadata.num_global_kv_heads ) )
                .withKeyEqualsValue( metadata.key_equals_value )
                .withHiddenDimension( static_cast<dim_t>( metadata.hidden_dim ) )
                .withRMSNormEpsilon( metadata.norm_epsilon )
                .withWindow( static_cast<dim_t>( metadata.window ) )
                .withSlidingWindowPattern( static_cast<dim_t>( metadata.sliding_window_pattern ) )
                .withGlobalRotaryDim( static_cast<dim_t>( metadata.global_rotary_dim ) )
                .withRoPETheta( metadata.rope_theta_local )
                .withGlobalRoPETheta( metadata.rope_theta_global )
                .withFinalLogitSoftcapping( metadata.final_logit_softcapping )
                .withTieWordEmbeddings( metadata.tie_word_embeddings )
                .withMixtureOfExperts( static_cast<dim_t>( metadata.num_experts ),
                    static_cast<dim_t>( metadata.top_k_experts ), static_cast<dim_t>( metadata.expert_hidden_dim ) );

            return config;
        }

        DeviceTokens deviceTokens( const std::vector<std::int32_t>& ids )
        {
            HostTokens host( Device::Cpu(), shape_t{ kBatch, static_cast<int64_t>( ids.size() ) } );
            std::copy( ids.begin(), ids.end(), host.data() );

            DeviceTokens device( Device::Cuda( 0 ), shape_t{ kBatch, static_cast<int64_t>( ids.size() ) } );
            copy( host, device );

            return device;
        }

        void constructRouted( const GemmaConfig& config )
        {
            RoutedNetwork<TensorDataType::BF16> network( "gemma", config, Device::Cuda( 0 ) );
        }

        void constructDense( const GemmaConfig& config )
        {
            DenseNetwork<TensorDataType::BF16> network( "gemma", config, Device::Cuda( 0 ) );
        }
    }

    class GemmaMixtureOfExpertsCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            context_ = createExecutionContext( Device::Cuda( 0 ) );
        }

        template<TensorDataType TPrecision>
        void expectLogitsMatchReference()
        {
            const TensorDataType precision = TPrecision;

            Serialization::WeightsReader weights( weightsPath( precision ) );
            const GemmaConfig config = configFromMetadata( weights.getWeightsMetadata() );

            ASSERT_TRUE( config.hasMixtureOfExperts() ) << "the converted weights carry no expert geometry";

            RoutedNetwork<TPrecision> network( "gemma", config, Device::Cuda( 0 ) );
            network.build( BuildContext( shape_t{ kBatch, kContext }, RuntimeMode::Inference ).withPrefillSize( kContext ) );
            network.loadParameters( weights );

            Serialization::WeightsReader reference( referencePath() );
            auto token_blob = reference.readTensorBlob<CpuMemoryResource>( "tokens" );
            auto logits_blob = reference.readTensorBlob<CpuMemoryResource>( "logits" );

            const int64_t total = static_cast<int64_t>( token_blob.getMetadata().shape[ 0 ] );
            const int64_t steps = static_cast<int64_t>( logits_blob.getMetadata().shape[ 0 ] );
            const int64_t vocab = static_cast<int64_t>( logits_blob.getMetadata().shape[ 1 ] );
            const int64_t prompt = total - ( steps - 1 );

            ASSERT_LE( total, kContext );

            std::vector<std::int32_t> tokens( static_cast<std::size_t>( total ) );
            std::memcpy( tokens.data(), token_blob.data(), tokens.size() * sizeof( std::int32_t ) );

            std::vector<float> expected( static_cast<std::size_t>( steps * vocab ) );
            ASSERT_EQ( logits_blob.sizeBytes(), expected.size() * sizeof( float ) );
            std::memcpy( expected.data(), logits_blob.data(), logits_blob.sizeBytes() );

            std::vector<float> actual;

            auto append = [&]( const auto& logits )
            {
                network.synchronize();

                auto host = toHost<TensorDataType::FP32>( logits, context_.get() );
                actual.insert( actual.end(), host.data(), host.data() + host.size() );
            };

            append( network.prefill( deviceTokens( std::vector<std::int32_t>( tokens.begin(), tokens.begin() + prompt ) ) ) );

            for ( int64_t step = 1; step < steps; ++step )
            {
                const int64_t position = prompt + step - 1;

                append( network.decode( deviceTokens( { tokens[ static_cast<std::size_t>( position ) ] } ), position ) );
            }

            ASSERT_EQ( actual.size(), expected.size() );

            for ( int64_t step = 0; step < steps; ++step )
            {
                const float* got = actual.data() + step * vocab;
                const float* want = expected.data() + step * vocab;

                int64_t mismatches = 0;
                double worst = 0.0;
                double difference_squares = 0.0;
                double reference_squares = 0.0;

                for ( int64_t column = 0; column < vocab; ++column )
                {
                    const double difference = static_cast<double>( got[ column ] ) - want[ column ];

                    worst = std::max( worst, std::fabs( difference ) );
                    difference_squares += difference * difference;
                    reference_squares += static_cast<double>( want[ column ] ) * want[ column ];

                    if ( std::fabs( difference ) > kFp32LogitTolerance )
                    {
                        ++mismatches;
                    }
                }

                const double relative_l2 = std::sqrt( difference_squares / reference_squares );
                const auto got_argmax = std::max_element( got, got + vocab ) - got;
                const auto want_argmax = std::max_element( want, want + vocab ) - want;

                std::cout << std::format( "[ reference ] {} step {}: worst {:.3e}, relative L2 {:.3e}, argmax {} vs {}\n",
                    precision == TensorDataType::BF16 ? "BF16" : "FP32", step, worst, relative_l2, got_argmax, want_argmax );

                if ( precision == TensorDataType::BF16 )
                {
                    EXPECT_LE( relative_l2, kBf16RelativeL2 ) << "step " << step;
                    EXPECT_EQ( got_argmax, want_argmax ) << "step " << step;
                }
                else
                {
                    EXPECT_EQ( mismatches, 0 ) << "step " << step << " of " << vocab << " logits; worst " << worst;
                }
            }
        }

        std::unique_ptr<IExecutionContext> context_;
    };

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( GemmaMixtureOfExpertsCudaTests, Fp32_LogitsMatchHuggingFace )
    {
        if ( !captureExists() )
        {
            GTEST_SKIP() << "tiny MoE capture not present at: " << captureDirectory().string();
        }

        expectLogitsMatchReference<TensorDataType::FP32>();
    }

    TEST_F( GemmaMixtureOfExpertsCudaTests, Bf16_LogitsMatchHuggingFace )
    {
        if ( !captureExists() )
        {
            GTEST_SKIP() << "tiny MoE capture not present at: " << captureDirectory().string();
        }

        expectLogitsMatchReference<TensorDataType::BF16>();
    }

    // ====================================================================
    // Names -- the converter's vocabulary is the network's
    // ====================================================================

    TEST_F( GemmaMixtureOfExpertsCudaTests, FlatNames_AreTheConvertedFileNames )
    {
        if ( !captureExists() )
        {
            GTEST_SKIP() << "tiny MoE capture not present at: " << captureDirectory().string();
        }

        const fs::path saved = fs::temp_directory_path() / "mila_gemma_moe_names.safetensors";
        std::vector<std::string> expected;

        {
            Serialization::WeightsReader weights( weightsPath( TensorDataType::FP32 ) );
            expected = weights.getTensorNames();

            RoutedNetwork<TensorDataType::FP32> network( "gemma", configFromMetadata( weights.getWeightsMetadata() ), Device::Cuda( 0 ) );
            network.build( BuildContext( shape_t{ kBatch, kContext }, RuntimeMode::Inference ).withPrefillSize( kContext ) );
            network.loadParameters( weights );

            Serialization::SafeTensorsWriter writer( saved );
            network.saveFlatTensors( writer, "", Serialization::TensorSavePass::Declare );
            writer.beginData();
            network.saveFlatTensors( writer, "", Serialization::TensorSavePass::Write );
            writer.close();
        }

        std::vector<std::string> actual;
        {
            Serialization::WeightsReader reader( saved );
            actual = reader.getTensorNames();
        }

        std::error_code ignored;
        fs::remove( saved, ignored );

        std::sort( expected.begin(), expected.end() );
        std::sort( actual.begin(), actual.end() );

        EXPECT_EQ( actual, expected );

        const auto expert_tensors = std::count_if( actual.begin(), actual.end(),
            []( const std::string& name ) { return name.find( ".experts." ) != std::string::npos; } );

        EXPECT_EQ( expert_tensors, 2 * kLayers ) << "gate_up_proj and down_proj per layer";
    }

    // ====================================================================
    // G. Footprint
    // ====================================================================

    // One network alive at a time: the process-wide RoPE cache makes a second one under-report.
    TEST_F( GemmaMixtureOfExpertsCudaTests, Bf16_FootprintPredictedAndInactiveBytesReported )
    {
        const BuildContext context = BuildContext( shape_t{ kBatch, kContext }, RuntimeMode::Inference )
            .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) )
            .withPrefillSize( kContext );

        MemoryStats predicted;
        {
            RoutedNetwork<TensorDataType::BF16> predictor( "gemma", routedConfig(), Device::Cuda( 0 ) );
            predicted = predictor.getRequiredMemory( context );
        }

        MemoryStats built;
        {
            RoutedNetwork<TensorDataType::BF16> network( "gemma", routedConfig(), Device::Cuda( 0 ) );
            network.build( context );
            built = network.getMemoryStats();
        }

        EXPECT_EQ( predicted.device_parameter_bytes, built.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, built.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_gradient_bytes, built.device_gradient_bytes ) << "gradients";
        EXPECT_EQ( predicted.device_inactive_parameter_bytes, built.device_inactive_parameter_bytes ) << "inactive parameters";

        // Each layer's bank is 8 experts x 3 x 64 x 128 at BF16; top-2 leaves six of the eight unread.
        const std::size_t bank_bytes = static_cast<std::size_t>( kExperts * 3 * kExpertHidden * kModelDim ) * 2;

        EXPECT_EQ( built.device_inactive_parameter_bytes,
            static_cast<std::size_t>( kLayers ) * bank_bytes / kExperts * ( kExperts - kTopK ) );
    }

    TEST_F( GemmaMixtureOfExpertsCudaTests, DeploymentFootprint_ReadsExpertGeometryFromTheWeights )
    {
        if ( !captureExists() )
        {
            GTEST_SKIP() << "tiny MoE capture not present at: " << captureDirectory().string();
        }

        const DeploymentFootprint footprint = GemmaModel<DeviceType::Cuda, TensorDataType::FP32>::getDeploymentFootprint(
            weightsPath( TensorDataType::FP32 ), GemmaModelConfig( kContext ), Device::Cuda( 0 ) );

        EXPECT_GT( footprint.memory.device_inactive_parameter_bytes, 0u );
    }

    // The routed chassis builds FP4 at group 64: the dispatch, the stored scheme name and the model all agree.
    TEST_F( GemmaMixtureOfExpertsCudaTests, Fp4Load_UsesTheRoutedGroup )
    {
        if ( !captureExists() )
        {
            GTEST_SKIP() << "tiny MoE capture not present at: " << captureDirectory().string();
        }

        GemmaModelConfig config( kContext );
        config.withWeightQuantization( WeightQuantization::FP4 );

        const auto model = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::load(
            weightsPath( TensorDataType::BF16 ), config, Device::Cuda( 0 ) );

        EXPECT_EQ( model->weightQuantizationScheme(), "per_group_fp4_64" );
    }

    // ====================================================================
    // Refusals
    // ====================================================================

    TEST_F( GemmaMixtureOfExpertsCudaTests, DenseConfig_RefusedByRoutedTransformer )
    {
        GemmaConfig dense = routedConfig();
        dense.withMixtureOfExperts( 0, 0, 0 );

        EXPECT_THROW( constructRouted( dense ), std::invalid_argument );
    }

    TEST_F( GemmaMixtureOfExpertsCudaTests, RoutedConfig_RefusedByDenseTransformer )
    {
        EXPECT_THROW( constructDense( routedConfig() ), std::invalid_argument );
    }
}
