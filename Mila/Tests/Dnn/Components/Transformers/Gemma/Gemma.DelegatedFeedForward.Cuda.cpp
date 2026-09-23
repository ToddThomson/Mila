/**
 * @file Gemma.DelegatedFeedForward.Cuda.cpp
 * @brief GemmaTransformer with kDelegatedFeedForward against the inline FFN, on identical weights.
 *
 * The flag moves every block's FFN into a GatedMLP child. That must rename the FFN tensors and
 * change nothing else, so one FP32 source network is initialized and saved, both arms load that
 * file -- the delegated arm under the renamed paths -- and their prefill and decode logits must be
 * bit-identical. The config is sized so PerGroupFp4<128> groups every projection input.
 *
 * CUDA device tests -- skipped when no CUDA device is present.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
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
        using PerGroupFp4 = Mila::Dnn::Quant::Weight::PerGroupFp4<128>;
        using HostTokens = Tensor<TensorDataType::INT32, CpuMemoryResource>;
        using DeviceTokens = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>;

        constexpr int64_t kModelDim = 128;
        constexpr int64_t kLayers = 2;
        constexpr int64_t kHeads = 4;
        constexpr int64_t kKVHeads = 2;
        constexpr int64_t kHeadDim = 64;
        constexpr int64_t kGlobalHeadDim = 128;
        constexpr int64_t kHidden = 128;
        constexpr int64_t kVocab = 128;
        constexpr int64_t kMaxSequence = 32;
        constexpr int64_t kContext = 16;
        constexpr int64_t kPrompt = 12;
        constexpr int64_t kDecodeSteps = 3;
        constexpr int64_t kBatch = 1;

        // Pattern 2 over two layers: layer 0 is sliding and layer 1 global, so both block kinds
        // carry the flag. Every Linear input is 128 or 256, which PerGroupFp4<128> requires.
        GemmaConfig delegationConfig()
        {
            return GemmaConfig( kModelDim, kLayers )
                .withVocabularyLength( kVocab )
                .withNumHeads( kHeads )
                .withNumKVHeads( kKVHeads )
                .withHeadDim( kHeadDim )
                .withGlobalHeadDim( kGlobalHeadDim )
                .withNumGlobalKVHeads( 1 )
                .withKeyEqualsValue( true )
                .withHiddenDimension( kHidden )
                .withMaxSequenceLength( kMaxSequence )
                .withRMSNormEpsilon( 1e-6f )
                .withWindow( 8 )
                .withSlidingWindowPattern( 2 )
                .withGlobalRotaryDim( 64 )
                .withRoPETheta( 10000.0f )
                .withGlobalRoPETheta( 1000000.0f )
                .withFinalLogitSoftcapping( 30.0f );
        }

        BuildContext inferenceContext()
        {
            return BuildContext( shape_t{ kBatch, kContext }, RuntimeMode::Inference );
        }

        fs::path scratchPath( const std::string& stem )
        {
            static int counter = 0;

            return fs::temp_directory_path()
                / std::format( "mila_gemma_delegated_{}_{}.safetensors", stem, counter++ );
        }

        // The one renaming the flag is allowed to cause: the FFN projections move under `mlp`.
        std::string toDelegatedName( std::string name )
        {
            for ( const char* projection : { ".fc_gate_up.", ".fc_down." } )
            {
                if ( const auto at = name.find( projection ); at != std::string::npos )
                {
                    name.insert( at, ".mlp" );
                }
            }

            return name;
        }

        template<typename TNetwork>
        void saveNetwork( const TNetwork& network, const fs::path& path )
        {
            Serialization::SafeTensorsWriter writer( path );
            network.saveFlatTensors( writer, "", Serialization::TensorSavePass::Declare );
            writer.beginData();
            network.saveFlatTensors( writer, "", Serialization::TensorSavePass::Write );
            writer.close();
        }

        // Loads an FP32 file into a network of either precision. A BF16 network receives the top
        // 16 bits of each FP32 pattern, which is exactly bfloat16; layer_scalar is an FP32 [1] by
        // contract at every precision and passes through.
        template<typename TNetwork>
        void loadNetwork( TNetwork& network, const fs::path& path, TensorDataType precision, bool delegated )
        {
            Serialization::WeightsReader reader( path );

            for ( const auto& name : reader.getTensorNames() )
            {
                auto source = reader.readTensorBlob<CpuMemoryResource>( name );
                ASSERT_EQ( source.getMetadata().dtype, TensorDataType::FP32 ) << name;

                const std::string target_name = delegated ? toDelegatedName( name ) : name;
                const auto last_dot = target_name.rfind( '.' );
                auto target = network.findComponent( target_name.substr( 0, last_dot ) );
                const std::string parameter = target_name.substr( last_dot + 1 );

                if ( precision == TensorDataType::BF16 && parameter != "layer_scalar" )
                {
                    std::vector<float> values( source.sizeBytes() / sizeof( float ) );
                    std::memcpy( values.data(), source.data(), source.sizeBytes() );

                    std::vector<std::uint16_t> bits( values.size() );

                    for ( std::size_t i = 0; i < values.size(); ++i )
                    {
                        bits[ i ] = static_cast<std::uint16_t>( std::bit_cast<std::uint32_t>( values[ i ] ) >> 16 );
                    }

                    const std::size_t bytes = bits.size() * sizeof( std::uint16_t );
                    Serialization::TensorMetadata meta{ TensorDataType::BF16, source.getMetadata().shape, bytes };
                    Serialization::TensorBlobView blob( meta, bits.data(), bytes );

                    target->loadParameter( parameter, blob );
                }
                else
                {
                    target->loadParameter( parameter, source );
                }

                network.synchronize();
            }
        }

        DeviceTokens deviceTokens( const std::vector<std::int32_t>& ids )
        {
            HostTokens host( Device::Cpu(), shape_t{ kBatch, static_cast<int64_t>( ids.size() ) } );
            std::copy( ids.begin(), ids.end(), host.data() );

            DeviceTokens device( Device::Cuda( 0 ), shape_t{ kBatch, static_cast<int64_t>( ids.size() ) } );
            copy( host, device );

            return device;
        }

        // Prefill logits followed by each decode step's logits, concatenated.
        template<typename TNetwork>
        std::vector<float> prefillThenDecode( TNetwork& network, IExecutionContext* context )
        {
            std::vector<float> logits;

            auto append = [&]( const auto& tensor )
            {
                network.synchronize();

                auto host = toHost<TensorDataType::FP32>( tensor, context );
                logits.insert( logits.end(), host.data(), host.data() + host.size() );
            };

            std::vector<std::int32_t> prompt( static_cast<std::size_t>( kPrompt ) );

            for ( std::size_t i = 0; i < prompt.size(); ++i )
            {
                prompt[ i ] = static_cast<std::int32_t>( ( i * 37 + 5 ) % static_cast<std::size_t>( kVocab ) );
            }

            append( network.prefill( deviceTokens( prompt ) ) );

            for ( int64_t step = 0; step < kDecodeSteps; ++step )
            {
                const auto token = static_cast<std::int32_t>( ( step * 53 + 11 ) % kVocab );
                append( network.decode( deviceTokens( { token } ), kPrompt + step ) );
            }

            return logits;
        }
    }

    class GemmaDelegatedFeedForwardCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            context_ = createExecutionContext( Device::Cuda( 0 ) );
            source_path_ = scratchPath( "source" );

            // Scoped so the source is gone before any arm builds: the process-wide RoPE cache
            // makes a second live network report less than it allocates.
            GemmaTransformer<DeviceType::Cuda, TensorDataType::FP32> source( "gemma", delegationConfig(), Device::Cuda( 0 ) );
            source.build( BuildContext( shape_t{ kBatch, kContext }, RuntimeMode::Inference, true ) );
            saveNetwork( source, source_path_ );
        }

        void TearDown() override
        {
            std::error_code ignored;
            fs::remove( source_path_, ignored );
        }

        template<TensorDataType TPrecision, typename TWeightQuantization>
        void expectDelegatedLogitsBitIdentical()
        {
            using InlineNetwork = GemmaTransformer<DeviceType::Cuda, TPrecision, TWeightQuantization, NoKvCompression, false>;
            using DelegatedNetwork = GemmaTransformer<DeviceType::Cuda, TPrecision, TWeightQuantization, NoKvCompression, true>;

            InlineNetwork inline_network( "gemma", delegationConfig(), Device::Cuda( 0 ) );
            inline_network.build( inferenceContext() );
            loadNetwork( inline_network, source_path_, TPrecision, false );

            DelegatedNetwork delegated_network( "gemma", delegationConfig(), Device::Cuda( 0 ) );
            delegated_network.build( inferenceContext() );
            loadNetwork( delegated_network, source_path_, TPrecision, true );

            const std::vector<float> inline_logits = prefillThenDecode( inline_network, context_.get() );
            const std::vector<float> delegated_logits = prefillThenDecode( delegated_network, context_.get() );

            ASSERT_EQ( inline_logits.size(), static_cast<std::size_t>( ( 1 + kDecodeSteps ) * kVocab ) );
            ASSERT_EQ( delegated_logits.size(), inline_logits.size() );

            std::size_t mismatches = 0;
            std::size_t nonzero = 0;
            std::size_t non_finite = 0;

            for ( std::size_t i = 0; i < inline_logits.size(); ++i )
            {
                if ( std::bit_cast<std::uint32_t>( inline_logits[ i ] ) != std::bit_cast<std::uint32_t>( delegated_logits[ i ] ) )
                {
                    ++mismatches;
                }

                if ( inline_logits[ i ] != 0.0f )
                {
                    ++nonzero;
                }

                if ( !std::isfinite( inline_logits[ i ] ) )
                {
                    ++non_finite;
                }
            }

            EXPECT_EQ( non_finite, 0u ) << "the inline arm produced non-finite logits; the weights did not load";
            EXPECT_GT( nonzero, 0u ) << "the inline arm produced all zeros; the comparison proves nothing";
            EXPECT_EQ( mismatches, 0u ) << "of " << inline_logits.size() << " logits";
        }

        // Pooling must survive delegation: the delegated build allocates exactly what the inline
        // one does, and predicts it. One network alive at a time, for the RoPE cache reason above.
        template<TensorDataType TPrecision, typename TWeightQuantization>
        void expectDelegatedFootprintUnchanged()
        {
            using InlineNetwork = GemmaTransformer<DeviceType::Cuda, TPrecision, TWeightQuantization, NoKvCompression, false>;
            using DelegatedNetwork = GemmaTransformer<DeviceType::Cuda, TPrecision, TWeightQuantization, NoKvCompression, true>;

            MemoryStats inline_built;
            {
                InlineNetwork network( "gemma", delegationConfig(), Device::Cuda( 0 ) );
                network.build( inferenceContext() );
                inline_built = network.getMemoryStats();
            }

            MemoryStats delegated_predicted;
            {
                DelegatedNetwork predictor( "gemma", delegationConfig(), Device::Cuda( 0 ) );
                delegated_predicted = predictor.getRequiredMemory( inferenceContext()
                    .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) )
                    .withAvailableDeviceBytes( readFreeDeviceBytes( Device::Cuda( 0 ) ) ) );
            }

            MemoryStats delegated_built;
            {
                DelegatedNetwork network( "gemma", delegationConfig(), Device::Cuda( 0 ) );
                network.build( inferenceContext() );
                delegated_built = network.getMemoryStats();
            }

            EXPECT_EQ( delegated_predicted.device_parameter_bytes, delegated_built.device_parameter_bytes ) << "predicted parameters";
            EXPECT_EQ( delegated_predicted.device_state_bytes, delegated_built.device_state_bytes ) << "predicted state";
            EXPECT_EQ( delegated_predicted.device_gradient_bytes, delegated_built.device_gradient_bytes ) << "predicted gradients";

            EXPECT_EQ( delegated_built.device_parameter_bytes, inline_built.device_parameter_bytes ) << "parameters against inline";
            EXPECT_EQ( delegated_built.device_state_bytes, inline_built.device_state_bytes ) << "state against inline";
            EXPECT_EQ( delegated_built.device_gradient_bytes, inline_built.device_gradient_bytes ) << "gradients against inline";
        }

        std::unique_ptr<IExecutionContext> context_;
        fs::path source_path_;
    };

    TEST_F( GemmaDelegatedFeedForwardCudaTests, Fp32_LogitsBitIdenticalToInline )
    {
        expectDelegatedLogitsBitIdentical<TensorDataType::FP32, NoWeightQuant>();
    }

    TEST_F( GemmaDelegatedFeedForwardCudaTests, Bf16_LogitsBitIdenticalToInline )
    {
        expectDelegatedLogitsBitIdentical<TensorDataType::BF16, NoWeightQuant>();
    }

    TEST_F( GemmaDelegatedFeedForwardCudaTests, PerGroupFp4_LogitsBitIdenticalToInline )
    {
        expectDelegatedLogitsBitIdentical<TensorDataType::BF16, PerGroupFp4>();
    }

    TEST_F( GemmaDelegatedFeedForwardCudaTests, Bf16_FootprintUnchangedAndPredicted )
    {
        expectDelegatedFootprintUnchanged<TensorDataType::BF16, NoWeightQuant>();
    }

    TEST_F( GemmaDelegatedFeedForwardCudaTests, PerGroupFp4_FootprintUnchangedAndPredicted )
    {
        expectDelegatedFootprintUnchanged<TensorDataType::BF16, PerGroupFp4>();
    }

    // The renamed vocabulary is exactly the inline one with the FFN projections moved under
    // `mlp` -- the list a converter and a republish will have to produce, and nothing more.
    TEST_F( GemmaDelegatedFeedForwardCudaTests, FlatNames_AreInlineNamesWithFeedForwardUnderMlp )
    {
        using DelegatedNetwork = GemmaTransformer<DeviceType::Cuda, TensorDataType::FP32, NoWeightQuant, NoKvCompression, true>;

        const fs::path delegated_path = scratchPath( "delegated" );

        {
            DelegatedNetwork network( "gemma", delegationConfig(), Device::Cuda( 0 ) );
            network.build( inferenceContext() );
            loadNetwork( network, source_path_, TensorDataType::FP32, true );
            saveNetwork( network, delegated_path );
        }

        std::vector<std::string> expected;
        {
            Serialization::WeightsReader reader( source_path_ );

            for ( const auto& name : reader.getTensorNames() )
            {
                expected.push_back( toDelegatedName( name ) );
            }
        }

        std::vector<std::string> actual;
        {
            Serialization::WeightsReader reader( delegated_path );
            actual = reader.getTensorNames();
        }

        std::error_code ignored;
        fs::remove( delegated_path, ignored );

        std::sort( expected.begin(), expected.end() );
        std::sort( actual.begin(), actual.end() );

        EXPECT_EQ( actual, expected );

        const auto under_mlp = std::count_if( actual.begin(), actual.end(),
            []( const std::string& name ) { return name.find( ".mlp." ) != std::string::npos; } );

        EXPECT_EQ( under_mlp, 2 * kLayers ) << "fc_gate_up and fc_down per layer, bias-free";
    }
}
