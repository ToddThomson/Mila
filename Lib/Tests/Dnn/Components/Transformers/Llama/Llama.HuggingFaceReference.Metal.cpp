/**
 * @file Llama.HuggingFaceReference.Metal.cpp
 * @brief LlamaTransformer on Metal against the tiny HuggingFace Llama, in FP16 and BF16 on the BF16 weights.
 *
 * The capture Llama.HuggingFaceReference.Cuda.cpp reads, through the whole network: prefill, then decode of the
 * reference's own tokens. The first check that Llama's operations are wired as a model on Metal (MetalBackend.md Phase 4).
 */

#include <gtest/gtest.h>

import Mila;
import std;

#include "Measurement/LogLikelihoodHarness.h"

namespace Mila::Tests::Dnn::Components::Transformers::Llama
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        namespace fs = std::filesystem;

        template<TensorDataType TPrecision>
        using MetalNetwork = LlamaTransformer<DeviceType::Metal, TPrecision>;

        constexpr std::int64_t kBatch = 1;
        constexpr std::int64_t kContext = 32;

        // The CUDA test's BF16 bar, which is what HuggingFace's own bfloat16 run measured against its float32 one.
        // FP16 is held to the same bar: it has more significand bits, and what it risks is range, which misses by far more.
        constexpr double kRelativeL2 = 1e-1;

        enum class Capture
        {
            Unscaled,
            Scaled
        };

        fs::path captureDirectory( Capture capture )
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "llama"
                / ( capture == Capture::Scaled ? "llama_tiny_scaled" : "llama_tiny" );
        }

        // Every Metal precision reads the BF16 file, as it reads a published package (MetalBackend.md section 5).
        fs::path weightsPath( Capture capture )
        {
            return captureDirectory( capture ) / "llama_tiny_bf16.bin";
        }

        fs::path referencePath( Capture capture )
        {
            return captureDirectory( capture ) / "llama_tiny_reference.safetensors";
        }

        LlamaConfig capturedConfig( Capture capture )
        {
            Serialization::WeightsReader weights( weightsPath( capture ) );

            return LlamaModel<DeviceType::Metal, TensorDataType::FP16>::configFromMetadata( weights.getWeightsMetadata() );
        }

        template<TensorDataType TPrecision>
        std::unique_ptr<MetalNetwork<TPrecision>> loadedNetwork( Capture capture, const LlamaConfig& config )
        {
            Serialization::WeightsReader weights( weightsPath( capture ) );

            auto network = std::make_unique<MetalNetwork<TPrecision>>( "llama", config, Device::Metal( 0 ) );
            network->build( BuildContext( shape_t{ kBatch, kContext }, RuntimeMode::Inference ).withPrefillSize( kContext ) );
            network->loadParameters( weights );

            return network;
        }

        template<typename TValue>
        std::vector<TValue> referenceValues( Capture capture, const std::string& name )
        {
            Serialization::WeightsReader reference( referencePath( capture ) );
            auto blob = reference.readTensorBlob<CpuMemoryResource>( name );

            std::vector<TValue> values( blob.sizeBytes() / sizeof( TValue ) );
            std::memcpy( values.data(), blob.data(), blob.sizeBytes() );

            return values;
        }

        template<typename TPrecisionConstant>
        class LlamaHuggingFaceReferenceMetal : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                if ( getDeviceCount( DeviceType::Metal ) == 0 )
                {
                    GTEST_SKIP() << "No Metal device available";
                }

                for ( const Capture capture : { Capture::Unscaled, Capture::Scaled } )
                {
                    if ( !fs::exists( referencePath( capture ) ) || !fs::exists( weightsPath( capture ) ) )
                    {
                        GTEST_SKIP() << "tiny Llama capture not present at: " << captureDirectory( capture ).string();
                    }
                }
            }
        };

        using Precisions = ::testing::Types<
            std::integral_constant<TensorDataType, TensorDataType::FP16>,
            std::integral_constant<TensorDataType, TensorDataType::BF16>>;

        struct PrecisionNames
        {
            template<typename T>
            static std::string GetName( int )
            {
                return T::value == TensorDataType::FP16 ? "Fp16" : "Bf16";
            }
        };

        TYPED_TEST_SUITE( LlamaHuggingFaceReferenceMetal, Precisions, PrecisionNames );

        /// Prefill the prompt, then decode the reference's own tokens one at a time, against HuggingFace's logits.
        /// `position_offset` decodes each token at a position it is not at, which the gate must catch.
        template<TensorDataType TPrecision>
        void expectLogitsMatchReference( Capture capture, std::int64_t position_offset, bool expect_match )
        {
            auto network = loadedNetwork<TPrecision>( capture, capturedConfig( capture ) );

            const std::vector<std::int32_t> tokens = referenceValues<std::int32_t>( capture, "tokens" );
            const std::vector<float> expected = referenceValues<float>( capture, "logits" );

            const std::int64_t vocab = capturedConfig( capture ).getVocabSize();
            const std::int64_t steps = static_cast<std::int64_t>( expected.size() ) / vocab;
            const std::int64_t total = static_cast<std::int64_t>( tokens.size() );
            const std::int64_t prompt = total - ( steps - 1 );

            ASSERT_LE( total, kContext );

            std::vector<float> actual;

            auto append = [&]( const auto& logits )
            {
                const std::vector<float> host = Measurement::hostLogits( *network, logits );
                actual.insert( actual.end(), host.begin(), host.end() );
            };

            append( network->prefill( Measurement::deviceTokens( *network,
                std::vector<std::int32_t>( tokens.begin(), tokens.begin() + prompt ) ) ) );

            for ( std::int64_t step = 1; step < steps; ++step )
            {
                const std::int64_t position = prompt + step - 1;

                append( network->decode( Measurement::deviceTokens( *network, { tokens[ static_cast<std::size_t>( position ) ] } ),
                    position + position_offset ) );
            }

            ASSERT_EQ( actual.size(), expected.size() );

            bool every_step_held = true;

            for ( std::int64_t step = 0; step < steps; ++step )
            {
                const float* got = actual.data() + step * vocab;
                const float* want = expected.data() + step * vocab;

                double difference_squares = 0.0;
                double reference_squares = 0.0;
                double worst = 0.0;
                bool finite = true;

                for ( std::int64_t column = 0; column < vocab; ++column )
                {
                    const double difference = static_cast<double>( got[ column ] ) - want[ column ];

                    finite = finite && std::isfinite( got[ column ] );
                    worst = std::max( worst, std::fabs( difference ) );
                    difference_squares += difference * difference;
                    reference_squares += static_cast<double>( want[ column ] ) * want[ column ];
                }

                const double relative_l2 = std::sqrt( difference_squares / reference_squares );
                const auto got_argmax = std::max_element( got, got + vocab ) - got;
                const auto want_argmax = std::max_element( want, want + vocab ) - want;

                // Logits off by up to `worst` anywhere cannot order two tokens nearer than that in HuggingFace's own.
                const double reference_gap = static_cast<double>( want[ want_argmax ] ) - want[ got_argmax ];
                const bool argmax_held = got_argmax == want_argmax || reference_gap < worst;
                const bool step_held = finite && relative_l2 <= kRelativeL2 && argmax_held;

                std::cout << std::format( "[ reference ] {} {} step {}: relative L2 {:.3e}, largest error {:.3e}, argmax {} vs {}\n",
                    capture == Capture::Scaled ? "scaled" : "unscaled", TPrecision == TensorDataType::FP16 ? "FP16" : "BF16",
                    step, relative_l2, worst, got_argmax, want_argmax );

                if ( expect_match )
                {
                    EXPECT_TRUE( finite ) << "step " << step << ": a logit is not finite";
                    EXPECT_LE( relative_l2, kRelativeL2 ) << "step " << step;
                    EXPECT_TRUE( argmax_held ) << "step " << step << ": argmax " << got_argmax << " where HuggingFace chose "
                        << want_argmax << ", " << reference_gap << " apart there against a largest error of " << worst;
                }

                every_step_held = every_step_held && step_held;
            }

            if ( !expect_match )
            {
                EXPECT_FALSE( every_step_held ) << "decoding at the wrong position still matched HuggingFace, so the check cannot see positions";
            }
        }
    }

    TYPED_TEST( LlamaHuggingFaceReferenceMetal, LogitsMatchHuggingFace )
    {
        expectLogitsMatchReference<TypeParam::value>( Capture::Unscaled, 0, true );
    }

    TYPED_TEST( LlamaHuggingFaceReferenceMetal, Scaled_LogitsMatchHuggingFace )
    {
        expectLogitsMatchReference<TypeParam::value>( Capture::Scaled, 0, true );
    }

    // Metal runs inference only: a backward pass is refused by name, before any operation is reached.
    TYPED_TEST( LlamaHuggingFaceReferenceMetal, Backward_IsRefused )
    {
        using Network = MetalNetwork<TypeParam::value>;

        Network network( "llama", capturedConfig( Capture::Unscaled ), Device::Metal( 0 ) );

        typename Network::TokenIndexType input( Device::Metal( 0 ), shape_t{ 1, 1 } );
        typename Network::TensorType output_grad( Device::Metal( 0 ), shape_t{ 1, 1, capturedConfig( Capture::Unscaled ).getVocabSize() } );

        try
        {
            (void)network.backward( input, output_grad );
            FAIL() << "backward ran on Metal";
        }
        catch ( const std::logic_error& error )
        {
            EXPECT_NE( std::string( error.what() ).find( "Metal runs inference only" ), std::string::npos ) << error.what();
        }
    }

    // The gate can fail: every decoded token placed one position early must miss HuggingFace somewhere. Early, since the
    // capture fills the context and a late position is refused before anything is computed.
    TYPED_TEST( LlamaHuggingFaceReferenceMetal, DecodedAtTheWrongPosition_MissesHuggingFace )
    {
        expectLogitsMatchReference<TypeParam::value>( Capture::Unscaled, -1, false );
    }
}
