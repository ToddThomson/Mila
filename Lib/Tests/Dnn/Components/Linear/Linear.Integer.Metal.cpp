/**
 * @file Linear.Integer.Metal.cpp
 * @brief Linear on Metal on Q4_0 and six-bit weights, Gemma's body and tied head, in FP16 and BF16, against a
 *        double-precision reference on the codes and scales as the library's own codecs write and read them.
 *
 * The gate is Notebooks/MetalGemma.md section 1.2.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Dnn.Components.Linear;
import Dnn.Quantization.Weight.Int4Packing;
import Dnn.Quantization.Weight.Int6Packing;
import Serialization.Tensor;

#include "../MetalReference.h"

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Tests::Dnn::MetalReference;

    namespace
    {
        constexpr dim_t kGroup = 32;

        template<TensorDataType TPrecision, typename TPolicy>
        struct Case
        {
            static constexpr TensorDataType kPrecision = TPrecision;
            using Policy = TPolicy;

            static constexpr bool kIsInt6 = HasInt6Codes<TPolicy>;

            static dim_t rowBytes( dim_t inputs )
            {
                return kIsInt6 ? packedRowBytesForSixBitCodes( inputs ) : packedRowBytesForFourBitCodes( inputs );
            }

            static void quantize( const float* weights, dim_t outputs, dim_t inputs, std::uint8_t* codes, std::uint16_t* scales )
            {
                if constexpr ( kIsInt6 )
                {
                    quantizeInt6( weights, outputs, inputs, kGroup, codes, scales );
                } else {
                    quantizeInt4( weights, outputs, inputs, kGroup, codes, scales );
                }
            }

            static void decode( const std::uint8_t* codes, const std::uint16_t* scales, dim_t outputs, dim_t inputs, float* values )
            {
                if constexpr ( kIsInt6 )
                {
                    dequantizeInt6( codes, scales, outputs, inputs, kGroup, values );
                } else {
                    dequantizeInt4( codes, scales, outputs, inputs, kGroup, values );
                }
            }
        };

        using Cases = ::testing::Types<
            Case<TensorDataType::FP16, PerGroupInt4<32>>,
            Case<TensorDataType::BF16, PerGroupInt4<32>>,
            Case<TensorDataType::FP16, PerGroupInt6<32>>,
            Case<TensorDataType::BF16, PerGroupInt6<32>>>;

        struct CaseNames
        {
            template<typename T>
            static std::string GetName( int )
            {
                return std::string( T::kPrecision == TensorDataType::FP16 ? "Fp16" : "Bf16" ) + ( T::kIsInt6 ? "Int6" : "Q4_0" );
            }
        };

        TensorBlobView bytesBlob( std::span<const std::uint8_t> bytes, const shape_t& shape )
        {
            TensorMetadata metadata;
            metadata.dtype = TensorDataType::UINT8;
            metadata.shape = shape;
            metadata.total_bytes = bytes.size_bytes();

            return TensorBlobView( std::move( metadata ), bytes.data(), bytes.size_bytes() );
        }

        /// Normal weights, a row of zeros, and a row with one large weight, so that the rest of its group rounds to the smallest steps.
        std::vector<float> testWeights( dim_t outputs, dim_t inputs, std::uint32_t seed )
        {
            std::vector<float> values = randomValues( static_cast<std::size_t>( outputs * inputs ), seed, 0.02f );

            if ( outputs > 2 )
            {
                std::fill_n( values.begin() + inputs, inputs, 0.0f );
                values[static_cast<std::size_t>( 2 * inputs + 3 )] = 7.5f;
            }

            return values;
        }

        template<typename TCase>
        struct Fixture
        {
            static constexpr TensorDataType kPrecision = TCase::kPrecision;

            using LinearType = Mila::Dnn::Linear<DeviceType::Metal, kPrecision, typename TCase::Policy>;
            using TensorType = Tensor<kPrecision, MetalMemoryResource>;

            dim_t inputs;
            dim_t outputs;
            std::vector<std::uint8_t> codes;
            std::vector<std::uint16_t> scales;
            std::vector<float> weight_values;
            std::unique_ptr<LinearType> linear;

            Fixture( dim_t input_features, dim_t output_features, dim_t built_rows )
                : inputs( input_features ), outputs( output_features ),
                  codes( static_cast<std::size_t>( output_features * TCase::rowBytes( input_features ) ) ),
                  scales( static_cast<std::size_t>( output_features * input_features / kGroup ) ),
                  weight_values( static_cast<std::size_t>( output_features * input_features ) )
            {
                const std::vector<float> weights = testWeights( outputs, inputs, 37 );

                TCase::quantize( weights.data(), outputs, inputs, codes.data(), scales.data() );
                TCase::decode( codes.data(), scales.data(), outputs, inputs, weight_values.data() );

                LinearConfig config( input_features, output_features );
                config.withBias( false );

                linear = std::make_unique<LinearType>( "linear", config, Device::Metal( 0 ) );
                linear->build( BuildContext( shape_t{ built_rows, input_features }, RuntimeMode::Inference, false ) );

                linear->loadParameter( "weight", bytesBlob( codes, shape_t{ outputs, TCase::rowBytes( inputs ) } ) );
                linear->loadParameter( "weight_scale", blobOver( scales, shape_t{ outputs, inputs / kGroup }, TensorDataType::FP16 ) );
            }

            /// Run `rows` rows, through decode() or forward(), and return the error against the reference at every `stride`th output.
            double errorAt( dim_t rows, dim_t stride = 1, bool decode = false )
            {
                const std::vector<float> input_values = randomValues( static_cast<std::size_t>( rows * inputs ), 13, 1.0f );
                TensorType input = metalTensor<kPrecision>( input_values, shape_t{ rows, inputs } );

                const TensorType& output = decode ? linear->decode( input ) : linear->forward( input );
                linear->synchronize();

                const std::vector<float> all = readFloats<kPrecision>( output );
                const std::vector<double> stored_input = asStored<kPrecision>( std::span<const float>( input_values ) );

                std::vector<float> actual;
                std::vector<double> reference;

                for ( dim_t r = 0; r < rows; ++r )
                {
                    for ( dim_t c = 0; c < outputs; c += stride )
                    {
                        double sum = 0.0;

                        for ( dim_t k = 0; k < inputs; ++k )
                        {
                            sum += stored_input[static_cast<std::size_t>( r * inputs + k )] * weight_values[static_cast<std::size_t>( c * inputs + k )];
                        }

                        actual.push_back( all[static_cast<std::size_t>( r * outputs + c )] );
                        reference.push_back( sum );
                    }
                }

                return relativeError( actual, reference );
            }
        };

        template<typename TCase>
        class LinearIntegerMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( LinearIntegerMetal, Cases, CaseNames );

        // Own lanes for one row and for several, expanded weights from 16 rows, tiles of 128 and of 8 with rows left over.
        constexpr dim_t kEveryRowCount[] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 63, 64, 65, 127, 128, 129, 130, 135, 136, 143, 256, 1023, 1024, 1031 };
    }

    TYPED_TEST( LinearIntegerMetal, MatchesTheReferenceAtEveryRowCount )
    {
        Fixture<TypeParam> fixture( 256, 48, 1031 );

        for ( const dim_t rows : kEveryRowCount )
        {
            EXPECT_LE( fixture.errorAt( rows ), kBar<TypeParam::kPrecision> ) << rows << " rows";
        }
    }

    TYPED_TEST( LinearIntegerMetal, MatchesTheReferenceWhenOutputsAreOffATileOrABlock )
    {
        for ( const dim_t outputs : { dim_t{ 1 }, dim_t{ 3 }, dim_t{ 17 }, dim_t{ 35 }, dim_t{ 1030 } } )
        {
            Fixture<TypeParam> fixture( 256, outputs, 1024 );

            for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 2 }, dim_t{ 8 }, dim_t{ 19 }, dim_t{ 64 }, dim_t{ 130 }, dim_t{ 1024 } } )
            {
                EXPECT_LE( fixture.errorAt( rows ), kBar<TypeParam::kPrecision> ) << outputs << " outputs, " << rows << " rows";
            }
        }
    }

    TYPED_TEST( LinearIntegerMetal, AnEarlierCallLeavesNoTraceInALaterOne )
    {
        Fixture<TypeParam> fixture( 256, 48, 1024 );

        for ( const dim_t rows : { dim_t{ 1024 }, dim_t{ 1 }, dim_t{ 136 }, dim_t{ 7 }, dim_t{ 64 }, dim_t{ 1 }, dim_t{ 1024 } } )
        {
            EXPECT_LE( fixture.errorAt( rows ), kBar<TypeParam::kPrecision> ) << rows << " rows";
        }
    }

    // One row of Q4_0 or of six bits reads 512 inputs a pass and eight outputs a threadgroup (MetalBackend.md section 4.2):
    // one pass, three, and four and a half, each with outputs that fill no threadgroup, fill one, and leave one partly empty.
    TYPED_TEST( LinearIntegerMetal, OneRowMatchesTheReferenceOverWholeAndPartialPasses )
    {
        for ( const dim_t inputs : { dim_t{ 512 }, dim_t{ 1536 }, dim_t{ 2304 } } )
        {
            for ( const dim_t outputs : { dim_t{ 1 }, dim_t{ 7 }, dim_t{ 8 }, dim_t{ 9 }, dim_t{ 48 } } )
            {
                Fixture<TypeParam> fixture( inputs, outputs, 8 );

                EXPECT_LE( fixture.errorAt( 1 ), kBar<TypeParam::kPrecision> ) << inputs << " by " << outputs;
                EXPECT_LE( fixture.errorAt( 1, 1, true ), kBar<TypeParam::kPrecision> ) << inputs << " by " << outputs << ", decode";
            }
        }
    }

    TYPED_TEST( LinearIntegerMetal, MatchesTheReferenceOnGemmasShapes )
    {
        // Gemma 4 12B's global layers: Q and the one K, 8704 by 3840, and the output projection of 4096 inputs.
        for ( const auto [inputs, outputs] : { std::pair{ dim_t{ 3840 }, dim_t{ 8704 } }, std::pair{ dim_t{ 4096 }, dim_t{ 3840 } } } )
        {
            Fixture<TypeParam> fixture( inputs, outputs, 130 );

            for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 8 }, dim_t{ 16 }, dim_t{ 130 } } )
            {
                EXPECT_LE( fixture.errorAt( rows, 7 ), kBar<TypeParam::kPrecision> ) << inputs << " by " << outputs << ", " << rows << " rows";
            }
        }
    }

    TYPED_TEST( LinearIntegerMetal, MatchesTheReferenceAtTheTiedHeadsWidth )
    {
        // The head's 3840 inputs and 32768 of its 262144 outputs, 32 blocks to expand, at a decode's row and 16 rows, every 97th
        // output checked. The whole head's weights would take 8 GB of host memory to check.
        Fixture<TypeParam> fixture( 3840, 32768, 16 );

        for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 16 } } )
        {
            EXPECT_LE( fixture.errorAt( rows, 97 ), kBar<TypeParam::kPrecision> ) << rows << " rows";
        }
    }

    // A speculative round's verify (MetalBackend.md section 4.6): decode() of up to 8 rows, built for a prefill chunk, at Gemma's shapes
    // and the tied head's width too, since a verify runs the head at every row.
    TYPED_TEST( LinearIntegerMetal, DecodeMatchesTheReferenceAtEveryRowItTakes )
    {
        Fixture<TypeParam> fixture( 3840, 8704, 1024 );

        for ( dim_t rows = 1; rows <= 8; ++rows )
        {
            EXPECT_LE( fixture.errorAt( rows, 7, true ), kBar<TypeParam::kPrecision> ) << rows << " rows";
        }

        Fixture<TypeParam> head( 3840, 32768, 16 );

        for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 5 }, dim_t{ 8 } } )
        {
            EXPECT_LE( head.errorAt( rows, 97, true ), kBar<TypeParam::kPrecision> ) << "head, " << rows << " rows";
        }
    }

    TYPED_TEST( LinearIntegerMetal, DecodeRefusesMoreThanEightRows )
    {
        using TensorType = Tensor<TypeParam::kPrecision, MetalMemoryResource>;

        Fixture<TypeParam> fixture( 256, 48, 16 );
        TensorType input( Device::Metal( 0 ), shape_t{ 9, 256 } );

        EXPECT_THROW( fixture.linear->decode( input ), std::invalid_argument );
    }

    TYPED_TEST( LinearIntegerMetal, RefusesFullPrecisionWeightsAndNamesThem )
    {
        Fixture<TypeParam> fixture( 256, 8, 8 );
        const std::vector<std::uint16_t> bits = randomBf16( 8 * 256, 51, 0.02f );

        try
        {
            fixture.linear->loadParameter( "weight", blobOver( bits, shape_t{ 8, 256 } ) );
            FAIL() << "BF16 weights were taken";
        }
        catch ( const std::invalid_argument& error )
        {
            EXPECT_NE( std::string( error.what() ).find( "linear.weight" ), std::string::npos ) << error.what();
        }
    }

    TYPED_TEST( LinearIntegerMetal, RefusesInputsItCannotReadInChunks )
    {
        using LinearType = typename Fixture<TypeParam>::LinearType;

        // Not a whole group of 32.
        LinearConfig config( 240, 16 );
        config.withBias( false );
        LinearType linear( "linear", config, Device::Metal( 0 ) );

        EXPECT_THROW( linear.build( BuildContext( shape_t{ 8, 240 }, RuntimeMode::Inference, false ) ), std::exception );
    }

    TEST( LinearIntegerMetalRows, FormatsExistAtFp16AndBf16Only )
    {
        EXPECT_TRUE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::FP16, PerGroupInt4<32>> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::BF16, PerGroupInt4<32>> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::FP16, PerGroupInt6<32>> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::BF16, PerGroupInt6<32>> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::FP32, PerGroupInt4<32>> ) );
    }
}
