/**
 * @file Linear.Metal.cpp
 * @brief Linear on Metal, in FP16 and BF16, against a double-precision reference on the weights as stored.
 *
 * Weights arrive as a BF16 blob through loadParameter, as a model hands them: no published model stores FP16. The bars and the
 * shapes are the gate's, in Notebooks/MetalLlamaOperations.md.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Dnn.Components.Linear;
import Serialization.Tensor;

#include "../MetalReference.h"

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Tests::Dnn::MetalReference;

    namespace
    {
        template<TensorDataType TPrecision>
        struct Fixture
        {
            using LinearType = Mila::Dnn::Linear<DeviceType::Metal, TPrecision>;
            using TensorType = Tensor<TPrecision, MetalMemoryResource>;

            dim_t inputs;
            dim_t outputs;
            std::vector<std::uint16_t> weight_bits;
            std::vector<std::uint16_t> bias_bits;
            std::unique_ptr<LinearType> linear;

            Fixture( dim_t input_features, dim_t output_features, dim_t built_rows, bool with_bias = false )
                : inputs( input_features ), outputs( output_features ),
                  weight_bits( randomBf16( static_cast<std::size_t>( input_features * output_features ), 11, 0.02f ) )
            {
                LinearConfig config( input_features, output_features );
                config.withBias( with_bias );

                linear = std::make_unique<LinearType>( "linear", config, Device::Metal( 0 ) );
                linear->build( BuildContext( shape_t{ built_rows, input_features }, RuntimeMode::Inference, false ) );

                linear->loadParameter( "weight", blobOver( weight_bits, shape_t{ output_features, input_features } ) );

                if ( with_bias )
                {
                    bias_bits = randomBf16( static_cast<std::size_t>( output_features ), 12, 0.5f );
                    linear->loadParameter( "bias", blobOver( bias_bits, shape_t{ output_features } ) );
                }
            }

            /// Run `rows` rows, and return the error against the reference on what the tensors hold.
            double errorAt( dim_t rows )
            {
                const std::vector<float> input_values = randomValues( static_cast<std::size_t>( rows * inputs ), 13, 1.0f );
                TensorType input = metalTensor<TPrecision>( input_values, shape_t{ rows, inputs } );

                const TensorType& output = linear->forward( input );
                linear->synchronize();

                const std::vector<float> actual = readFloats<TPrecision>( output );
                const std::vector<double> stored_input = asStored<TPrecision>( std::span<const float>( input_values ) );

                std::vector<double> reference( static_cast<std::size_t>( rows * outputs ) );

                for ( dim_t r = 0; r < rows; ++r )
                {
                    for ( dim_t c = 0; c < outputs; ++c )
                    {
                        double sum = bias_bits.empty() ? 0.0
                            : asStored<TPrecision>( fromBf16Bits( bias_bits[static_cast<std::size_t>( c )] ) );

                        for ( dim_t k = 0; k < inputs; ++k )
                        {
                            sum += stored_input[static_cast<std::size_t>( r * inputs + k )]
                                * asStored<TPrecision>( fromBf16Bits( weight_bits[static_cast<std::size_t>( c * inputs + k )] ) );
                        }

                        reference[static_cast<std::size_t>( r * outputs + c )] = sum;
                    }
                }

                return relativeError( actual, reference );
            }
        };

        template<typename TPrecisionConstant>
        class LinearMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( LinearMetal, Precisions, PrecisionNames );

        // 136 and 143 are where a tile of 8 follows tiles of 128, at an offset into every matrix.
        constexpr dim_t kEveryRowCount[] = { 1, 2, 3, 7, 8, 9, 15, 16, 127, 128, 129, 136, 143, 256 };
    }

    TYPED_TEST( LinearMetal, MatchesTheReferenceAtEveryRowCount )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Fixture<kPrecision> fixture( 64, 48, 256 );

        for ( const dim_t rows : kEveryRowCount )
        {
            EXPECT_LE( fixture.errorAt( rows ), kBar<kPrecision> ) << rows << " rows";
        }
    }

    TYPED_TEST( LinearMetal, FewerRowsThanWereBuiltGiveThoseRowsTheSameValues )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Fixture<kPrecision> fixture( 64, 48, 128 );

        // 128 rows go through a tile of 128, and their first 9 alone through a tile of 8 and the own kernel. The values match bit for bit
        // only if each path reads the same rows and rounds the same way, which is itself what this pins.
        const std::vector<float> values = randomValues( 128 * 64, 14, 1.0f );
        auto input = metalTensor<kPrecision>( values, shape_t{ 128, 64 } );

        const std::vector<float> all = readFloats<kPrecision>( [&]() -> const auto& {
            const auto& output = fixture.linear->forward( input );
            fixture.linear->synchronize();

            return output;
        }() );

        const std::vector<float> few_values( values.begin(), values.begin() + 9 * 64 );
        auto few = metalTensor<kPrecision>( few_values, shape_t{ 9, 64 } );

        const auto& viewed = fixture.linear->forward( few );
        fixture.linear->synchronize();

        EXPECT_EQ( readFloats<kPrecision>( viewed ), std::vector<float>( all.begin(), all.begin() + 9 * 48 ) );
    }

    TYPED_TEST( LinearMetal, MatchesTheReferenceWhenOutputsAreOffTheTile )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        for ( const dim_t outputs : { dim_t{ 1 }, dim_t{ 3 }, dim_t{ 5 }, dim_t{ 15 }, dim_t{ 17 }, dim_t{ 35 } } )
        {
            Fixture<kPrecision> fixture( 64, outputs, 128 );

            for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 7 }, dim_t{ 8 }, dim_t{ 19 }, dim_t{ 128 } } )
            {
                EXPECT_LE( fixture.errorAt( rows ), kBar<kPrecision> ) << outputs << " outputs, " << rows << " rows";
            }
        }
    }

    TYPED_TEST( LinearMetal, MatchesTheReferenceWhenInputsAreOffAVector )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        for ( const dim_t inputs : { dim_t{ 3 }, dim_t{ 13 }, dim_t{ 65 } } )
        {
            Fixture<kPrecision> fixture( inputs, 32, 32 );

            for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 3 }, dim_t{ 8 }, dim_t{ 19 } } )
            {
                EXPECT_LE( fixture.errorAt( rows ), kBar<kPrecision> ) << inputs << " inputs, " << rows << " rows";
            }
        }
    }

    TYPED_TEST( LinearMetal, MatchesTheReferenceWithABias )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Fixture<kPrecision> fixture( 64, 48, 256, true );

        for ( const dim_t rows : kEveryRowCount )
        {
            EXPECT_LE( fixture.errorAt( rows ), kBar<kPrecision> ) << rows << " rows";
        }
    }

    TYPED_TEST( LinearMetal, AnEarlierCallLeavesNoTraceInALaterOne )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Fixture<kPrecision> fixture( 64, 48, 128 );

        EXPECT_LE( fixture.errorAt( 128 ), kBar<kPrecision> );
        EXPECT_LE( fixture.errorAt( 1 ), kBar<kPrecision> );
        EXPECT_LE( fixture.errorAt( 128 ), kBar<kPrecision> );
    }

    TYPED_TEST( LinearMetal, RefusesABlobOfAnotherType )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Fixture<kPrecision> fixture( 8, 8, 8 );
        const std::vector<std::uint16_t> bits( 64, 0 );

        EXPECT_THROW( fixture.linear->loadParameter( "weight", blobOver( bits, shape_t{ 8, 8 }, TensorDataType::INT16 ) ), std::invalid_argument );
    }

    TEST( LinearMetalFp16, HoldsBf16WeightsExactlyInItsNormalRange )
    {
        Fixture<TensorDataType::FP16> fixture( 4, 1, 1 );

        // 3 at the first input, so that the output is the weight read back, times 3.
        const std::array<float, 4> values{ 3.0f, 0.0f, 0.0f, 0.0f };
        const std::vector<std::uint16_t> bits{ toBf16Bits( 1.0f / 256.0f ), 0, 0, 0 };

        fixture.linear->loadParameter( "weight", blobOver( bits, shape_t{ 1, 4 } ) );

        auto input = metalTensor<TensorDataType::FP16>( values, shape_t{ 1, 4 } );
        const auto& output = fixture.linear->forward( input );
        fixture.linear->synchronize();

        EXPECT_EQ( readFloats<TensorDataType::FP16>( output )[0], 3.0f / 256.0f );
    }

    TEST( LinearMetalFp16, RefusesAWeightFp16CannotHoldAndNamesIt )
    {
        Fixture<TensorDataType::FP16> fixture( 4, 1, 1 );

        for ( const float value : { 70144.0f, -70144.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() } )
        {
            const std::vector<std::uint16_t> bits{ 0, toBf16Bits( value ), 0, 0 };

            try
            {
                fixture.linear->loadParameter( "weight", blobOver( bits, shape_t{ 1, 4 } ) );
                FAIL() << "A weight of " << value << " was accepted";
            }
            catch ( const std::invalid_argument& error )
            {
                EXPECT_NE( std::string( error.what() ).find( "linear.weight" ), std::string::npos ) << error.what();
            }
        }
    }

    TEST( LinearMetalFp16, AcceptsTheLargestWeightBf16AndFp16BothHold )
    {
        Fixture<TensorDataType::FP16> fixture( 4, 1, 1 );

        // BF16's largest value below 65504: the next one up is 65536.
        const std::vector<std::uint16_t> bits{ toBf16Bits( 65280.0f ), 0, 0, 0 };

        EXPECT_NO_THROW( fixture.linear->loadParameter( "weight", blobOver( bits, shape_t{ 1, 4 } ) ) );
    }

    TEST( LinearMetalRows, OnlyFp16AndBf16Exist )
    {
        using namespace Mila::Dnn::Quant::Weight;

        EXPECT_TRUE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::FP16, NoWeightQuant> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::BF16, NoWeightQuant> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::LinearOp, DeviceType::Metal, TensorDataType::FP32, NoWeightQuant> ) );
    }
}
