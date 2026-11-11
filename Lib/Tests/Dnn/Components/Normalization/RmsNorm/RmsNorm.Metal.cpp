/**
 * @file RmsNorm.Metal.cpp
 * @brief RmsNorm on Metal, against a double-precision reference, including values whose sum of squares FP16 could not hold.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Serialization.Tensor;

#include "../../MetalReference.h"

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Tests::Dnn::MetalReference;

    namespace
    {
        constexpr float kEpsilon = 1e-5f;

        template<typename TPrecisionConstant>
        class RmsNormMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( RmsNormMetal, Precisions, PrecisionNames );

        struct Case
        {
            dim_t width;
            dim_t rows;
            float sigma{ 1.0f };
            bool with_bias{ false };
            float unit_offset{ 0.0f };
            dim_t group_size{ 0 };
        };

        template<TensorDataType TPrecision>
        double errorOf( const Case& test )
        {
            RmsNormConfig config( shape_t{ test.width } );
            config.withEpsilon( kEpsilon ).withBias( test.with_bias ).withUnitOffset( test.unit_offset ).withGroupSize( test.group_size );

            RmsNorm<DeviceType::Metal, TPrecision> norm( "norm", config, Device::Metal( 0 ) );
            norm.build( BuildContext( shape_t{ test.rows, test.width }, RuntimeMode::Inference, false ) );

            const std::vector<std::uint16_t> weight = randomBf16( static_cast<std::size_t>( test.width ), 31, 0.5f );
            norm.loadParameter( "weight", blobOver( weight, shape_t{ test.width } ) );

            std::vector<std::uint16_t> bias;

            if ( test.with_bias )
            {
                bias = randomBf16( static_cast<std::size_t>( test.width ), 32, 0.2f );
                norm.loadParameter( "bias", blobOver( bias, shape_t{ test.width } ) );
            }

            const std::vector<float> values = randomValues( static_cast<std::size_t>( test.rows * test.width ), 33, test.sigma );
            auto input = metalTensor<TPrecision>( values, shape_t{ test.rows, test.width } );

            const auto& output = norm.forward( input );
            norm.synchronize();

            const std::vector<float> actual = readFloats<TPrecision>( output );
            const std::vector<double> x = asStored<TPrecision>( std::span<const float>( values ) );

            const dim_t group = test.group_size > 0 ? test.group_size : test.width;
            std::vector<double> reference( x.size() );

            for ( dim_t r = 0; r < test.rows; ++r )
            {
                for ( dim_t first = 0; first < test.width; first += group )
                {
                    double sum = 0.0;

                    for ( dim_t k = 0; k < group; ++k )
                    {
                        const double value = x[static_cast<std::size_t>( r * test.width + first + k )];

                        sum += value * value;
                    }

                    const double inverse_deviation = 1.0 / std::sqrt( sum / static_cast<double>( group ) + static_cast<double>( kEpsilon ) );

                    for ( dim_t k = 0; k < group; ++k )
                    {
                        const std::size_t element = static_cast<std::size_t>( first + k );

                        const double scaled = x[static_cast<std::size_t>( r * test.width ) + element] * inverse_deviation
                            * ( asStored<TPrecision>( fromBf16Bits( weight[element] ) ) + static_cast<double>( test.unit_offset ) );

                        reference[static_cast<std::size_t>( r * test.width ) + element] = bias.empty()
                            ? scaled : scaled + asStored<TPrecision>( fromBf16Bits( bias[element] ) );
                    }
                }
            }

            return relativeError( actual, reference );
        }
    }

    TYPED_TEST( RmsNormMetal, MatchesTheReferenceAtEveryRowCount )
    {
        for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 2 }, dim_t{ 7 }, dim_t{ 8 }, dim_t{ 129 } } )
        {
            EXPECT_LE( errorOf<TypeParam::value>( { .width = 64, .rows = rows } ), kBar<TypeParam::value> ) << rows << " rows";
        }
    }

    TYPED_TEST( RmsNormMetal, MatchesTheReferenceWhenTheWidthIsOffASimdGroup )
    {
        for ( const dim_t width : { dim_t{ 1 }, dim_t{ 3 }, dim_t{ 31 }, dim_t{ 33 } } )
        {
            EXPECT_LE( errorOf<TypeParam::value>( { .width = width, .rows = 3 } ), kBar<TypeParam::value> ) << width << " wide";
        }
    }

    // A slice wider than 1024 takes the threadgroup of 1024 threads, four values a load: a width each thread reads once, widths
    // that leave one, two and three values past the last four, and one a thread reads twice with values left over.
    TYPED_TEST( RmsNormMetal, MatchesTheReferenceAcrossTheWiderThreadgroup )
    {
        for ( const dim_t width : { dim_t{ 1028 }, dim_t{ 3840 }, dim_t{ 3841 }, dim_t{ 3842 }, dim_t{ 3843 }, dim_t{ 5122 } } )
        {
            EXPECT_LE( errorOf<TypeParam::value>( { .width = width, .rows = 3 } ), kBar<TypeParam::value> ) << width << " wide";
        }

        EXPECT_LE( errorOf<TypeParam::value>( { .width = 3842, .rows = 2, .with_bias = true, .unit_offset = 1.0f } ), kBar<TypeParam::value> );
    }

    // The smaller threadgroup at its widest slice and at widths its threads read twice or leave values past the last four, and
    // groups that start off a four-value boundary: Gemma's head norms are slices of 256 of a wider row.
    TYPED_TEST( RmsNormMetal, MatchesTheReferenceAtTheSmallerThreadgroupsEdges )
    {
        for ( const dim_t width : { dim_t{ 1021 }, dim_t{ 1023 }, dim_t{ 1024 } } )
        {
            EXPECT_LE( errorOf<TypeParam::value>( { .width = width, .rows = 3 } ), kBar<TypeParam::value> ) << width << " wide";
        }

        EXPECT_LE( errorOf<TypeParam::value>( { .width = 4096, .rows = 3, .group_size = 256 } ), kBar<TypeParam::value> );
        EXPECT_LE( errorOf<TypeParam::value>( { .width = 63, .rows = 5, .with_bias = true, .group_size = 7 } ), kBar<TypeParam::value> );
        EXPECT_LE( errorOf<TypeParam::value>( { .width = 6150, .rows = 2, .group_size = 2050 } ), kBar<TypeParam::value> );
    }

    TYPED_TEST( RmsNormMetal, AWidthWhoseSumOfSquaresFp16CannotHoldStaysCorrect )
    {
        // 4096 values of about 10 sum to about 400000, against FP16's largest of 65504.
        EXPECT_LE( errorOf<TypeParam::value>( { .width = 4096, .rows = 2, .sigma = 10.0f } ), kBar<TypeParam::value> );
    }

    TYPED_TEST( RmsNormMetal, TakesABiasAndAUnitOffset )
    {
        EXPECT_LE( errorOf<TypeParam::value>( { .width = 64, .rows = 3, .with_bias = true } ), kBar<TypeParam::value> );
        EXPECT_LE( errorOf<TypeParam::value>( { .width = 64, .rows = 3, .unit_offset = 1.0f } ), kBar<TypeParam::value> );
    }

    TYPED_TEST( RmsNormMetal, TakesEachGroupOfARowOnItsOwn )
    {
        EXPECT_LE( errorOf<TypeParam::value>( { .width = 64, .rows = 3, .group_size = 8 } ), kBar<TypeParam::value> );
        EXPECT_LE( errorOf<TypeParam::value>( { .width = 64, .rows = 3, .with_bias = true, .group_size = 16 } ), kBar<TypeParam::value> );
    }

    TYPED_TEST( RmsNormMetal, ATallRowDoesNotBleedIntoTheNext )
    {
        // One row far larger than the next: a sum across rows would flatten the small one.
        constexpr TensorDataType kPrecision = TypeParam::value;

        RmsNorm<DeviceType::Metal, kPrecision> norm( "norm", RmsNormConfig( shape_t{ 4 } ).withEpsilon( kEpsilon ), Device::Metal( 0 ) );
        norm.build( BuildContext( shape_t{ 2, 4 }, RuntimeMode::Inference, false ) );

        const std::vector<std::uint16_t> weight( 4, toBf16Bits( 1.0f ) );
        norm.loadParameter( "weight", blobOver( weight, shape_t{ 4 } ) );

        const std::array<float, 8> values{ 100.0f, 100.0f, 100.0f, 100.0f, 0.5f, 0.5f, 0.5f, 0.5f };
        auto input = metalTensor<kPrecision>( values, shape_t{ 2, 4 } );

        const auto& output = norm.forward( input );
        norm.synchronize();

        const std::vector<float> actual = readFloats<kPrecision>( output );

        for ( const float value : actual )
        {
            EXPECT_NEAR( value, 1.0f, 2e-2f );
        }
    }

    TEST( RmsNormMetalBuild, RefusesAnAxisThatIsNotTheInnermost )
    {
        RmsNorm<DeviceType::Metal, TensorDataType::FP16> norm( "norm", RmsNormConfig( shape_t{ 4, 8 } ), Device::Metal( 0 ) );

        EXPECT_THROW( norm.build( BuildContext( shape_t{ 2, 4, 8 }, RuntimeMode::Inference, false ) ), std::invalid_argument );
    }
}
