/**
 * @file Rope.Metal.cpp
 * @brief Rope on Metal, against a double-precision reference, with a decode's position given as a network gives it.
 *
 * Positions stay low enough that the kernel's single-precision angle is within the bar. CUDA's kernel has the same angle, and a far
 * position is a model's question, for Phase 4.
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
        constexpr dim_t kHeadDim = 8;
        constexpr dim_t kHeads = 4;
        constexpr dim_t kKvHeads = 2;
        constexpr dim_t kChannels = kHeads * kHeadDim;
        constexpr dim_t kKvChannels = kKvHeads * kHeadDim;
        constexpr dim_t kMaxSequence = 2048;
        constexpr double kBase = 10000.0;

        template<typename TPrecisionConstant>
        class RopeMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( RopeMetal, Precisions, PrecisionNames );

        /// What sets one configuration's angles, as the formula in Rope.Angle.cuh and HuggingFace's _compute_llama3_parameters have it.
        struct Spectrum
        {
            dim_t half{ kHeadDim / 2 };
            dim_t denominator{ kHeadDim };
            RopeFrequencyScaling scaling{};

            [[nodiscard]] double inverseFrequency( dim_t pair ) const
            {
                const double inverse = std::pow( kBase, -2.0 * static_cast<double>( pair ) / static_cast<double>( denominator ) );

                if ( scaling.original_context_length <= 0 )
                {
                    return inverse;
                }

                const double original = static_cast<double>( scaling.original_context_length );
                const double factor = static_cast<double>( scaling.factor );
                const double low = static_cast<double>( scaling.low_frequency_factor );
                const double high = static_cast<double>( scaling.high_frequency_factor );
                const double wavelength = 2.0 * std::numbers::pi / inverse;

                if ( wavelength > original / low )
                {
                    return inverse / factor;
                }

                if ( wavelength < original / high )
                {
                    return inverse;
                }

                const double smooth = ( original / wavelength - low ) / ( high - low );

                return ( 1.0 - smooth ) * inverse / factor + smooth * inverse;
            }
        };

        std::vector<double> rotated(
            std::vector<double> values, dim_t batches, dim_t tokens, dim_t heads, dim_t first_position, dim_t pairs_turning,
            const Spectrum& spectrum = Spectrum{} )
        {
            const dim_t half = spectrum.half;

            for ( dim_t b = 0; b < batches; ++b )
            {
                for ( dim_t t = 0; t < tokens; ++t )
                {
                    for ( dim_t h = 0; h < heads; ++h )
                    {
                        const std::size_t first = static_cast<std::size_t>( ( ( b * tokens + t ) * heads + h ) * kHeadDim );

                        for ( dim_t i = 0; i < pairs_turning; ++i )
                        {
                            const double angle = static_cast<double>( first_position + t ) * spectrum.inverseFrequency( i );

                            const std::size_t a = first + static_cast<std::size_t>( i );
                            const std::size_t c = a + static_cast<std::size_t>( half );
                            const double x0 = values[a];
                            const double x1 = values[c];

                            values[a] = x0 * std::cos( angle ) - x1 * std::sin( angle );
                            values[c] = x0 * std::sin( angle ) + x1 * std::cos( angle );
                        }
                    }
                }
            }

            return values;
        }

        template<TensorDataType TPrecision>
        struct Inputs
        {
            std::vector<float> q_values;
            std::vector<float> k_values;
            Tensor<TPrecision, MetalMemoryResource> q;
            Tensor<TPrecision, MetalMemoryResource> k;

            Inputs( dim_t batches, dim_t tokens )
                : q_values( randomValues( static_cast<std::size_t>( batches * tokens * kChannels ), 41, 1.0f ) ),
                  k_values( randomValues( static_cast<std::size_t>( batches * tokens * kKvChannels ), 42, 1.0f ) ),
                  q( metalTensor<TPrecision>( q_values, shape_t{ batches, tokens, kChannels } ) ),
                  k( metalTensor<TPrecision>( k_values, shape_t{ batches, tokens, kKvChannels } ) )
            {
            }

            void expect( dim_t batches, dim_t tokens, dim_t first_position )
            {
                const std::vector<double> q_expected = rotated(
                    asStored<TPrecision>( std::span<const float>( q_values ) ), batches, tokens, kHeads, first_position, kHeadDim / 2 );
                const std::vector<double> k_expected = rotated(
                    asStored<TPrecision>( std::span<const float>( k_values ) ), batches, tokens, kKvHeads, first_position, kHeadDim / 2 );

                EXPECT_LE( relativeError( readFloats<TPrecision>( q ), q_expected ), kBar<TPrecision> ) << "Q, position " << first_position;
                EXPECT_LE( relativeError( readFloats<TPrecision>( k ), k_expected ), kBar<TPrecision> ) << "K, position " << first_position;
            }
        };

        RopeConfig ropeConfig()
        {
            return RopeConfig( kChannels, kHeads, kKvHeads, kMaxSequence );
        }
    }

    TYPED_TEST( RopeMetal, ForwardRotatesFromPositionZero )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Rope<DeviceType::Metal, kPrecision> rope( "rope", ropeConfig(), Device::Metal( 0 ) );
        rope.build( BuildContext( shape_t{ 2, 16 }, RuntimeMode::Inference, false ) );

        Inputs<kPrecision> inputs( 2, 16 );
        rope.forward( inputs.q, inputs.k );
        rope.synchronize();

        inputs.expect( 2, 16, 0 );
    }

    TYPED_TEST( RopeMetal, PrefillRotatesFromItsOffset )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Rope<DeviceType::Metal, kPrecision> rope( "rope", ropeConfig(), Device::Metal( 0 ) );
        rope.build( BuildContext( shape_t{ 1, kMaxSequence }, RuntimeMode::Inference, false ) );

        Inputs<kPrecision> inputs( 1, 32 );
        rope.prefill( inputs.q, inputs.k, 1000 );
        rope.synchronize();

        inputs.expect( 1, 32, 1000 );
    }

    TYPED_TEST( RopeMetal, DecodeRotatesAtThePositionSetOnTheContext )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Rope<DeviceType::Metal, kPrecision> rope( "rope", ropeConfig(), Device::Metal( 0 ) );
        rope.build( BuildContext( shape_t{ 2, kMaxSequence }, RuntimeMode::Inference, false ) );

        for ( const dim_t position : { dim_t{ 0 }, dim_t{ 3 }, dim_t{ 1500 } } )
        {
            Inputs<kPrecision> inputs( 2, 1 );

            rope.getExecutionContext()->setDecodePosition( position );
            rope.decode( inputs.q, inputs.k, position );
            rope.synchronize();

            inputs.expect( 2, 1, position );
        }
    }

    TYPED_TEST( RopeMetal, ADecodeReadsThePositionFromMemoryNotFromItsArgument )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Rope<DeviceType::Metal, kPrecision> rope( "rope", ropeConfig(), Device::Metal( 0 ) );
        rope.build( BuildContext( shape_t{ 1, kMaxSequence }, RuntimeMode::Inference, false ) );

        Inputs<kPrecision> inputs( 1, 1 );

        rope.getExecutionContext()->setDecodePosition( 700 );
        rope.decode( inputs.q, inputs.k, 3 );
        rope.synchronize();

        inputs.expect( 1, 1, 700 );
    }

    TYPED_TEST( RopeMetal, ADecodeWithNoPositionSetIsRefused )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Rope<DeviceType::Metal, kPrecision> rope( "rope", ropeConfig(), Device::Metal( 0 ) );
        rope.build( BuildContext( shape_t{ 1, 8 }, RuntimeMode::Inference, false ) );

        Inputs<kPrecision> inputs( 1, 1 );

        EXPECT_THROW( rope.decode( inputs.q, inputs.k, 0 ), std::logic_error );
    }

    TYPED_TEST( RopeMetal, APartialRotaryLeavesThePairsPastItUnturned )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Rope<DeviceType::Metal, kPrecision> rope( "rope", ropeConfig().withRotaryDim( 4 ), Device::Metal( 0 ) );
        rope.build( BuildContext( shape_t{ 1, 8 }, RuntimeMode::Inference, false ) );

        Inputs<kPrecision> inputs( 1, 8 );
        rope.forward( inputs.q, inputs.k );
        rope.synchronize();

        // Two pairs of four turn, on the whole-head spectrum.
        const std::vector<double> q_expected = rotated(
            asStored<kPrecision>( std::span<const float>( inputs.q_values ) ), 1, 8, kHeads, 0, 2 );

        EXPECT_LE( relativeError( readFloats<kPrecision>( inputs.q ), q_expected ), kBar<kPrecision> );
    }

    TYPED_TEST( RopeMetal, APrefixLayoutCompressesItsSpectrumAndLeavesTheRestAlone )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Rope<DeviceType::Metal, kPrecision> rope(
            "rope", ropeConfig().withRotaryDim( 4 ).withRotaryLayout( RotaryLayout::RotaryPrefix ), Device::Metal( 0 ) );
        rope.build( BuildContext( shape_t{ 1, 64 }, RuntimeMode::Inference, false ) );

        Inputs<kPrecision> inputs( 1, 8 );
        rope.prefill( inputs.q, inputs.k, 40 );
        rope.synchronize();

        const std::vector<double> q_expected = rotated(
            asStored<kPrecision>( std::span<const float>( inputs.q_values ) ), 1, 8, kHeads, 40, 2, Spectrum{ .half = 2, .denominator = 4 } );

        EXPECT_LE( relativeError( readFloats<kPrecision>( inputs.q ), q_expected ), kBar<kPrecision> );
    }

    TYPED_TEST( RopeMetal, FrequencyScalingFollowsTheWavelength )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        // Llama 3's own scaling, on a head wide enough that its pairs fall in all three bands.
        constexpr dim_t kWideHead = 256;
        const RopeFrequencyScaling scaling{ 32.0f, 1.0f, 4.0f, 8192 };

        Rope<DeviceType::Metal, kPrecision> rope(
            "rope", RopeConfig( kWideHead, 1, 1, kMaxSequence ).withFrequencyScaling( scaling ), Device::Metal( 0 ) );
        rope.build( BuildContext( shape_t{ 1, kMaxSequence }, RuntimeMode::Inference, false ) );

        const std::vector<float> q_values = randomValues( kWideHead, 43, 1.0f );
        const std::vector<float> k_values = randomValues( kWideHead, 44, 1.0f );
        auto q = metalTensor<kPrecision>( q_values, shape_t{ 1, 1, kWideHead } );
        auto k = metalTensor<kPrecision>( k_values, shape_t{ 1, 1, kWideHead } );

        rope.prefill( q, k, 1000 );
        rope.synchronize();

        const Spectrum spectrum{ .half = kWideHead / 2, .denominator = kWideHead, .scaling = scaling };

        std::vector<double> expected = asStored<kPrecision>( std::span<const float>( q_values ) );

        for ( dim_t i = 0; i < kWideHead / 2; ++i )
        {
            const double angle = 1000.0 * spectrum.inverseFrequency( i );
            const std::size_t a = static_cast<std::size_t>( i );
            const std::size_t c = a + static_cast<std::size_t>( kWideHead / 2 );
            const double x0 = expected[a];
            const double x1 = expected[c];

            expected[a] = x0 * std::cos( angle ) - x1 * std::sin( angle );
            expected[c] = x0 * std::sin( angle ) + x1 * std::cos( angle );
        }

        EXPECT_LE( relativeError( readFloats<kPrecision>( q ), expected ), kBar<kPrecision> );
    }

    TYPED_TEST( RopeMetal, SettingTheNextPositionLeavesEncodedWorkAtItsOwn )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Rope<DeviceType::Metal, kPrecision> rope( "rope", ropeConfig(), Device::Metal( 0 ) );
        rope.build( BuildContext( shape_t{ 1, kMaxSequence }, RuntimeMode::Inference, false ) );

        Inputs<kPrecision> first( 1, 1 );
        Inputs<kPrecision> second( 1, 1 );

        // Nothing synchronizes the first decode by hand: setting the position is what must let it finish at 300.
        rope.getExecutionContext()->setDecodePosition( 300 );
        rope.decode( first.q, first.k, 300 );

        rope.getExecutionContext()->setDecodePosition( 900 );
        rope.decode( second.q, second.k, 900 );
        rope.synchronize();

        first.expect( 1, 1, 300 );
        second.expect( 1, 1, 900 );
    }

    TYPED_TEST( RopeMetal, RefusesPositionsPastWhatItWasBuiltFor )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Rope<DeviceType::Metal, kPrecision> rope( "rope", ropeConfig(), Device::Metal( 0 ) );
        rope.build( BuildContext( shape_t{ 1, 8 }, RuntimeMode::Inference, false ) );

        Inputs<kPrecision> inputs( 1, 4 );

        EXPECT_THROW( rope.prefill( inputs.q, inputs.k, 5 ), std::invalid_argument );
    }
}
