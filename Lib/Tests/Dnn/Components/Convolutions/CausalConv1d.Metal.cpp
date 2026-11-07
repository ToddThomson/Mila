/**
 * @file CausalConv1d.Metal.cpp
 * @brief The depthwise causal convolution on Metal, against a double-precision convolution of the whole sequence.
 *
 * Prefill from the start, prefill continuing one, and decodes after it must each give what the whole sequence convolved at
 * once gives, and the retained rows must be the sequence's last rows, exactly.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Serialization.Tensor;

#include "../MetalReference.h"

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Tests::Dnn::MetalReference;

    namespace
    {
        template<typename TPrecisionConstant>
        class CausalConv1dMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( CausalConv1dMetal, Precisions, PrecisionNames );

        constexpr dim_t kKernelWidth = 4;

        struct Geometry
        {
            dim_t batch;
            dim_t steps;
            dim_t channels;
            dim_t dilation;
            bool has_bias;
        };

        /// The filter, bias and input of one case, as BF16 would have stored them.
        struct Case
        {
            Geometry geometry;
            std::vector<float> weight;
            std::vector<float> bias;
            std::vector<float> input;
        };

        Case caseOf( const Geometry& geometry, std::uint32_t seed )
        {
            return Case{ geometry,
                randomValues( static_cast<std::size_t>( geometry.channels * kKernelWidth ), seed, 0.5f ),
                randomValues( static_cast<std::size_t>( geometry.channels ), seed + 1, 0.5f ),
                randomValues( static_cast<std::size_t>( geometry.batch * geometry.steps * geometry.channels ), seed + 2, 1.0f ) };
        }

        /// The whole sequence convolved, with zeros left of its start, in double precision on the values as stored.
        template<TensorDataType TPrecision>
        std::vector<double> referenceOf( const Case& c )
        {
            const Geometry& g = c.geometry;
            const std::vector<double> x = asStored<TPrecision>( std::span<const float>( c.input ) );
            const std::vector<double> w = asStored<TPrecision>( std::span<const float>( c.weight ) );
            const std::vector<double> bias = asStored<TPrecision>( std::span<const float>( c.bias ) );
            const dim_t state_rows = ( kKernelWidth - 1 ) * g.dilation;

            std::vector<double> out( x.size() );

            for ( dim_t b = 0; b < g.batch; ++b )
            {
                for ( dim_t t = 0; t < g.steps; ++t )
                {
                    for ( dim_t ch = 0; ch < g.channels; ++ch )
                    {
                        double total = g.has_bias ? bias[static_cast<std::size_t>( ch )] : 0.0;

                        for ( dim_t i = 0; i < kKernelWidth; ++i )
                        {
                            const dim_t source = t - state_rows + i * g.dilation;

                            if ( source >= 0 )
                            {
                                total += w[static_cast<std::size_t>( ch * kKernelWidth + i )]
                                    * x[static_cast<std::size_t>( ( b * g.steps + source ) * g.channels + ch )];
                            }
                        }

                        out[static_cast<std::size_t>( ( b * g.steps + t ) * g.channels + ch )] = total;
                    }
                }
            }

            return out;
        }

        template<TensorDataType TPrecision>
        using MetalTensorOf = Tensor<TPrecision, MetalMemoryResource>;

        template<TensorDataType TPrecision>
        std::unique_ptr<CausalConv1d<DeviceType::Metal, TPrecision>> builtConvolution( const Case& c, dim_t built_steps )
        {
            const Geometry& g = c.geometry;
            auto convolution = std::make_unique<CausalConv1d<DeviceType::Metal, TPrecision>>( "conv",
                CausalConv1dConfig( g.channels, kKernelWidth ).withDilation( g.dilation ).withBias( g.has_bias ), Device::Metal( 0 ) );
            convolution->build( BuildContext( shape_t{ g.batch, built_steps, g.channels }, RuntimeMode::Inference, false ) );

            const std::vector<ITensor*> parameters = convolution->getParameters();
            fill( *static_cast<MetalTensorOf<TPrecision>*>( parameters[0] ), std::span<const float>( c.weight ) );

            if ( g.has_bias )
            {
                fill( *static_cast<MetalTensorOf<TPrecision>*>( parameters[1] ), std::span<const float>( c.bias ) );
            }

            return convolution;
        }

        /// Rows [first, first + count) of every batch item, as a [B, count, C] Metal tensor.
        template<TensorDataType TPrecision>
        MetalTensorOf<TPrecision> stepsOf( const Case& c, dim_t first, dim_t count )
        {
            const Geometry& g = c.geometry;
            std::vector<float> values;

            for ( dim_t b = 0; b < g.batch; ++b )
            {
                const auto begin = c.input.begin() + static_cast<std::ptrdiff_t>( ( b * g.steps + first ) * g.channels );
                values.insert( values.end(), begin, begin + static_cast<std::ptrdiff_t>( count * g.channels ) );
            }

            return metalTensor<TPrecision>( values, shape_t{ g.batch, count, g.channels } );
        }

        /// Write a [B, count, C] result into rows [first, first + count) of a whole-sequence array.
        void place( std::vector<float>& whole, const std::vector<float>& part, const Geometry& g, dim_t first, dim_t count )
        {
            for ( dim_t b = 0; b < g.batch; ++b )
            {
                std::ranges::copy( part.begin() + static_cast<std::ptrdiff_t>( b * count * g.channels ),
                    part.begin() + static_cast<std::ptrdiff_t>( ( b + 1 ) * count * g.channels ),
                    whole.begin() + static_cast<std::ptrdiff_t>( ( b * g.steps + first ) * g.channels ) );
            }
        }

        /// The sequence convolved in pieces: a prefill of `first_chunk` from the start, then the rest as decodes or one prefill.
        template<TensorDataType TPrecision>
        std::vector<float> inPieces( const Case& c, dim_t first_chunk, bool decode_the_rest )
        {
            const Geometry& g = c.geometry;
            auto convolution = builtConvolution<TPrecision>( c, g.steps );
            std::vector<float> whole( c.input.size() );

            auto first = stepsOf<TPrecision>( c, 0, first_chunk );
            auto& first_output = convolution->prefill( first, 0 );
            convolution->synchronize();
            place( whole, readFloats<TPrecision>( first_output ), g, 0, first_chunk );

            if ( decode_the_rest )
            {
                for ( dim_t t = first_chunk; t < g.steps; ++t )
                {
                    auto step = stepsOf<TPrecision>( c, t, 1 );
                    auto& output = convolution->decode( step, t );
                    convolution->synchronize();
                    place( whole, readFloats<TPrecision>( output ), g, t, 1 );
                }
            } else {
                auto rest = stepsOf<TPrecision>( c, first_chunk, g.steps - first_chunk );
                auto& output = convolution->prefill( rest, first_chunk );
                convolution->synchronize();
                place( whole, readFloats<TPrecision>( output ), g, first_chunk, g.steps - first_chunk );
            }

            return whole;
        }

        template<TensorDataType TPrecision>
        double wholeError( const Case& c )
        {
            auto convolution = builtConvolution<TPrecision>( c, c.geometry.steps );
            auto input = stepsOf<TPrecision>( c, 0, c.geometry.steps );

            auto& output = convolution->prefill( input, 0 );
            convolution->synchronize();

            return relativeError( readFloats<TPrecision>( output ), referenceOf<TPrecision>( c ) );
        }
    }

    TYPED_TEST( CausalConv1dMetal, AWholePrefillMatchesTheReference )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        EXPECT_LE( wholeError<kPrecision>( caseOf( Geometry{ 1, 9, 24, 1, false }, 3 ) ), kBar<kPrecision> );
        EXPECT_LE( wholeError<kPrecision>( caseOf( Geometry{ 2, 9, 24, 1, true }, 5 ) ), kBar<kPrecision> );
        EXPECT_LE( wholeError<kPrecision>( caseOf( Geometry{ 2, 11, 24, 2, true }, 7 ) ), kBar<kPrecision> );

        // Qwen 3.8's convolution over q, k and v: 2 * 16 * 128 + 48 * 128 channels.
        EXPECT_LE( wholeError<kPrecision>( caseOf( Geometry{ 1, 64, 10240, 1, false }, 11 ) ), kBar<kPrecision> );
    }

    TYPED_TEST( CausalConv1dMetal, PiecesGiveTheWholeSequence )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        for ( const Geometry geometry : { Geometry{ 2, 13, 24, 1, true }, Geometry{ 1, 13, 24, 2, false } } )
        {
            const Case c = caseOf( geometry, 13 );
            const std::vector<double> expected = referenceOf<kPrecision>( c );

            // A second prefill continuing the first, a chunk shorter than the retained rows, and decodes.
            EXPECT_LE( relativeError( inPieces<kPrecision>( c, 8, false ), expected ), kBar<kPrecision> ) << "dilation " << geometry.dilation;
            EXPECT_LE( relativeError( inPieces<kPrecision>( c, 2, false ), expected ), kBar<kPrecision> ) << "dilation " << geometry.dilation;
            EXPECT_LE( relativeError( inPieces<kPrecision>( c, 1, true ), expected ), kBar<kPrecision> ) << "dilation " << geometry.dilation;
        }
    }

    TYPED_TEST( CausalConv1dMetal, RetainsTheSequencesLastRowsExactly )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        const Geometry g{ 2, 10, 24, 2, false };
        const Case c = caseOf( g, 17 );
        const dim_t state_rows = ( kKernelWidth - 1 ) * g.dilation;

        auto convolution = builtConvolution<kPrecision>( c, g.steps );
        auto first = stepsOf<kPrecision>( c, 0, 7 );
        convolution->prefill( first, 0 );

        for ( dim_t t = 7; t < g.steps; ++t )
        {
            auto step = stepsOf<kPrecision>( c, t, 1 );
            convolution->decode( step, t );
        }

        auto snapshot = convolution->makeStateSnapshot();
        convolution->snapshotState( snapshot );

        const std::vector<double> stored = asStored<kPrecision>( std::span<const float>( c.input ) );

        for ( dim_t b = 0; b < g.batch; ++b )
        {
            for ( dim_t j = 0; j < state_rows; ++j )
            {
                for ( dim_t ch = 0; ch < g.channels; ++ch )
                {
                    const dim_t source = ( b * g.steps + g.steps - state_rows + j ) * g.channels + ch;

                    ASSERT_EQ( static_cast<double>( snapshot.data()[( b * state_rows + j ) * g.channels + ch] ), stored[static_cast<std::size_t>( source )] )
                        << "batch " << b << ", row " << j << ", channel " << ch;
                }
            }
        }
    }
}
