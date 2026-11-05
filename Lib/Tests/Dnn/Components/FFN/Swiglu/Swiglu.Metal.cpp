/**
 * @file Swiglu.Metal.cpp
 * @brief Swiglu on Metal, against a double-precision reference.
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
        template<typename TPrecisionConstant>
        class SwigluMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( SwigluMetal, Precisions, PrecisionNames );

        template<TensorDataType TPrecision>
        double errorOf( const shape_t& built, const shape_t& shape, float sigma )
        {
            Swiglu<DeviceType::Metal, TPrecision> swiglu( "swiglu", SwigluConfig(), Device::Metal( 0 ) );
            swiglu.build( BuildContext( built, RuntimeMode::Inference, false ) );

            dim_t elements = 1;

            for ( const dim_t axis : shape )
            {
                elements *= axis;
            }

            const std::vector<float> values = randomValues( static_cast<std::size_t>( elements ), 51, sigma );
            auto input = metalTensor<TPrecision>( values, shape );

            const auto& output = swiglu.forward( input );
            swiglu.synchronize();

            const std::vector<double> x = asStored<TPrecision>( std::span<const float>( values ) );
            const dim_t half = shape.back() / 2;
            const dim_t rows = elements / shape.back();

            std::vector<double> reference( static_cast<std::size_t>( rows * half ) );

            for ( dim_t r = 0; r < rows; ++r )
            {
                for ( dim_t c = 0; c < half; ++c )
                {
                    const double gate = x[static_cast<std::size_t>( r * 2 * half + c )];
                    const double up = x[static_cast<std::size_t>( r * 2 * half + half + c )];

                    reference[static_cast<std::size_t>( r * half + c )] = gate / ( 1.0 + std::exp( -gate ) ) * up;
                }
            }

            return relativeError( readFloats<TPrecision>( output ), reference );
        }
    }

    TYPED_TEST( SwigluMetal, MatchesTheReferenceOnEachShape )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        EXPECT_LE( errorOf<kPrecision>( shape_t{ 1, 8, 64 }, shape_t{ 1, 8, 64 }, 1.0f ), kBar<kPrecision> );
        EXPECT_LE( errorOf<kPrecision>( shape_t{ 1, 8, 64 }, shape_t{ 1, 1, 64 }, 1.0f ), kBar<kPrecision> );
        EXPECT_LE( errorOf<kPrecision>( shape_t{ 3, 5, 14 }, shape_t{ 3, 5, 14 }, 1.0f ), kBar<kPrecision> );
    }

    TYPED_TEST( SwigluMetal, HoldsUpWhereTheGateIsLargeEitherWay )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        // A gate of -100 makes exp( 100 ), far past FP16's range. The result is near zero either way, so this checks that nothing turns
        // into a NaN, not that the kernel's arithmetic must be 32 bits: a forced failure in the element type also passes the bar.
        EXPECT_LE( errorOf<kPrecision>( shape_t{ 1, 4, 64 }, shape_t{ 1, 4, 64 }, 30.0f ), kBar<kPrecision> );
    }
}
