/**
 * @file Geglu.Metal.cpp
 * @brief GeGLU on Metal, Gemma's feed-forward gate, against a double-precision reference of GELU's tanh form.
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
        class GegluMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( GegluMetal, Precisions, PrecisionNames );

        /// GELU's tanh form, HuggingFace's gelu_pytorch_tanh, which Gemma 4 names.
        double geluTanh( double x )
        {
            return 0.5 * x * ( 1.0 + std::tanh( std::sqrt( 2.0 / std::numbers::pi ) * ( x + 0.044715 * x * x * x ) ) );
        }

        template<TensorDataType TPrecision>
        double errorOf( const shape_t& built, const shape_t& shape, float sigma )
        {
            Swiglu<DeviceType::Metal, TPrecision, ActivationType::Gelu> geglu( "geglu", SwigluConfig(), Device::Metal( 0 ) );
            geglu.build( BuildContext( built, RuntimeMode::Inference, false ) );

            dim_t elements = 1;

            for ( const dim_t axis : shape )
            {
                elements *= axis;
            }

            const std::vector<float> values = randomValues( static_cast<std::size_t>( elements ), 53, sigma );
            auto input = metalTensor<TPrecision>( values, shape );

            const auto& output = geglu.forward( input );
            geglu.synchronize();

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

                    reference[static_cast<std::size_t>( r * half + c )] = geluTanh( gate ) * up;
                }
            }

            return relativeError( readFloats<TPrecision>( output ), reference );
        }
    }

    TYPED_TEST( GegluMetal, MatchesTheReferenceOnEachShape )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        EXPECT_LE( errorOf<kPrecision>( shape_t{ 1, 8, 64 }, shape_t{ 1, 8, 64 }, 1.0f ), kBar<kPrecision> );
        EXPECT_LE( errorOf<kPrecision>( shape_t{ 1, 8, 64 }, shape_t{ 1, 1, 64 }, 1.0f ), kBar<kPrecision> );
        EXPECT_LE( errorOf<kPrecision>( shape_t{ 3, 5, 14 }, shape_t{ 3, 5, 14 }, 1.0f ), kBar<kPrecision> );
    }

    TYPED_TEST( GegluMetal, MatchesTheReferenceAtGemmasWidth )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        // Gemma 4 12B's hidden of 15360, a prefill's rows and a decode's.
        EXPECT_LE( errorOf<kPrecision>( shape_t{ 1, 16, 30720 }, shape_t{ 1, 16, 30720 }, 2.0f ), kBar<kPrecision> );
        EXPECT_LE( errorOf<kPrecision>( shape_t{ 1, 16, 30720 }, shape_t{ 1, 1, 30720 }, 2.0f ), kBar<kPrecision> );
    }

    TYPED_TEST( GegluMetal, HoldsUpWhereTheGateIsLargeEitherWay )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        // A gate of 30 cubed is 27000 inside tanh: the result must be the gate, or zero, and never a NaN.
        EXPECT_LE( errorOf<kPrecision>( shape_t{ 1, 4, 64 }, shape_t{ 1, 4, 64 }, 30.0f ), kBar<kPrecision> );
    }
}
