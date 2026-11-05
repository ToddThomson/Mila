/**
 * @file Residual.Metal.cpp
 * @brief Residual on Metal, against a double-precision reference.
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
        class ResidualMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( ResidualMetal, Precisions, PrecisionNames );
    }

    TYPED_TEST( ResidualMetal, AddsItsInputsOnEachShape )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Residual<DeviceType::Metal, kPrecision> residual( "residual", ResidualConfig(), Device::Metal( 0 ) );
        residual.build( BuildContext( shape_t{ 2, 16, 64 }, RuntimeMode::Inference, false ) );

        for ( const shape_t& shape : { shape_t{ 2, 16, 64 }, shape_t{ 1, 1, 64 } } )
        {
            const auto count = static_cast<std::size_t>( shape[0] * shape[1] * shape[2] );
            const std::vector<float> a = randomValues( count, 61, 1.0f );
            const std::vector<float> b = randomValues( count, 62, 1.0f );

            auto first = metalTensor<kPrecision>( a, shape );
            auto second = metalTensor<kPrecision>( b, shape );

            const auto& output = residual.forward( first, second );
            residual.synchronize();

            const std::vector<double> stored_a = asStored<kPrecision>( std::span<const float>( a ) );
            const std::vector<double> stored_b = asStored<kPrecision>( std::span<const float>( b ) );

            std::vector<double> reference( count );

            for ( std::size_t i = 0; i < count; ++i )
            {
                reference[i] = stored_a[i] + stored_b[i];
            }

            EXPECT_LE( relativeError( readFloats<kPrecision>( output ), reference ), kBar<kPrecision> );
        }
    }

}
