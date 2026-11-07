/**
 * @file Activation.Metal.cpp
 * @brief Every elementwise activation on Metal, against a double-precision reference of what its functor computes.
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
        class ActivationMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( ActivationMetal, Precisions, PrecisionNames );

        constexpr double kLeakyReluAlpha = 0.01;

        double reference( ActivationType activation, double x )
        {
            const double sigmoid = 1.0 / ( 1.0 + std::exp( -x ) );

            switch ( activation )
            {
                case ActivationType::None:
                    return x;
                case ActivationType::Gelu:
                    return 0.5 * x * ( 1.0 + std::tanh( std::sqrt( 2.0 / std::numbers::pi ) * ( x + 0.044715 * x * x * x ) ) );
                case ActivationType::Silu:
                    return x * sigmoid;
                case ActivationType::Relu:
                    return x > 0.0 ? x : 0.0;
                case ActivationType::Tanh:
                    return std::tanh( x );
                case ActivationType::Sigmoid:
                    return sigmoid;
                case ActivationType::LeakyRelu:
                    return x > 0.0 ? x : kLeakyReluAlpha * x;
                case ActivationType::Mish:
                    return x * std::tanh( x > 20.0 ? x : std::log1p( std::exp( x ) ) );
                default:
                    throw std::logic_error( "no reference for this activation" );
            }
        }

        template<TensorDataType TPrecision, ActivationType TActivation>
        double errorOf( const shape_t& shape, float sigma, std::uint32_t seed )
        {
            Activation<DeviceType::Metal, TPrecision, TActivation> activation( "activation", ActivationConfig( TActivation ), Device::Metal( 0 ) );
            activation.build( BuildContext( shape, RuntimeMode::Inference, false ) );

            dim_t elements = 1;

            for ( const dim_t axis : shape )
            {
                elements *= axis;
            }

            const std::vector<float> values = randomValues( static_cast<std::size_t>( elements ), seed, sigma );
            auto input = metalTensor<TPrecision>( values, shape );

            const auto& output = activation.forward( input );
            activation.synchronize();

            const std::vector<double> x = asStored<TPrecision>( std::span<const float>( values ) );
            std::vector<double> expected( x.size() );
            std::ranges::transform( x, expected.begin(), []( double value ) { return reference( TActivation, value ); } );

            return relativeError( readFloats<TPrecision>( output ), expected );
        }

        template<TensorDataType TPrecision, ActivationType TActivation>
        void expectEachShape()
        {
            const std::string name( activationTypeToString( TActivation ) );

            EXPECT_LE( ( errorOf<TPrecision, TActivation>( shape_t{ 1, 1, 64 }, 1.0f, 3 ) ), kBar<TPrecision> ) << name << ", a decode row";
            EXPECT_LE( ( errorOf<TPrecision, TActivation>( shape_t{ 2, 7, 33 }, 1.0f, 5 ) ), kBar<TPrecision> ) << name << ", an odd shape";

            // Qwen's DeltaNet convolution output, 10240 channels, over a prefill's rows.
            EXPECT_LE( ( errorOf<TPrecision, TActivation>( shape_t{ 1, 16, 10240 }, 2.0f, 7 ) ), kBar<TPrecision> ) << name << ", Qwen's width";

            // Large either way: past every saturation, and past softplus's cutover at 20.
            EXPECT_LE( ( errorOf<TPrecision, TActivation>( shape_t{ 1, 4, 256 }, 30.0f, 11 ) ), kBar<TPrecision> ) << name << ", large inputs";
        }
    }

    TYPED_TEST( ActivationMetal, EveryFunctorMatchesItsReference )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        expectEachShape<kPrecision, ActivationType::None>();
        expectEachShape<kPrecision, ActivationType::Gelu>();
        expectEachShape<kPrecision, ActivationType::Silu>();
        expectEachShape<kPrecision, ActivationType::Relu>();
        expectEachShape<kPrecision, ActivationType::Tanh>();
        expectEachShape<kPrecision, ActivationType::Sigmoid>();
        expectEachShape<kPrecision, ActivationType::LeakyRelu>();
        expectEachShape<kPrecision, ActivationType::Mish>();
    }
}
