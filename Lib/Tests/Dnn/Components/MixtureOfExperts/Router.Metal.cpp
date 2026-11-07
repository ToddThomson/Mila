/**
 * @file Router.Metal.cpp
 * @brief Mixture-of-experts selection on Metal, in FP16 and BF16, against a double-precision reference on the logits as stored.
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
        template<TensorDataType TPrecision>
        using RouterOp = typename OperationTraits<OperationType::RouterOp, DeviceType::Metal, TPrecision>::type;

        using IndexTensor = Tensor<TensorDataType::INT32, MetalMemoryResource>;

        struct Routed
        {
            std::vector<float> weights;
            std::vector<std::int32_t> indices;
        };

        template<TensorDataType TPrecision>
        Routed route( std::span<const float> logits, std::span<const float> per_expert_scale, dim_t rows, dim_t experts, dim_t top_k )
        {
            const auto context = createExecutionContext( Device::Metal( 0 ) );

            RouterOp<TPrecision> op( context.get(), RouterConfig( 16, experts, top_k ) );

            auto scale = metalTensor<TPrecision>( per_expert_scale, shape_t{ experts } );
            auto input = metalTensor<TPrecision>( logits, shape_t{ rows, experts } );
            Tensor<TPrecision, MetalMemoryResource> weights( Device::Metal( 0 ), shape_t{ rows, top_k } );
            IndexTensor indices( Device::Metal( 0 ), shape_t{ rows, top_k } );

            op.setParameters( &scale, nullptr );
            op.build( BuildContext( shape_t{ rows, experts }, RuntimeMode::Inference, false ) );
            op.forward( input, weights, indices );
            context->synchronize();

            Tensor<TensorDataType::INT32, CpuMemoryResource> host_indices( Device::Cpu(), indices.shape() );
            copy( indices, host_indices );

            return { readFloats<TPrecision>( weights ), std::vector<std::int32_t>( host_indices.data(), host_indices.data() + host_indices.size() ) };
        }

        /// The reference's choice per row: the top_k by stored logit, the lower index on a tie, with their renormalized softmax times the scale.
        Routed reference( std::span<const double> logits, std::span<const double> per_expert_scale, dim_t rows, dim_t experts, dim_t top_k )
        {
            Routed result;

            for ( dim_t row = 0; row < rows; ++row )
            {
                std::vector<dim_t> order( static_cast<std::size_t>( experts ) );
                std::iota( order.begin(), order.end(), dim_t{ 0 } );

                const auto value = [&]( dim_t expert ) { return logits[static_cast<std::size_t>( row * experts + expert )]; };

                std::ranges::stable_sort( order, [&]( dim_t a, dim_t b ) { return value( a ) > value( b ); } );

                double mass = 0.0;

                for ( dim_t slot = 0; slot < top_k; ++slot )
                {
                    mass += std::exp( value( order[static_cast<std::size_t>( slot )] ) - value( order[0] ) );
                }

                for ( dim_t slot = 0; slot < top_k; ++slot )
                {
                    const dim_t expert = order[static_cast<std::size_t>( slot )];

                    result.indices.push_back( static_cast<std::int32_t>( expert ) );
                    result.weights.push_back( static_cast<float>(
                        std::exp( value( expert ) - value( order[0] ) ) / mass * per_expert_scale[static_cast<std::size_t>( expert )] ) );
                }
            }

            return result;
        }

        template<typename TPrecisionConstant>
        class RouterMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( RouterMetal, Precisions, PrecisionNames );

        template<TensorDataType TPrecision>
        void expectMatchesTheReference( dim_t rows, dim_t experts, dim_t top_k, std::uint32_t seed )
        {
            const std::vector<float> logits = randomValues( static_cast<std::size_t>( rows * experts ), seed, 2.0f );
            const std::vector<float> scale = randomValues( static_cast<std::size_t>( experts ), seed + 1, 1.0f );

            const Routed actual = route<TPrecision>( logits, scale, rows, experts, top_k );
            const Routed expected = reference( asStored<TPrecision>( std::span<const float>( logits ) ),
                asStored<TPrecision>( std::span<const float>( scale ) ), rows, experts, top_k );

            EXPECT_EQ( actual.indices, expected.indices );

            const std::vector<double> expected_weights( expected.weights.begin(), expected.weights.end() );

            EXPECT_LE( relativeError( actual.weights, expected_weights ), kBar<TPrecision> );
        }
    }

    TYPED_TEST( RouterMetal, MatchesTheReferenceAtGemmasRouting )
    {
        // Gemma 4 26B: the top 8 of 128 experts, a prefill's rows and a decode's.
        expectMatchesTheReference<TypeParam::value>( 37, 128, 8, 11 );
        expectMatchesTheReference<TypeParam::value>( 1, 128, 8, 12 );
    }

    TYPED_TEST( RouterMetal, MatchesTheReferenceOnOtherShapes )
    {
        expectMatchesTheReference<TypeParam::value>( 5, 4, 2, 13 );
        expectMatchesTheReference<TypeParam::value>( 9, 33, 16, 14 );
        expectMatchesTheReference<TypeParam::value>( 3, 7, 1, 15 );
    }

    TYPED_TEST( RouterMetal, AnExactTieSelectsTheLowerIndex )
    {
        const std::array<float, 4> logits{ 2.0f, 1.0f, 1.0f, -1.0f };
        const std::array<float, 4> scale{ 1.0f, 0.5f, 2.0f, 1.0f };

        const Routed routed = route<TypeParam::value>( logits, scale, 1, 4, 2 );

        EXPECT_EQ( routed.indices, ( std::vector<std::int32_t>{ 0, 1 } ) );
    }

    TYPED_TEST( RouterMetal, RefusesABiasAndATopKPastTheKernel )
    {
        const auto context = createExecutionContext( Device::Metal( 0 ) );

        EXPECT_THROW( RouterOp<TypeParam::value>( context.get(), RouterConfig( 16, 64, 17 ) ), std::invalid_argument );

        RouterOp<TypeParam::value> op( context.get(), RouterConfig( 16, 4, 2 ) );
        Tensor<TypeParam::value, MetalMemoryResource> bias( Device::Metal( 0 ), shape_t{ 4 } );

        EXPECT_THROW( op.setParameters( &bias, &bias ), std::invalid_argument );
    }

    TEST( RouterMetalRows, OnlyFp16AndBf16Exist )
    {
        EXPECT_FALSE( ( OperationSupported<OperationType::RouterOp, DeviceType::Metal, TensorDataType::FP32> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::RouterOp, DeviceType::Metal, TensorDataType::FP16> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::RouterOp, DeviceType::Metal, TensorDataType::BF16> ) );
    }
}
