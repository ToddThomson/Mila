/**
 * @file MultiHeadAttention.Cpu.cpp
 * @brief Concrete-component tests for MultiHeadAttention<DeviceType::Cpu, FP32>.
 *
 * Concrete-component archetype for a stateless attention leaf (see
 * Specifications/Testing.md). MHA consumes a concatenated-QKV tensor
 * [B, T, 3*model_dim] and produces [B, T, model_dim] via causal self-attention.
 *
 * Layout (matches CpuAttentionOp): for token (b, t) at base = (b*T+t)*3C,
 *   Q[b,t,h,k] = X[base + 0*C + h*HS + k]
 *   K[b,t,h,k] = X[base + 1*C + h*HS + k]
 *   V[b,t,h,k] = X[base + 2*C + h*HS + k]
 * scores = scale * Q.K^T (scale = 1/sqrt(HS)), causal (key j <= query i),
 * softmax over keys, output = att.V, written as out[(b*T+i)*C + h*HS + k].
 *
 * CPU is FP32-only (CpuAttentionOp). The forward reference is an independent
 * implementation of the same spec, so a match validates both.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::Attention::MHA
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        constexpr int64_t kModelDim = 8;   // C
        constexpr int64_t kNumHeads = 2;   // NH
        constexpr int64_t kHeadDim = kModelDim / kNumHeads;  // HS = 4

        using MhaCpu = Mila::Dnn::MultiHeadAttention<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static_assert( MhaCpu::getDeviceType() == DeviceType::Cpu );
        static_assert( MhaCpu::getPrecision() == TensorDataType::FP32 );

        // Causal multi-head self-attention from concatenated QKV [B, T, 3C] -> [B, T, C].
        void referenceAttention(
            const float* X, int64_t B, int64_t T, int64_t C, int64_t NH, std::vector<float>& Y )
        {
            const int64_t HS = C / NH;
            const int64_t qkv = 3 * C;
            const float scale = 1.0f / std::sqrt( static_cast<float>( HS ) );
            Y.assign( static_cast<size_t>( B * T * C ), 0.0f );

            std::vector<float> scores( static_cast<size_t>( T ) );

            for ( int64_t b = 0; b < B; ++b )
            {
                for ( int64_t h = 0; h < NH; ++h )
                {
                    for ( int64_t i = 0; i < T; ++i )
                    {
                        const float* q = X + ( b * T + i ) * qkv + 0 * C + h * HS;

                        float maxv = -INFINITY;
                        for ( int64_t j = 0; j <= i; ++j )
                        {
                            const float* k = X + ( b * T + j ) * qkv + 1 * C + h * HS;
                            float s = 0.0f;
                            for ( int64_t d = 0; d < HS; ++d )
                            {
                                s += q[ d ] * k[ d ];
                            }
                            s *= scale;
                            scores[ j ] = s;
                            maxv = std::max( maxv, s );
                        }

                        double sum = 0.0;
                        for ( int64_t j = 0; j <= i; ++j )
                        {
                            scores[ j ] = std::exp( scores[ j ] - maxv );
                            sum += scores[ j ];
                        }

                        for ( int64_t d = 0; d < HS; ++d )
                        {
                            double acc = 0.0;
                            for ( int64_t j = 0; j <= i; ++j )
                            {
                                const float* v = X + ( b * T + j ) * qkv + 2 * C + h * HS;
                                acc += ( scores[ j ] / sum ) * v[ d ];
                            }
                            Y[ ( b * T + i ) * C + h * HS + d ] = static_cast<float>( acc );
                        }
                    }
                }
            }
        }

        void fillSpread( TensorFp32& t, float phase )
        {
            for ( size_t i = 0; i < t.size(); ++i )
            {
                t.data()[ i ] = std::sin( 0.2f * static_cast<float>( i ) + phase );
            }
        }
    }

    class MhaCpuTests : public ::testing::Test
    {
    protected:
        static MultiHeadAttentionConfig config()
        {
            return MultiHeadAttentionConfig( kModelDim, kNumHeads );
        }

        std::unique_ptr<MhaCpu> builtMha( const shape_t& shape, RuntimeMode mode )
        {
            auto mha = std::make_unique<MhaCpu>( "mha", config(), Device::Cpu() );
            mha->build( BuildContext( shape, mode, false ) );

            return mha;
        }
    };

    // ====================================================================
    // A. Construction
    // ====================================================================

    TEST_F( MhaCpuTests, Construct_StandaloneSucceeds )
    {
        MhaCpu mha( "mha", config(), Device::Cpu() );

        EXPECT_EQ( mha.getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( MhaCpuTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW( MhaCpu( "mha", config(), Device::Cuda( 0 ) ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle (preconditions)
    // ====================================================================

    TEST_F( MhaCpuTests, Forward_ThrowsBeforeBuild )
    {
        MhaCpu mha( "mha", config(), Device::Cpu() );
        TensorFp32 input( Device::Cpu(), shape_t{ 2, 3, 3 * kModelDim } );

        EXPECT_THROW( mha.forward( input ), std::runtime_error );
    }

    TEST_F( MhaCpuTests, Build_ThrowsOnWrongTrailingDim )
    {
        MhaCpu mha( "mha", config(), Device::Cpu() );

        // Trailing must be 3 * model_dim; model_dim alone is wrong.
        EXPECT_THROW(
            mha.build( BuildContext( shape_t{ 2, 3, kModelDim }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    TEST_F( MhaCpuTests, Build_ThrowsOnNonRank3 )
    {
        MhaCpu mha( "mha", config(), Device::Cpu() );

        EXPECT_THROW(
            mha.build( BuildContext( shape_t{ 2, 3 * kModelDim }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // E. Forward (numeric vs causal-attention reference)
    // ====================================================================

    TEST_F( MhaCpuTests, Forward_MatchesCausalReference )
    {
        const int64_t B = 2;
        const int64_t T = 3;
        const shape_t shape{ B, T, 3 * kModelDim };

        auto mha = builtMha( shape, RuntimeMode::Inference );

        TensorFp32 input( Device::Cpu(), shape );
        fillSpread( input, 0.0f );

        auto& output = mha->forward( input );

        std::vector<float> expected;
        referenceAttention( input.data(), B, T, kModelDim, kNumHeads, expected );

        ASSERT_EQ( output.shape(), ( shape_t{ B, T, kModelDim } ) );
        ASSERT_EQ( output.size(), expected.size() );

        for ( size_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], expected[ i ], 1e-4f ) << "attention mismatch at index " << i;
        }
    }

    // ====================================================================
    // D/F. Backward (training mode + input-gradient shape)
    // ====================================================================

    TEST_F( MhaCpuTests, Backward_ThrowsWhenBuiltForInference )
    {
        const shape_t shape{ 2, 3, 3 * kModelDim };
        auto mha = builtMha( shape, RuntimeMode::Inference );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape_t{ 2, 3, kModelDim } );
        fillSpread( input, 0.0f );

        mha->forward( input );

        EXPECT_THROW( mha->backward( input, output_grad ), std::runtime_error );
    }

    TEST_F( MhaCpuTests, Backward_ReturnsInputShapedGradient )
    {
        const shape_t shape{ 2, 3, 3 * kModelDim };
        auto mha = builtMha( shape, RuntimeMode::Training );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape_t{ 2, 3, kModelDim } );
        fillSpread( input, 0.0f );
        fillSpread( output_grad, 1.3f );

        mha->forward( input );
        auto& input_grad = mha->backward( input, output_grad );

        ASSERT_EQ( input_grad.shape(), shape );
        for ( size_t i = 0; i < input_grad.size(); ++i )
        {
            EXPECT_TRUE( std::isfinite( input_grad.data()[ i ] ) ) << "non-finite dX at index " << i;
        }
    }

    // ====================================================================
    // G. Parameters (stateless) & J. Type
    // ====================================================================

    TEST_F( MhaCpuTests, Parameters_AreEmpty )
    {
        auto mha = builtMha( shape_t{ 2, 3, 3 * kModelDim }, RuntimeMode::Inference );

        EXPECT_EQ( mha->parameterCount(), 0u );
        EXPECT_TRUE( mha->getParameters().empty() );
        EXPECT_TRUE( mha->getGradients().empty() );
    }

    TEST_F( MhaCpuTests, Accessors_ReturnConfiguredDims )
    {
        MhaCpu mha( "mha", config(), Device::Cpu() );

        EXPECT_EQ( mha.getModelDim(), kModelDim );
        EXPECT_EQ( mha.getNumHeads(), kNumHeads );
    }

    TEST_F( MhaCpuTests, GetType_IsMultiHeadAttention )
    {
        MhaCpu mha( "mha", config(), Device::Cpu() );

        EXPECT_EQ( mha.getType(), ComponentType::MultiHeadAttention );
    }
}
