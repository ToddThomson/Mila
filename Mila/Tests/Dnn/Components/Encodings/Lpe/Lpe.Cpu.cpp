/**
 * @file Lpe.Cpu.cpp
 * @brief Concrete-component tests for Lpe<DeviceType::Cpu, INT32, FP32>.
 *
 * Lpe is the GPT-2 learned positional encoder: a leaf WITH parameters (wte token
 * table + wpe position table). Forward is a token gather plus a per-position add:
 *   output[b, t, :] = wte[ X[b, t], : ] + wpe[ t, : ]
 *
 * CPU is FP32-only (the CPU LpeOp is INT32 -> FP32). This file also carries the
 * FP32 backward oracle: the gradient is a scatter-add of the upstream gradient
 * into wte (by token id) and wpe (by position).
 *
 * Supersedes the pre-methodology Encoder.Cpu.cpp (retired in place, Section 3).
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::Encodings::Lpe
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        constexpr int64_t kVocab = 8;
        constexpr int64_t kMaxSeq = 8;
        constexpr int64_t kEmbed = 4;

        using LpeCpu = Mila::Dnn::Lpe<DeviceType::Cpu, TensorDataType::INT32, TensorDataType::FP32>;
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using TensorIndex = Tensor<TensorDataType::INT32, CpuMemoryResource>;

        static_assert( LpeCpu::getDeviceType() == DeviceType::Cpu );
        static_assert( LpeCpu::getPrecision() == TensorDataType::FP32 );

        float wteValue( int64_t v, int64_t c )
        {
            return 0.1f * static_cast<float>( v ) + 0.01f * static_cast<float>( c );
        }

        float wpeValue( int64_t p, int64_t c )
        {
            return -0.05f * static_cast<float>( p ) + 0.2f * static_cast<float>( c );
        }
    }

    class LpeCpuTests : public ::testing::Test
    {
    protected:
        LpeConfig config()
        {
            return LpeConfig()
                .withEmbeddingDim( kEmbed )
                .withMaxSequenceLength( kMaxSeq )
                .withVocabularyLength( kVocab );
        }

        std::unique_ptr<LpeCpu> builtLpe( const shape_t& token_shape, RuntimeMode mode )
        {
            auto lpe = std::make_unique<LpeCpu>( "lpe", config(), Device::Cpu() );
            lpe->build( BuildContext( token_shape, mode, false ) );

            return lpe;
        }

        // params order is { wte, wpe }; both are FP32 CPU tensors.
        void setKnownTables( LpeCpu& lpe )
        {
            auto params = lpe.getParameters();
            auto* wte = static_cast<TensorFp32*>( params[ 0 ] );
            auto* wpe = static_cast<TensorFp32*>( params[ 1 ] );

            for ( int64_t v = 0; v < kVocab; ++v )
            {
                for ( int64_t c = 0; c < kEmbed; ++c )
                {
                    wte->data()[ v * kEmbed + c ] = wteValue( v, c );
                }
            }

            for ( int64_t p = 0; p < kMaxSeq; ++p )
            {
                for ( int64_t c = 0; c < kEmbed; ++c )
                {
                    wpe->data()[ p * kEmbed + c ] = wpeValue( p, c );
                }
            }
        }

        TensorIndex tokens( const shape_t& shape )
        {
            TensorIndex t( Device::Cpu(), shape );
            for ( dim_t i = 0; i < t.size(); ++i )
            {
                // Deliberate repeats across batch so wte gradient accumulation is exercised.
                t.data()[ i ] = static_cast<int32_t>( i % 3 );
            }

            return t;
        }
    };

    // ====================================================================
    // A. Construction
    // ====================================================================

    TEST_F( LpeCpuTests, Construct_StandaloneSucceeds )
    {
        LpeCpu lpe( "lpe", config(), Device::Cpu() );

        EXPECT_EQ( lpe.getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( LpeCpuTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW( LpeCpu( "lpe", config(), Device::Cuda( 0 ) ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle (preconditions)
    // ====================================================================

    TEST_F( LpeCpuTests, Forward_ThrowsBeforeBuild )
    {
        LpeCpu lpe( "lpe", config(), Device::Cpu() );
        TensorIndex input( Device::Cpu(), shape_t{ 2, 3 } );

        EXPECT_THROW( lpe.forward( input ), std::runtime_error );
    }

    TEST_F( LpeCpuTests, Build_ThrowsOnNonRank2Input )
    {
        LpeCpu lpe( "lpe", config(), Device::Cpu() );

        EXPECT_THROW(
            lpe.build( BuildContext( shape_t{ 2, 3, kEmbed }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    TEST_F( LpeCpuTests, Build_ThrowsWhenSeqExceedsMax )
    {
        LpeCpu lpe( "lpe", config(), Device::Cpu() );

        EXPECT_THROW(
            lpe.build( BuildContext( shape_t{ 2, kMaxSeq + 1 }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( LpeCpuTests, Forward_TokenEmbeddingPlusPositional )
    {
        const shape_t token_shape{ 2, 3 };
        const int64_t B = 2;
        const int64_t T = 3;

        auto lpe = builtLpe( token_shape, RuntimeMode::Inference );
        setKnownTables( *lpe );

        auto input = tokens( token_shape );
        auto& output = lpe->forward( input );

        ASSERT_EQ( output.shape(), ( shape_t{ B, T, kEmbed } ) );

        for ( int64_t b = 0; b < B; ++b )
        {
            for ( int64_t t = 0; t < T; ++t )
            {
                const int32_t idx = input.data()[ b * T + t ];
                for ( int64_t c = 0; c < kEmbed; ++c )
                {
                    const float expected = wteValue( idx, c ) + wpeValue( t, c );
                    const float actual = output.data()[ ( b * T + t ) * kEmbed + c ];

                    EXPECT_NEAR( actual, expected, 1e-5f )
                        << "mismatch at (b,t,c)=(" << b << "," << t << "," << c << ")";
                }
            }
        }
    }

    // ====================================================================
    // D/F. Backward (training mode + gradient scatter)
    // ====================================================================

    TEST_F( LpeCpuTests, Backward_ThrowsWhenBuiltForInference )
    {
        const shape_t token_shape{ 2, 3 };
        auto lpe = builtLpe( token_shape, RuntimeMode::Inference );
        setKnownTables( *lpe );

        auto input = tokens( token_shape );
        lpe->forward( input );

        TensorFp32 output_grad( Device::Cpu(), shape_t{ 2, 3, kEmbed } );

        EXPECT_THROW( lpe->backward( input, output_grad ), std::runtime_error );
    }

    TEST_F( LpeCpuTests, Backward_ScattersGradientIntoTables )
    {
        const shape_t token_shape{ 2, 3 };
        const int64_t B = 2;
        const int64_t T = 3;

        auto lpe = builtLpe( token_shape, RuntimeMode::Training );
        setKnownTables( *lpe );

        auto input = tokens( token_shape );

        TensorFp32 output_grad( Device::Cpu(), shape_t{ B, T, kEmbed } );
        for ( dim_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = 0.01f * static_cast<float>( i % 7 ) + 0.1f;
        }

        lpe->forward( input );
        lpe->backward( input, output_grad );

        // Reference: dWte[v,c] = sum over (b,t) with X[b,t]==v of og[b,t,c];
        //            dWpe[p,c] = sum over b of og[b,p,c].
        std::vector<float> expected_wte( static_cast<size_t>( kVocab * kEmbed ), 0.0f );
        std::vector<float> expected_wpe( static_cast<size_t>( kMaxSeq * kEmbed ), 0.0f );

        for ( int64_t b = 0; b < B; ++b )
        {
            for ( int64_t t = 0; t < T; ++t )
            {
                const int32_t idx = input.data()[ b * T + t ];
                for ( int64_t c = 0; c < kEmbed; ++c )
                {
                    const float g = output_grad.data()[ ( b * T + t ) * kEmbed + c ];
                    expected_wte[ idx * kEmbed + c ] += g;
                    expected_wpe[ t * kEmbed + c ] += g;
                }
            }
        }

        auto* wte_grad = lpe->getWteGrad();
        auto* wpe_grad = lpe->getWpeGrad();
        ASSERT_NE( wte_grad, nullptr );
        ASSERT_NE( wpe_grad, nullptr );

        for ( size_t i = 0; i < expected_wte.size(); ++i )
        {
            EXPECT_NEAR( wte_grad->data()[ i ], expected_wte[ i ], 1e-5f ) << "wte_grad mismatch at " << i;
        }

        for ( size_t i = 0; i < expected_wpe.size(); ++i )
        {
            EXPECT_NEAR( wpe_grad->data()[ i ], expected_wpe[ i ], 1e-5f ) << "wpe_grad mismatch at " << i;
        }
    }

    // ====================================================================
    // G. Parameters & Gradients
    // ====================================================================

    TEST_F( LpeCpuTests, Parameters_WteAndWpe )
    {
        auto lpe = builtLpe( shape_t{ 2, 3 }, RuntimeMode::Inference );

        EXPECT_EQ( lpe->getParameters().size(), 2u );
        EXPECT_EQ( lpe->parameterCount(), static_cast<size_t>( kVocab * kEmbed + kMaxSeq * kEmbed ) );
    }

    TEST_F( LpeCpuTests, Gradients_PresentOnlyForTrainingBuild )
    {
        auto inference = builtLpe( shape_t{ 2, 3 }, RuntimeMode::Inference );
        EXPECT_TRUE( inference->getGradients().empty() );

        auto training = builtLpe( shape_t{ 2, 3 }, RuntimeMode::Training );
        EXPECT_EQ( training->getGradients().size(), 2u );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TEST_F( LpeCpuTests, GetType_IsLpe )
    {
        LpeCpu lpe( "lpe", config(), Device::Cpu() );

        EXPECT_EQ( lpe.getType(), ComponentType::Lpe );
    }
}
