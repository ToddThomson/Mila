/**
 * @file Softmax.Cpu.cpp
 * @brief Concrete-component tests for Softmax<DeviceType::Cpu, FP32>.
 *
 * Concrete-component archetype for a stateless activation leaf (see
 * Specifications/Testing.md). Softmax over the configured axis (default -1):
 *   y_i = exp(x_i - max) / sum_j exp(x_j - max)
 * Backward: dX_i = Y_i * (dY_i - sum_j Y_j * dY_j).
 *
 * CPU is FP32-only (CpuSoftmaxOp). Re-greened from the pre-methodology version.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::Normalization
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using SoftmaxCpu = Mila::Dnn::Softmax<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static_assert( SoftmaxCpu::getDeviceType() == DeviceType::Cpu );
        static_assert( SoftmaxCpu::getPrecision() == TensorDataType::FP32 );

        constexpr int64_t kRows = 4;
        constexpr int64_t kChannels = 8;

        // Row-wise softmax over the trailing dimension.
        void referenceSoftmax( const float* X, int64_t rows, int64_t channels, std::vector<float>& Y )
        {
            Y.assign( static_cast<size_t>( rows * channels ), 0.0f );

            for ( int64_t r = 0; r < rows; ++r )
            {
                const float* x = X + r * channels;

                float maxv = -INFINITY;
                for ( int64_t i = 0; i < channels; ++i )
                {
                    maxv = std::max( maxv, x[ i ] );
                }

                double sum = 0.0;
                for ( int64_t i = 0; i < channels; ++i )
                {
                    sum += std::exp( static_cast<double>( x[ i ] - maxv ) );
                }

                for ( int64_t i = 0; i < channels; ++i )
                {
                    Y[ r * channels + i ] = static_cast<float>( std::exp( static_cast<double>( x[ i ] - maxv ) ) / sum );
                }
            }
        }

        void fillSpread( TensorFp32& t )
        {
            for ( size_t i = 0; i < t.size(); ++i )
            {
                t.data()[ i ] = static_cast<float>( i ) / t.size() * 4.0f - 2.0f;
            }
        }
    }

    class SoftmaxCpuTests : public ::testing::Test
    {
    protected:
        std::unique_ptr<SoftmaxCpu> builtSoftmax( const shape_t& shape, RuntimeMode mode, int64_t axis = -1 )
        {
            auto softmax = std::make_unique<SoftmaxCpu>( "softmax", SoftmaxConfig().withAxis( axis ), Device::Cpu() );
            softmax->build( BuildContext( shape, mode, false ) );

            return softmax;
        }
    };

    // ====================================================================
    // A. Construction
    // ====================================================================

    TEST_F( SoftmaxCpuTests, Construct_StandaloneSucceeds )
    {
        SoftmaxCpu softmax( "softmax", SoftmaxConfig(), Device::Cpu() );

        EXPECT_EQ( softmax.getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( SoftmaxCpuTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW( SoftmaxCpu( "softmax", SoftmaxConfig(), Device::Cuda( 0 ) ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle (preconditions)
    // ====================================================================

    TEST_F( SoftmaxCpuTests, Forward_ThrowsBeforeBuild )
    {
        SoftmaxCpu softmax( "softmax", SoftmaxConfig(), Device::Cpu() );
        TensorFp32 input( Device::Cpu(), shape_t{ kRows, kChannels } );
        TensorFp32 output( Device::Cpu(), shape_t{ kRows, kChannels } );

        EXPECT_THROW( softmax.forward( input, output ), std::runtime_error );
    }

    TEST_F( SoftmaxCpuTests, Build_ThrowsOnAxisOutOfBounds )
    {
        SoftmaxCpu softmax( "softmax", SoftmaxConfig().withAxis( 5 ), Device::Cpu() );

        // Axis 5 is out of bounds for a rank-2 input.
        EXPECT_THROW(
            softmax.build( BuildContext( shape_t{ kRows, kChannels }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( SoftmaxCpuTests, Forward_MatchesReference )
    {
        const shape_t shape{ kRows, kChannels };
        auto softmax = builtSoftmax( shape, RuntimeMode::Inference );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output( Device::Cpu(), shape );
        fillSpread( input );

        softmax->forward( input, output );

        std::vector<float> expected;
        referenceSoftmax( input.data(), kRows, kChannels, expected );

        ASSERT_EQ( output.size(), expected.size() );

        for ( size_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], expected[ i ], 1e-5f ) << "forward mismatch at index " << i;
        }

        // Each row must sum to 1.
        for ( int64_t r = 0; r < kRows; ++r )
        {
            double row_sum = 0.0;
            for ( int64_t c = 0; c < kChannels; ++c )
            {
                row_sum += output.data()[ r * kChannels + c ];
            }

            EXPECT_NEAR( row_sum, 1.0, 1e-5 ) << "row " << r << " does not sum to 1";
        }
    }

    // ====================================================================
    // D/F. Backward (training mode + numeric gradient)
    // ====================================================================

    TEST_F( SoftmaxCpuTests, Backward_ThrowsWhenBuiltForInference )
    {
        const shape_t shape{ kRows, kChannels };
        auto softmax = builtSoftmax( shape, RuntimeMode::Inference );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape );
        TensorFp32 input_grad( Device::Cpu(), shape );
        fillSpread( input );

        EXPECT_THROW( softmax->backward( input, output_grad, input_grad ), std::runtime_error );
    }

    TEST_F( SoftmaxCpuTests, Backward_MatchesReferenceGradient )
    {
        const shape_t shape{ kRows, kChannels };
        auto softmax = builtSoftmax( shape, RuntimeMode::Training );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape );
        TensorFp32 input_grad( Device::Cpu(), shape );
        fillSpread( input );
        for ( size_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = 0.1f * static_cast<float>( ( i % 7 ) + 1 );
        }

        softmax->backward( input, output_grad, input_grad );

        std::vector<float> y;
        referenceSoftmax( input.data(), kRows, kChannels, y );

        // dX_i = Y_i * (dY_i - sum_j Y_j * dY_j), per row.
        std::vector<float> expected_dx( static_cast<size_t>( kRows * kChannels ), 0.0f );
        for ( int64_t r = 0; r < kRows; ++r )
        {
            double dot = 0.0;
            for ( int64_t c = 0; c < kChannels; ++c )
            {
                dot += static_cast<double>( y[ r * kChannels + c ] ) * output_grad.data()[ r * kChannels + c ];
            }

            for ( int64_t c = 0; c < kChannels; ++c )
            {
                const double yi = y[ r * kChannels + c ];
                const double dyi = output_grad.data()[ r * kChannels + c ];
                expected_dx[ r * kChannels + c ] = static_cast<float>( yi * ( dyi - dot ) );
            }
        }

        ASSERT_EQ( input_grad.size(), expected_dx.size() );
        for ( size_t i = 0; i < input_grad.size(); ++i )
        {
            EXPECT_NEAR( input_grad.data()[ i ], expected_dx[ i ], 1e-5f ) << "dX mismatch at index " << i;
        }
    }

    // ====================================================================
    // G. Parameters (stateless) & J. Type
    // ====================================================================

    TEST_F( SoftmaxCpuTests, Parameters_AreEmpty )
    {
        auto softmax = builtSoftmax( shape_t{ kRows, kChannels }, RuntimeMode::Inference );

        EXPECT_EQ( softmax->parameterCount(), 0u );
        EXPECT_TRUE( softmax->getParameters().empty() );
        EXPECT_TRUE( softmax->getGradients().empty() );
    }

    TEST_F( SoftmaxCpuTests, GetType_IsSoftmax )
    {
        SoftmaxCpu softmax( "softmax", SoftmaxConfig(), Device::Cpu() );

        EXPECT_EQ( softmax.getType(), ComponentType::Softmax );
    }
}
