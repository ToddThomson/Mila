/**
 * @file SequenceLogLikelihood.Cuda.cpp
 * @brief The device next-token log-probability against its host oracle, Dnn::nextTokenLogProbability.
 *
 * Random logits at real vocabulary widths, FP32 and BF16, with and without Gemma's softcap; every row the
 * device writes must equal the host reduction of the same values within FP32 storage of the result.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <random>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Core
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        // The device sums in a different order and stores FP32: about 1e-7 relative of a log-probability
        // near -20, where these rows land. Written before the first run.
        constexpr double kTolerance = 1e-5;

        template<TensorDataType TPrecision>
        using OpType = typename OperationTraits<OperationType::NextTokenLogProbabilityOp, DeviceType::Cuda, TPrecision>::type;

        struct Case
        {
            dim_t vocab;
            dim_t rows;
            float softcap;
        };

        /// Scores `rows` rows of random logits at `first_position` on the device and compares each with the host.
        template<TensorDataType TPrecision>
        void expectDeviceMatchesHost( IExecutionContext* context, const Case& shape, dim_t first_position )
        {
            std::mt19937 rng( 20260928 );
            std::normal_distribution<float> logit( 0.0f, 4.0f );
            std::uniform_int_distribution<std::int32_t> token( 0, static_cast<std::int32_t>( shape.vocab - 1 ) );

            const dim_t sequence_length = first_position + shape.rows + 1;

            Tensor<TensorDataType::FP32, CpuMemoryResource> host_logits( Device::Cpu(), shape_t{ 1, shape.rows, shape.vocab } );

            for ( dim_t index = 0; index < shape.rows * shape.vocab; ++index )
            {
                host_logits.data()[ index ] = logit( rng );
            }

            Tensor<TensorDataType::INT32, CpuMemoryResource> host_tokens( Device::Cpu(), shape_t{ 1, sequence_length } );

            for ( dim_t position = 0; position < sequence_length; ++position )
            {
                host_tokens.data()[ position ] = token( rng );
            }

            // One target is each row's own maximum, where the log-probability is nearest zero.
            host_tokens.data()[ first_position + 1 ] = 0;
            host_logits.data()[ 0 ] = 30.0f;

            Tensor<TPrecision, CudaDeviceMemoryResource> device_logits( Device::Cuda( 0 ), host_logits.shape() );
            Tensor<TensorDataType::INT32, CudaDeviceMemoryResource> device_tokens( Device::Cuda( 0 ), host_tokens.shape() );

            copy( host_logits, device_logits, context );
            copy( host_tokens, device_tokens, context );

            // The host reads the values the device holds: BF16 rounds them.
            Tensor<TensorDataType::FP32, CpuMemoryResource> stored_logits( Device::Cpu(), host_logits.shape() );
            copy( device_logits, stored_logits, context );
            context->synchronize();

            OpType<TPrecision> op( context, shape.softcap );
            op.begin( sequence_length - 1 );
            op.forward( device_logits, device_tokens, first_position, shape.rows );
            context->synchronize();

            const auto device = op.logProbabilities( sequence_length - 1 );

            for ( dim_t row = 0; row < shape.rows; ++row )
            {
                const double expected = nextTokenLogProbability( stored_logits.data() + row * shape.vocab, shape.vocab,
                    host_tokens.data()[ first_position + row + 1 ], shape.softcap );

                EXPECT_NEAR( device[ static_cast<std::size_t>( first_position + row ) ], expected, kTolerance )
                    << std::format( "vocab {}, softcap {}, row {}", shape.vocab, shape.softcap, row );
            }
        }
    }

    class NextTokenLogProbabilityCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "CUDA device not available";
            }

            context_ = createExecutionContext( Device::Cuda( 0 ) );
        }

        std::unique_ptr<IExecutionContext> context_;
    };

    TEST_F( NextTokenLogProbabilityCudaTests, Fp32_MatchesHost )
    {
        for ( const Case& shape : { Case{ 128256, 64, 0.0f }, Case{ 262144, 7, 30.0f }, Case{ 1001, 3, 0.0f } } )
        {
            expectDeviceMatchesHost<TensorDataType::FP32>( context_.get(), shape, 5 );
        }
    }

    TEST_F( NextTokenLogProbabilityCudaTests, Bf16_MatchesHost )
    {
        for ( const Case& shape : { Case{ 128256, 64, 0.0f }, Case{ 262144, 7, 30.0f }, Case{ 1001, 3, 0.0f } } )
        {
            expectDeviceMatchesHost<TensorDataType::BF16>( context_.get(), shape, 5 );
        }
    }

    // Two windows enqueued before one synchronize, as a network scores a sequence: each lands at its own
    // positions, and the second does not disturb the first.
    TEST_F( NextTokenLogProbabilityCudaTests, WindowsLandAtTheirPositions )
    {
        OpType<TensorDataType::FP32> op( context_.get(), 0.0f );
        op.begin( 16 );

        Tensor<TensorDataType::FP32, CpuMemoryResource> uniform( Device::Cpu(), shape_t{ 1, 2, 4 } );
        Tensor<TensorDataType::FP32, CpuMemoryResource> peaked( Device::Cpu(), shape_t{ 1, 2, 4 } );
        Tensor<TensorDataType::INT32, CpuMemoryResource> host_tokens( Device::Cpu(), shape_t{ 1, 17 } );

        for ( dim_t index = 0; index < 8; ++index )
        {
            uniform.data()[ index ] = 0.0f;
            peaked.data()[ index ] = index % 4 == 1 ? std::log( 3.0f ) : 0.0f;
        }

        for ( dim_t position = 0; position < 17; ++position )
        {
            host_tokens.data()[ position ] = 1;
        }

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> first( Device::Cuda( 0 ), uniform.shape() );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> second( Device::Cuda( 0 ), peaked.shape() );
        Tensor<TensorDataType::INT32, CudaDeviceMemoryResource> device_tokens( Device::Cuda( 0 ), host_tokens.shape() );
        copy( uniform, first, context_.get() );
        copy( peaked, second, context_.get() );
        copy( host_tokens, device_tokens, context_.get() );

        op.forward( first, device_tokens, 2, 2 );
        op.forward( second, device_tokens, 9, 2 );
        context_->synchronize();

        const auto values = op.logProbabilities( 16 );

        // Equal logits over four tokens give the target 1/4; a target three times as likely as each of the
        // other three gives it 1/2.
        EXPECT_NEAR( values[ 2 ], std::log( 0.25 ), 1e-7 );
        EXPECT_NEAR( values[ 3 ], std::log( 0.25 ), 1e-7 );
        EXPECT_NEAR( values[ 9 ], std::log( 0.5 ), 1e-6 );
        EXPECT_NEAR( values[ 10 ], std::log( 0.5 ), 1e-6 );
    }

    TEST_F( NextTokenLogProbabilityCudaTests, PositionsPastBeginAreRefused )
    {
        OpType<TensorDataType::FP32> op( context_.get(), 0.0f );
        op.begin( 4 );

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> device_logits( Device::Cuda( 0 ), shape_t{ 1, 2, 4 } );
        Tensor<TensorDataType::INT32, CudaDeviceMemoryResource> device_tokens( Device::Cuda( 0 ), shape_t{ 1, 5 } );

        EXPECT_THROW( op.forward( device_logits, device_tokens, 3, 2 ), std::out_of_range );
    }
}
