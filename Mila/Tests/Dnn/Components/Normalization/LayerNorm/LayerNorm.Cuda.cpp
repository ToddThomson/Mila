/**
 * @file LayerNorm.Cuda.cpp
 * @brief Concrete-component tests for LayerNorm<DeviceType::Cuda, FP32>.
 *
 * CUDA companion to LayerNorm.Cpu.cpp. The CUDA LayerNormOp is specialized for
 * FP32 and FP16 (no BF16 -- GPT-2 lineage); FP16 is not used elsewhere in Mila,
 * so this file tests the validated FP32 path explicitly (no precision sweep). The
 * backward numeric oracle lives in the CPU companion; here we cover the CUDA
 * forward numerics and the structural surface.
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device at runtime.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

import Mila;
import Compute.ExecutionContext;

namespace Mila::Tests::Dnn::Components::Normalization::LayerNorm
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using LayerNormCuda = Mila::Dnn::LayerNorm<DeviceType::Cuda, TensorDataType::FP32>;
        using DeviceTensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        constexpr int64_t kChannels = 16;
        constexpr float kEpsilon = 1e-5f;

        float weightValue( int64_t i )
        {
            return 0.5f + 0.1f * static_cast<float>( ( i % 5 ) - 2 );
        }

        float biasValue( int64_t i )
        {
            return 0.05f * static_cast<float>( ( i % 7 ) - 3 );
        }

        void referenceForward(
            const float* X, const float* W, const float* B,
            int64_t outer, int64_t channels, float eps, std::vector<float>& Y )
        {
            Y.assign( static_cast<size_t>( outer * channels ), 0.0f );

            for ( int64_t r = 0; r < outer; ++r )
            {
                const float* x = X + r * channels;

                double mean = 0.0;
                for ( int64_t i = 0; i < channels; ++i ) mean += x[ i ];
                mean /= channels;

                double var = 0.0;
                for ( int64_t i = 0; i < channels; ++i ) { double d = x[ i ] - mean; var += d * d; }
                var /= channels;
                const double rstd = 1.0 / std::sqrt( var + eps );

                for ( int64_t i = 0; i < channels; ++i )
                {
                    Y[ r * channels + i ] = static_cast<float>( ( x[ i ] - mean ) * rstd * W[ i ] + B[ i ] );
                }
            }
        }

        static_assert( LayerNormCuda::getDeviceType() == DeviceType::Cuda );
        static_assert( LayerNormCuda::getPrecision() == TensorDataType::FP32 );
    }

    class LayerNormCudaTests : public ::testing::Test
    {
    protected:
        static LayerNormConfig config()
        {
            return LayerNormConfig( shape_t{ kChannels } ).withEpsilon( kEpsilon );
        }

        void SetUp() override
        {
            try
            {
                cuda_context_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                cuda_context_ = nullptr;
            }

            if ( !cuda_context_ )
            {
                GTEST_SKIP() << "CUDA device not available";
            }
        }

        std::unique_ptr<LayerNormCuda> builtLayerNorm( const shape_t& shape, RuntimeMode mode )
        {
            auto norm = std::make_unique<LayerNormCuda>( "layernorm", config(), Device::Cuda( 0 ) );
            norm->build( BuildContext( shape, mode, false ) );

            return norm;
        }

        void setKnownParameters( LayerNormCuda& norm )
        {
            auto params = norm.getParameters();

            HostTensor host_weight( Device::Cpu(), shape_t{ kChannels } );
            HostTensor host_bias( Device::Cpu(), shape_t{ kChannels } );
            for ( int64_t i = 0; i < kChannels; ++i )
            {
                host_weight.data()[ i ] = weightValue( i );
                host_bias.data()[ i ] = biasValue( i );
            }

            copy( host_weight, *static_cast<DeviceTensor*>( params[ 0 ] ), cuda_context_.get() );
            copy( host_bias, *static_cast<DeviceTensor*>( params[ 1 ] ), cuda_context_.get() );
            cuda_context_->synchronize();
        }

        HostTensor spreadHost( const shape_t& shape )
        {
            HostTensor host( Device::Cpu(), shape );
            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = static_cast<float>( i ) / host.size() * 4.0f - 2.0f;
            }

            return host;
        }

        DeviceTensor toDevice( const HostTensor& host )
        {
            DeviceTensor device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    // ====================================================================
    // A. Construction
    // ====================================================================

    TEST_F( LayerNormCudaTests, Construct_StandaloneSucceeds )
    {
        LayerNormCuda norm( "layernorm", config(), Device::Cuda( 0 ) );

        EXPECT_EQ( norm.getDeviceId().type, DeviceType::Cuda );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( LayerNormCudaTests, Forward_MatchesReference )
    {
        const shape_t shape{ 2, 3, kChannels };

        auto norm = builtLayerNorm( shape, RuntimeMode::Inference );
        setKnownParameters( *norm );

        auto host_in = spreadHost( shape );
        auto device_in = toDevice( host_in );

        auto& device_out = norm->forward( device_in );
        norm->synchronize();

        auto out = toHost<TensorDataType::FP32>( device_out, cuda_context_.get() );
        cuda_context_->synchronize();

        std::vector<float> weight( kChannels ), bias( kChannels );
        for ( int64_t i = 0; i < kChannels; ++i )
        {
            weight[ i ] = weightValue( i );
            bias[ i ] = biasValue( i );
        }

        std::vector<float> expected;
        referenceForward( host_in.data(), weight.data(), bias.data(), 2 * 3, kChannels, kEpsilon, expected );

        ASSERT_EQ( out.size(), expected.size() );

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            EXPECT_NEAR( out.data()[ i ], expected[ i ], 1e-3f ) << "forward mismatch at index " << i;
        }
    }

    // ====================================================================
    // G. Parameters & Gradients
    // ====================================================================

    TEST_F( LayerNormCudaTests, Parameters_WeightAndBias )
    {
        auto norm = builtLayerNorm( shape_t{ 2, 3, kChannels }, RuntimeMode::Inference );

        EXPECT_EQ( norm->getParameters().size(), 2u );
        EXPECT_EQ( norm->parameterCount(), static_cast<size_t>( 2 * kChannels ) );
    }

    TEST_F( LayerNormCudaTests, GetGradients_EmptyForInferenceBuild )
    {
        auto norm = builtLayerNorm( shape_t{ 2, 3, kChannels }, RuntimeMode::Inference );

        EXPECT_TRUE( norm->getGradients().empty() );
    }

    TEST_F( LayerNormCudaTests, GetGradients_PresentForTrainingBuild )
    {
        auto norm = builtLayerNorm( shape_t{ 2, 3, kChannels }, RuntimeMode::Training );

        EXPECT_EQ( norm->getGradients().size(), 2u );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TEST_F( LayerNormCudaTests, GetType_IsLayerNorm )
    {
        LayerNormCuda norm( "layernorm", config(), Device::Cuda( 0 ) );

        EXPECT_EQ( norm.getType(), ComponentType::LayerNorm );
    }
}
