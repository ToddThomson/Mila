/**
 * @file LayerNorm.Cuda.cpp
 * @brief Concrete-component tests for LayerNorm<DeviceType::Cuda, FP32>.
 *
 * CUDA companion to LayerNorm.Cpu.cpp. The CUDA LayerNormOp is specialized for
 * FP32 and FP16 (no BF16 -- GPT-2 lineage); FP16 is not used elsewhere in Mila,
 * so this file tests the validated FP32 path explicitly (no precision sweep).
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device at runtime.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

import Mila;
import Serialization.Tensor;

namespace Mila::Tests::Dnn::Components::Normalization::LayerNorm
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using Mila::Dnn::Serialization::TensorMetadata;
    using Mila::Dnn::Serialization::TensorBlobView;

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
                    Y[ r * channels + i ] = static_cast<float>( ( x[ i ] - mean ) * rstd * W[ i ] + ( B ? B[ i ] : 0.0f ) );
                }
            }
        }

        static_assert( LayerNormCuda::getDeviceType() == DeviceType::Cuda );
        static_assert( LayerNormCuda::getPrecision() == TensorDataType::FP32 );
    }

    class LayerNormCudaTests : public ::testing::Test
    {
    protected:
        static LayerNormConfig config( bool has_bias = true )
        {
            return LayerNormConfig( shape_t{ kChannels } ).withEpsilon( kEpsilon ).withBias( has_bias );
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

        std::unique_ptr<LayerNormCuda> builtLayerNorm( const shape_t& shape, RuntimeMode mode, bool has_bias = true )
        {
            auto norm = std::make_unique<LayerNormCuda>( "layernorm", config( has_bias ), Device::Cuda( 0 ) );
            norm->build( BuildContext( shape, mode, false ) );

            return norm;
        }

        void setKnownParameters( LayerNormCuda& norm, bool has_bias = true )
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

            if ( has_bias )
            {
                copy( host_bias, *static_cast<DeviceTensor*>( params[ 1 ] ), cuda_context_.get() );
            }

            cuda_context_->synchronize();
        }

        HostTensor toFloat( const DeviceTensor& device )
        {
            auto host = toHost<TensorDataType::FP32>( device, cuda_context_.get() );
            cuda_context_->synchronize();

            return host;
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
    // B. Build Lifecycle
    // ====================================================================

    TEST_F( LayerNormCudaTests, Forward_ThrowsBeforeBuild )
    {
        LayerNormCuda norm( "layernorm", config(), Device::Cuda( 0 ) );
        DeviceTensor input( Device::Cuda( 0 ), shape_t{ 2, 3, kChannels } );

        EXPECT_THROW( norm.forward( input ), std::runtime_error );
    }

    TEST_F( LayerNormCudaTests, Build_ThrowsOnTrailingDimMismatch )
    {
        LayerNormCuda norm( "layernorm", config(), Device::Cuda( 0 ) );

        EXPECT_THROW(
            norm.build( BuildContext( shape_t{ 2, 3, kChannels + 1 }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // D. Runtime mode
    // ====================================================================

    TEST_F( LayerNormCudaTests, Backward_ThrowsWhenBuiltForInference )
    {
        const shape_t shape{ 2, 3, kChannels };
        auto norm = builtLayerNorm( shape, RuntimeMode::Inference );
        setKnownParameters( *norm );

        auto device_in = toDevice( spreadHost( shape ) );
        DeviceTensor output_grad( Device::Cuda( 0 ), shape );

        norm->forward( device_in );

        EXPECT_THROW( norm->backward( device_in, output_grad ), std::runtime_error );
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

    TEST_F( LayerNormCudaTests, Forward_MatchesReferenceWithoutBias )
    {
        const shape_t shape{ 2, 3, kChannels };

        auto norm = builtLayerNorm( shape, RuntimeMode::Inference, false );
        setKnownParameters( *norm, false );

        auto host_in = spreadHost( shape );
        auto& device_out = norm->forward( toDevice( host_in ) );
        norm->synchronize();

        auto out = toFloat( device_out );

        std::vector<float> weight( kChannels );
        for ( int64_t i = 0; i < kChannels; ++i )
        {
            weight[ i ] = weightValue( i );
        }

        std::vector<float> expected;
        referenceForward( host_in.data(), weight.data(), nullptr, 2 * 3, kChannels, kEpsilon, expected );

        ASSERT_EQ( out.size(), expected.size() );

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            EXPECT_NEAR( out.data()[ i ], expected[ i ], 1e-3f ) << "no-bias forward mismatch at index " << i;
        }
    }

    // ====================================================================
    // F. Backward (analytic reference)
    // ====================================================================

    TEST_F( LayerNormCudaTests, Backward_MatchesReferenceGradients )
    {
        const int64_t outer = 2 * 3;
        const shape_t shape{ 2, 3, kChannels };

        auto norm = builtLayerNorm( shape, RuntimeMode::Training );
        setKnownParameters( *norm );

        auto host_in = spreadHost( shape );

        // Non-uniform, so a kernel that ignored its output gradient fails.
        HostTensor host_grad( Device::Cpu(), shape );
        for ( dim_t i = 0; i < host_grad.size(); ++i )
        {
            host_grad.data()[ i ] = 0.1f * static_cast<float>( ( i % 7 ) + 1 );
        }

        auto device_in = toDevice( host_in );
        auto device_grad = toDevice( host_grad );

        norm->zeroGradients();
        norm->forward( device_in );
        auto& device_in_grad = norm->backward( device_in, device_grad );
        norm->synchronize();

        auto input_grad = toFloat( device_in_grad );

        std::vector<double> expected_dx( static_cast<size_t>( outer * kChannels ), 0.0 );
        std::vector<double> expected_dweight( kChannels, 0.0 ), expected_dbias( kChannels, 0.0 );

        for ( int64_t r = 0; r < outer; ++r )
        {
            const float* x = host_in.data() + r * kChannels;
            const float* dy = host_grad.data() + r * kChannels;

            double mean = 0.0;
            for ( int64_t i = 0; i < kChannels; ++i ) mean += x[ i ];
            mean /= kChannels;

            double var = 0.0;
            for ( int64_t i = 0; i < kChannels; ++i ) { double d = x[ i ] - mean; var += d * d; }
            var /= kChannels;
            const double rstd = 1.0 / std::sqrt( var + kEpsilon );

            double dnorm_mean = 0.0;
            double dnorm_norm_mean = 0.0;

            for ( int64_t i = 0; i < kChannels; ++i )
            {
                const double norm_value = ( x[ i ] - mean ) * rstd;
                const double dnorm = weightValue( i ) * dy[ i ];
                dnorm_mean += dnorm;
                dnorm_norm_mean += dnorm * norm_value;
            }

            dnorm_mean /= kChannels;
            dnorm_norm_mean /= kChannels;

            for ( int64_t i = 0; i < kChannels; ++i )
            {
                const double norm_value = ( x[ i ] - mean ) * rstd;
                const double dnorm = weightValue( i ) * dy[ i ];
                expected_dx[ r * kChannels + i ] = ( dnorm - dnorm_mean - norm_value * dnorm_norm_mean ) * rstd;
                expected_dweight[ i ] += dy[ i ] * norm_value;
                expected_dbias[ i ] += dy[ i ];
            }
        }

        ASSERT_EQ( input_grad.size(), static_cast<dim_t>( expected_dx.size() ) );

        for ( dim_t i = 0; i < input_grad.size(); ++i )
        {
            EXPECT_NEAR( input_grad.data()[ i ], static_cast<float>( expected_dx[ i ] ), 1e-3f ) << "dX mismatch at index " << i;
        }

        auto grads = norm->getGradients();
        ASSERT_EQ( grads.size(), 2u );

        auto weight_grad = toFloat( *static_cast<DeviceTensor*>( grads[ 0 ] ) );
        auto bias_grad = toFloat( *static_cast<DeviceTensor*>( grads[ 1 ] ) );

        for ( int64_t i = 0; i < kChannels; ++i )
        {
            EXPECT_NEAR( weight_grad.data()[ i ], static_cast<float>( expected_dweight[ i ] ), 1e-3f ) << "dW mismatch at " << i;
            EXPECT_NEAR( bias_grad.data()[ i ], static_cast<float>( expected_dbias[ i ] ), 1e-3f ) << "dB mismatch at " << i;
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
    // H. Parameter load
    // ====================================================================

    TEST_F( LayerNormCudaTests, LoadParameter_SetsWeightAndBias )
    {
        const shape_t shape{ 2, kChannels };
        auto norm = builtLayerNorm( shape, RuntimeMode::Inference );

        std::vector<float> weight( kChannels ), bias( kChannels );
        for ( int64_t i = 0; i < kChannels; ++i )
        {
            weight[ i ] = weightValue( i );
            bias[ i ] = biasValue( i );
        }

        TensorMetadata weight_metadata{ TensorDataType::FP32, shape_t{ kChannels }, weight.size() * sizeof( float ) };
        norm->loadParameter( "weight", TensorBlobView( weight_metadata, weight.data(), weight.size() * sizeof( float ) ) );

        TensorMetadata bias_metadata{ TensorDataType::FP32, shape_t{ kChannels }, bias.size() * sizeof( float ) };
        norm->loadParameter( "bias", TensorBlobView( bias_metadata, bias.data(), bias.size() * sizeof( float ) ) );

        auto host_in = spreadHost( shape );
        auto& device_out = norm->forward( toDevice( host_in ) );
        norm->synchronize();

        auto out = toFloat( device_out );

        std::vector<float> expected;
        referenceForward( host_in.data(), weight.data(), bias.data(), 2, kChannels, kEpsilon, expected );

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            EXPECT_NEAR( out.data()[ i ], expected[ i ], 1e-3f ) << "loaded-weight forward mismatch at index " << i;
        }
    }

    TEST_F( LayerNormCudaTests, LoadParameter_ThrowsOnShapeMismatch )
    {
        auto norm = builtLayerNorm( shape_t{ 2, kChannels }, RuntimeMode::Inference );

        std::vector<float> weight( kChannels + 1, 0.0f );
        TensorMetadata metadata{ TensorDataType::FP32, shape_t{ kChannels + 1 }, weight.size() * sizeof( float ) };

        EXPECT_THROW(
            norm->loadParameter( "weight", TensorBlobView( metadata, weight.data(), weight.size() * sizeof( float ) ) ),
            std::invalid_argument );
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
