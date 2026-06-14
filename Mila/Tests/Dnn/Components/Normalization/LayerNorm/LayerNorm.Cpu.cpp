/**
 * @file LayerNorm.Cpu.cpp
 * @brief Concrete-component tests for LayerNorm<DeviceType::Cpu, FP32>.
 *
 * Concrete-component archetype for a normalization leaf WITH parameters (see
 * Specifications/Testing.md). CPU is FP32-only, so this is a plain explicit test;
 * the CUDA companion (Linear.Cuda-style sweep) lives in LayerNorm.Cuda.cpp.
 *
 * LayerNorm over the trailing normalized dimension:
 *   mean = mean(x) ; var = mean((x-mean)^2) (population) ; rstd = 1/sqrt(var+eps)
 *   y_i  = (x_i - mean) * rstd * weight_i + bias_i
 *
 * CPU device, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

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
        using LayerNormCpu = Mila::Dnn::LayerNorm<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        constexpr int64_t kChannels = 8;
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
                for ( int64_t i = 0; i < channels; ++i )
                {
                    mean += x[ i ];
                }
                mean /= channels;

                double var = 0.0;
                for ( int64_t i = 0; i < channels; ++i )
                {
                    const double d = x[ i ] - mean;
                    var += d * d;
                }
                var /= channels;

                const double rstd = 1.0 / std::sqrt( var + eps );

                for ( int64_t i = 0; i < channels; ++i )
                {
                    Y[ r * channels + i ] = static_cast<float>( ( x[ i ] - mean ) * rstd * W[ i ] + (B ? B[ i ] : 0.0f) );
                }
            }
        }

        static_assert( LayerNormCpu::getDeviceType() == DeviceType::Cpu );
        static_assert( LayerNormCpu::getPrecision() == TensorDataType::FP32 );
    }

    class LayerNormCpuTests : public ::testing::Test
    {
    protected:
        static LayerNormConfig config( bool has_bias = true )
        {
            return LayerNormConfig( shape_t{ kChannels } ).withEpsilon( kEpsilon ).withBias( has_bias );
        }

        std::unique_ptr<LayerNormCpu> builtLayerNorm( const shape_t& shape, bool has_bias, RuntimeMode mode )
        {
            auto norm = std::make_unique<LayerNormCpu>( "layernorm", config( has_bias ), Device::Cpu() );
            norm->build( BuildContext( shape, mode, false ) );

            return norm;
        }

        static void setKnownParameters( LayerNormCpu& norm, bool has_bias )
        {
            auto params = norm.getParameters();
            float* weight = static_cast<float*>( params[ 0 ]->rawData() );

            for ( int64_t i = 0; i < kChannels; ++i )
            {
                weight[ i ] = weightValue( i );
            }

            if ( has_bias )
            {
                float* bias = static_cast<float*>( params[ 1 ]->rawData() );
                for ( int64_t i = 0; i < kChannels; ++i )
                {
                    bias[ i ] = biasValue( i );
                }
            }
        }

        static void fillSpread( TensorFp32& t )
        {
            for ( size_t i = 0; i < t.size(); ++i )
            {
                t.data()[ i ] = static_cast<float>( i ) / t.size() * 4.0f - 2.0f;
            }
        }
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( LayerNormCpuTests, Construct_StandaloneSucceeds )
    {
        LayerNormCpu norm( "layernorm", config(), Device::Cpu() );

        EXPECT_EQ( norm.getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( LayerNormCpuTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW( LayerNormCpu( "layernorm", config(), Device::Cuda( 0 ) ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle
    // ====================================================================

    TEST_F( LayerNormCpuTests, Forward_ThrowsBeforeBuild )
    {
        LayerNormCpu norm( "layernorm", config(), Device::Cpu() );
        TensorFp32 input( Device::Cpu(), shape_t{ 2, 3, kChannels } );

        EXPECT_THROW( norm.forward( input ), std::runtime_error );
    }

    TEST_F( LayerNormCpuTests, Build_ThrowsOnTrailingDimMismatch )
    {
        LayerNormCpu norm( "layernorm", config(), Device::Cpu() );

        EXPECT_THROW(
            norm.build( BuildContext( shape_t{ 2, 3, kChannels + 1 }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // D. Runtime + Training Mode
    // ====================================================================

    TEST_F( LayerNormCpuTests, Backward_ThrowsWhenBuiltForInference )
    {
        const shape_t shape{ 2, 3, kChannels };
        auto norm = builtLayerNorm( shape, true, RuntimeMode::Inference );
        setKnownParameters( *norm, true );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape );
        fillSpread( input );

        norm->forward( input );

        EXPECT_THROW( norm->backward( input, output_grad ), std::runtime_error );
    }

    TEST_F( LayerNormCpuTests, GetGradients_ThrowsWhenBuiltForInference )
    {
        auto norm = builtLayerNorm( shape_t{ 2, 3, kChannels }, true, RuntimeMode::Inference );

        EXPECT_THROW( norm->getGradients(), std::runtime_error );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( LayerNormCpuTests, Forward_MatchesReference )
    {
        const shape_t shape{ 2, 3, kChannels };
        auto norm = builtLayerNorm( shape, true, RuntimeMode::Inference );
        setKnownParameters( *norm, true );

        TensorFp32 input( Device::Cpu(), shape );
        fillSpread( input );

        auto& output = norm->forward( input );

        std::vector<float> weight( kChannels ), bias( kChannels );
        for ( int64_t i = 0; i < kChannels; ++i )
        {
            weight[ i ] = weightValue( i );
            bias[ i ] = biasValue( i );
        }

        std::vector<float> expected;
        referenceForward( input.data(), weight.data(), bias.data(), 2 * 3, kChannels, kEpsilon, expected );

        ASSERT_EQ( output.size(), expected.size() );

        for ( size_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], expected[ i ], 1e-4f ) << "forward mismatch at index " << i;
        }
    }

    TEST_F( LayerNormCpuTests, Forward_MatchesReferenceWithoutBias )
    {
        const shape_t shape{ 2, 3, kChannels };
        auto norm = builtLayerNorm( shape, false, RuntimeMode::Inference );
        setKnownParameters( *norm, false );

        TensorFp32 input( Device::Cpu(), shape );
        fillSpread( input );

        auto& output = norm->forward( input );

        std::vector<float> weight( kChannels );
        for ( int64_t i = 0; i < kChannels; ++i )
        {
            weight[ i ] = weightValue( i );
        }

        std::vector<float> expected;
        referenceForward( input.data(), weight.data(), nullptr, 2 * 3, kChannels, kEpsilon, expected );

        for ( size_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], expected[ i ], 1e-4f ) << "no-bias forward mismatch at index " << i;
        }
    }

    // ====================================================================
    // F. Backward (numeric vs analytic gradient)
    // ====================================================================

    TEST_F( LayerNormCpuTests, Backward_MatchesReferenceGradients )
    {
        const int64_t outer = 2 * 3;
        const shape_t shape{ 2, 3, kChannels };
        auto norm = builtLayerNorm( shape, true, RuntimeMode::Training );
        setKnownParameters( *norm, true );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape );
        fillSpread( input );
        for ( size_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = 0.1f * static_cast<float>( ( i % 7 ) + 1 );
        }

        norm->forward( input );
        auto& input_grad = norm->backward( input, output_grad );

        std::vector<float> weight( kChannels );
        for ( int64_t i = 0; i < kChannels; ++i )
        {
            weight[ i ] = weightValue( i );
        }

        // Analytic LayerNorm input gradient + parameter gradients.
        std::vector<float> expected_dx( static_cast<size_t>( outer * kChannels ), 0.0f );
        std::vector<double> dweight( kChannels, 0.0 ), dbias( kChannels, 0.0 );

        for ( int64_t r = 0; r < outer; ++r )
        {
            const float* x = input.data() + r * kChannels;
            const float* dy = output_grad.data() + r * kChannels;

            double mean = 0.0;
            for ( int64_t i = 0; i < kChannels; ++i ) mean += x[ i ];
            mean /= kChannels;

            double var = 0.0;
            for ( int64_t i = 0; i < kChannels; ++i ) { double d = x[ i ] - mean; var += d * d; }
            var /= kChannels;
            const double rstd = 1.0 / std::sqrt( var + kEpsilon );

            double dnorm_mean = 0.0, dnorm_norm_mean = 0.0;
            for ( int64_t i = 0; i < kChannels; ++i )
            {
                const double norm = ( x[ i ] - mean ) * rstd;
                const double dnorm = weight[ i ] * dy[ i ];
                dnorm_mean += dnorm;
                dnorm_norm_mean += dnorm * norm;
            }
            dnorm_mean /= kChannels;
            dnorm_norm_mean /= kChannels;

            for ( int64_t i = 0; i < kChannels; ++i )
            {
                const double norm = ( x[ i ] - mean ) * rstd;
                const double dnorm = weight[ i ] * dy[ i ];
                expected_dx[ r * kChannels + i ] =
                    static_cast<float>( ( dnorm - dnorm_mean - norm * dnorm_norm_mean ) * rstd );

                dweight[ i ] += dy[ i ] * norm;
                dbias[ i ] += dy[ i ];
            }
        }

        ASSERT_EQ( input_grad.size(), expected_dx.size() );
        for ( size_t i = 0; i < input_grad.size(); ++i )
        {
            EXPECT_NEAR( input_grad.data()[ i ], expected_dx[ i ], 1e-3f ) << "dX mismatch at index " << i;
        }

        auto grads = norm->getGradients();
        ASSERT_EQ( grads.size(), 2u );
        const float* weight_grad = static_cast<const float*>( grads[ 0 ]->rawData() );
        const float* bias_grad = static_cast<const float*>( grads[ 1 ]->rawData() );

        for ( int64_t i = 0; i < kChannels; ++i )
        {
            EXPECT_NEAR( weight_grad[ i ], static_cast<float>( dweight[ i ] ), 1e-3f ) << "dW mismatch at " << i;
            EXPECT_NEAR( bias_grad[ i ], static_cast<float>( dbias[ i ] ), 1e-3f ) << "dB mismatch at " << i;
        }
    }

    // ====================================================================
    // G. Parameters & Gradients
    // ====================================================================

    TEST_F( LayerNormCpuTests, ParameterCount_WithBias )
    {
        auto norm = builtLayerNorm( shape_t{ 2, 3, kChannels }, true, RuntimeMode::Inference );

        EXPECT_EQ( norm->parameterCount(), static_cast<size_t>( 2 * kChannels ) );
        EXPECT_EQ( norm->getParameters().size(), 2u );
    }

    TEST_F( LayerNormCpuTests, ParameterCount_WithoutBias )
    {
        auto norm = builtLayerNorm( shape_t{ 2, 3, kChannels }, false, RuntimeMode::Inference );

        EXPECT_EQ( norm->parameterCount(), static_cast<size_t>( kChannels ) );
        EXPECT_EQ( norm->getParameters().size(), 1u );
    }

    // ====================================================================
    // G/H. Parameter load
    // ====================================================================

    TEST_F( LayerNormCpuTests, LoadParameter_SetsWeightAndBias )
    {
        const shape_t shape{ 2, kChannels };
        auto norm = builtLayerNorm( shape, true, RuntimeMode::Inference );

        std::vector<float> weight( kChannels ), bias( kChannels );
        for ( int64_t i = 0; i < kChannels; ++i )
        {
            weight[ i ] = weightValue( i );
            bias[ i ] = biasValue( i );
        }

        TensorMetadata wmeta{ TensorDataType::FP32, shape_t{ kChannels }, weight.size() * sizeof( float ) };
        TensorBlobView wblob( wmeta, weight.data(), weight.size() * sizeof( float ) );
        norm->loadParameter( "weight", wblob );

        TensorMetadata bmeta{ TensorDataType::FP32, shape_t{ kChannels }, bias.size() * sizeof( float ) };
        TensorBlobView bblob( bmeta, bias.data(), bias.size() * sizeof( float ) );
        norm->loadParameter( "bias", bblob );

        TensorFp32 input( Device::Cpu(), shape );
        fillSpread( input );

        auto& output = norm->forward( input );

        std::vector<float> expected;
        referenceForward( input.data(), weight.data(), bias.data(), 2, kChannels, kEpsilon, expected );

        for ( size_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], expected[ i ], 1e-4f ) << "loaded-weight forward mismatch at index " << i;
        }
    }

    TEST_F( LayerNormCpuTests, LoadParameter_ThrowsOnShapeMismatch )
    {
        auto norm = builtLayerNorm( shape_t{ 2, kChannels }, true, RuntimeMode::Inference );

        std::vector<float> weight( kChannels + 1, 0.0f );
        TensorMetadata meta{ TensorDataType::FP32, shape_t{ kChannels + 1 }, weight.size() * sizeof( float ) };
        TensorBlobView blob( meta, weight.data(), weight.size() * sizeof( float ) );

        EXPECT_THROW( norm->loadParameter( "weight", blob ), std::invalid_argument );
    }

    // ====================================================================
    // I. Diagnostics & J. Type identity
    // ====================================================================

    TEST_F( LayerNormCpuTests, ToString_DescribesComponent )
    {
        LayerNormCpu norm( "my_layernorm", config(), Device::Cpu() );

        const std::string text = norm.toString();

        EXPECT_NE( text.find( "LayerNorm" ), std::string::npos );
        EXPECT_NE( text.find( "my_layernorm" ), std::string::npos );
        EXPECT_NE( text.find( "Has Bias:" ), std::string::npos );
    }

    TEST_F( LayerNormCpuTests, GetType_IsLayerNorm )
    {
        LayerNormCpu norm( "layernorm", config(), Device::Cpu() );

        EXPECT_EQ( norm.getType(), ComponentType::LayerNorm );
    }
}
