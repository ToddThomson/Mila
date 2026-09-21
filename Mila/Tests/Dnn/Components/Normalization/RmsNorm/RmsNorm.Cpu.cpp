/**
 * @file RmsNorm.Cpu.cpp
 * @brief Concrete-component tests for RmsNorm<DeviceType::Cpu, FP32>.
 *
 * Concrete-component archetype for a normalization leaf WITH parameters (see
 * Specifications/Testing.md). CPU is FP32-only, so this is a plain explicit test; the CUDA
 * companion sweeps FP32 and BF16 in RmsNorm.Cuda.cpp.
 *
 * RMS normalization over the trailing normalized dimension:
 *   rstd = 1 / sqrt( mean(x^2) + eps )
 *   y_i  = x_i * rstd * ( weight_i + unit_offset ) + bias_i
 *
 * The unit offset is exercised explicitly: Qwen configures every norm with 1.0, and it is the term
 * the CUDA backward does not apply (Mila/Issues/Future.md, "Training (advanced)").
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

#include "Common/GradientCheck.h"

import Mila;

namespace Mila::Tests::Dnn::Components::Normalization::RmsNorm
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using RmsNormCpu = Mila::Dnn::RmsNorm<DeviceType::Cpu, TensorDataType::FP32>;
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
            int64_t outer, int64_t channels, float eps, float unit_offset, std::vector<float>& Y )
        {
            Y.assign( static_cast<size_t>( outer * channels ), 0.0f );

            for ( int64_t r = 0; r < outer; ++r )
            {
                const float* x = X + r * channels;

                double sum_of_squares = 0.0;
                for ( int64_t i = 0; i < channels; ++i )
                {
                    sum_of_squares += static_cast<double>( x[ i ] ) * x[ i ];
                }

                const double rstd = 1.0 / std::sqrt( sum_of_squares / channels + eps );

                for ( int64_t i = 0; i < channels; ++i )
                {
                    Y[ r * channels + i ] = static_cast<float>(
                        x[ i ] * rstd * ( W[ i ] + unit_offset ) + ( B ? B[ i ] : 0.0f ) );
                }
            }
        }

        static_assert( RmsNormCpu::getDeviceType() == DeviceType::Cpu );
        static_assert( RmsNormCpu::getPrecision() == TensorDataType::FP32 );
    }

    class RmsNormCpuTests : public ::testing::Test
    {
    protected:
        static RmsNormConfig config( bool has_bias, float unit_offset = 0.0f )
        {
            return RmsNormConfig( shape_t{ kChannels } )
                .withEpsilon( kEpsilon )
                .withBias( has_bias )
                .withUnitOffset( unit_offset );
        }

        std::unique_ptr<RmsNormCpu> builtRmsNorm( const shape_t& shape, bool has_bias, RuntimeMode mode, float unit_offset = 0.0f )
        {
            auto norm = std::make_unique<RmsNormCpu>( "rmsnorm", config( has_bias, unit_offset ), Device::Cpu() );
            norm->build( BuildContext( shape, mode, false ) );

            return norm;
        }

        static void setKnownParameters( RmsNormCpu& norm, bool has_bias )
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
            for ( dim_t i = 0; i < t.size(); ++i )
            {
                t.data()[ i ] = static_cast<float>( i ) / t.size() * 4.0f - 2.0f;
            }
        }

        static void expectMatchesReference( RmsNormCpu& norm, const shape_t& shape, bool has_bias, float unit_offset )
        {
            TensorFp32 input( Device::Cpu(), shape );
            fillSpread( input );

            auto& output = norm.forward( input );

            std::vector<float> weight( kChannels );
            std::vector<float> bias( kChannels );

            for ( int64_t i = 0; i < kChannels; ++i )
            {
                weight[ i ] = weightValue( i );
                bias[ i ] = biasValue( i );
            }

            std::vector<float> expected;
            referenceForward( input.data(), weight.data(), has_bias ? bias.data() : nullptr,
                input.size() / kChannels, kChannels, kEpsilon, unit_offset, expected );

            ASSERT_EQ( output.shape(), shape );
            ASSERT_EQ( output.size(), static_cast<dim_t>( expected.size() ) );

            for ( dim_t i = 0; i < output.size(); ++i )
            {
                EXPECT_NEAR( output.data()[ i ], expected[ i ], 1e-5f ) << "forward mismatch at index " << i;
            }
        }
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( RmsNormCpuTests, Construct_StandaloneSucceeds )
    {
        RmsNormCpu norm( "rmsnorm", config( true ), Device::Cpu() );

        EXPECT_EQ( norm.getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( RmsNormCpuTests, Forward_ThrowsBeforeBuild )
    {
        RmsNormCpu norm( "rmsnorm", config( true ), Device::Cpu() );
        TensorFp32 input( Device::Cpu(), shape_t{ 2, 3, kChannels } );

        EXPECT_THROW( norm.forward( input ), std::runtime_error );
    }

    TEST_F( RmsNormCpuTests, Build_ThrowsOnTrailingDimMismatch )
    {
        RmsNormCpu norm( "rmsnorm", config( true ), Device::Cpu() );

        EXPECT_THROW(
            norm.build( BuildContext( shape_t{ 2, 3, kChannels + 1 }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // D. Runtime mode
    // ====================================================================

    TEST_F( RmsNormCpuTests, Backward_ThrowsWhenBuiltForInference )
    {
        const shape_t shape{ 2, 3, kChannels };
        auto norm = builtRmsNorm( shape, true, RuntimeMode::Inference );
        setKnownParameters( *norm, true );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape );
        fillSpread( input );
        fillSpread( output_grad );

        norm->forward( input );

        EXPECT_THROW( norm->backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( RmsNormCpuTests, Forward_MatchesReference )
    {
        const shape_t shape{ 2, 3, kChannels };
        auto norm = builtRmsNorm( shape, true, RuntimeMode::Inference );
        setKnownParameters( *norm, true );

        expectMatchesReference( *norm, shape, true, 0.0f );
    }

    TEST_F( RmsNormCpuTests, Forward_MatchesReferenceWithoutBias )
    {
        const shape_t shape{ 2, 3, kChannels };
        auto norm = builtRmsNorm( shape, false, RuntimeMode::Inference );
        setKnownParameters( *norm, false );

        expectMatchesReference( *norm, shape, false, 0.0f );
    }

    TEST_F( RmsNormCpuTests, Forward_AppliesUnitOffset )
    {
        const shape_t shape{ 2, 3, kChannels };
        auto norm = builtRmsNorm( shape, false, RuntimeMode::Inference, 1.0f );
        setKnownParameters( *norm, false );

        expectMatchesReference( *norm, shape, false, 1.0f );
    }

    // A component built at the prefill width is called with a single row at decode.
    TEST_F( RmsNormCpuTests, Forward_NarrowerInputThanBuildMatchesReference )
    {
        auto norm = builtRmsNorm( shape_t{ 2, 3, kChannels }, true, RuntimeMode::Inference );
        setKnownParameters( *norm, true );

        expectMatchesReference( *norm, shape_t{ 2, 1, kChannels }, true, 0.0f );
    }

    // ====================================================================
    // F. Backward (numeric gradient)
    // ====================================================================

    // Finite-difference gradient-check archetype, with a unit offset so the (weight + offset)
    // factor in the input gradient is exercised rather than assumed.
    TEST_F( RmsNormCpuTests, Backward_MatchesNumericGradientWithUnitOffset )
    {
        const shape_t shape{ 2, 3, kChannels };
        auto norm = builtRmsNorm( shape, true, RuntimeMode::Training, 1.0f );
        setKnownParameters( *norm, true );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape );
        fillSpread( input );

        for ( dim_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = 0.1f * static_cast<float>( ( i % 7 ) + 1 );
        }

        norm->forward( input );
        auto& input_grad = norm->backward( input, output_grad );

        // Snapshot analytic gradients before the probe re-runs forward().
        std::vector<float> analytic_dx( input_grad.data(), input_grad.data() + input_grad.size() );

        auto params = norm->getParameters();
        auto grads = norm->getGradients();
        ASSERT_EQ( grads.size(), 2u );

        float* weight = static_cast<float*>( params[ 0 ]->rawData() );
        float* bias = static_cast<float*>( params[ 1 ]->rawData() );
        const float* weight_grad = static_cast<const float*>( grads[ 0 ]->rawData() );
        const float* bias_grad = static_cast<const float*>( grads[ 1 ]->rawData() );
        std::vector<float> analytic_dw( weight_grad, weight_grad + kChannels );
        std::vector<float> analytic_db( bias_grad, bias_grad + kChannels );

        auto evaluate = [&]() -> const float* { return norm->forward( input ).data(); };

        const auto numeric_dx = Mila::Tests::Common::centralDifferenceGradient(
            input.data(), input.size(), output_grad.data(), output_grad.size(), evaluate, 1e-2f );
        Mila::Tests::Common::expectGradientsClose( analytic_dx.data(), numeric_dx, 1e-2f, 1e-2f, "RmsNorm dX" );

        const auto numeric_dw = Mila::Tests::Common::centralDifferenceGradient(
            weight, kChannels, output_grad.data(), output_grad.size(), evaluate, 1e-2f );
        Mila::Tests::Common::expectGradientsClose( analytic_dw.data(), numeric_dw, 1e-2f, 1e-2f, "RmsNorm dW" );

        const auto numeric_db = Mila::Tests::Common::centralDifferenceGradient(
            bias, kChannels, output_grad.data(), output_grad.size(), evaluate, 1e-2f );
        Mila::Tests::Common::expectGradientsClose( analytic_db.data(), numeric_db, 1e-2f, 1e-2f, "RmsNorm dB" );
    }

    // ====================================================================
    // G. Parameters & Gradients
    // ====================================================================

    TEST_F( RmsNormCpuTests, ParameterCount_WithAndWithoutBias )
    {
        auto with_bias = builtRmsNorm( shape_t{ 2, 3, kChannels }, true, RuntimeMode::Inference );
        auto without_bias = builtRmsNorm( shape_t{ 2, 3, kChannels }, false, RuntimeMode::Inference );

        EXPECT_EQ( with_bias->parameterCount(), 2 * kChannels );
        EXPECT_EQ( without_bias->parameterCount(), kChannels );
    }

    TEST_F( RmsNormCpuTests, GetRequiredMemory_MatchesBuiltFootprint )
    {
        const BuildContext context( shape_t{ 2, 3, kChannels }, RuntimeMode::Training, false );

        RmsNormCpu predictor( "rmsnorm", config( true ), Device::Cpu() );
        const MemoryStats predicted = predictor.getRequiredMemory( context );

        RmsNormCpu built( "rmsnorm", config( true ), Device::Cpu() );
        built.build( context );
        const MemoryStats actual = built.getMemoryStats();

        EXPECT_EQ( predicted.device_parameter_bytes, actual.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, actual.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_gradient_bytes, actual.device_gradient_bytes ) << "gradients";
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TEST_F( RmsNormCpuTests, GetType_IsRmsNorm )
    {
        RmsNormCpu norm( "rmsnorm", config( true ), Device::Cpu() );

        EXPECT_EQ( norm.getType(), ComponentType::RmsNorm );
    }
}
