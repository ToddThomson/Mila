/**
 * @file Gelu.Cpu.cpp
 * @brief Concrete-component tests for Gelu<DeviceType::Cpu, FP32>.
 *
 * Reference instance of the concrete-component test archetype (see
 * Specifications/Testing.md). Tests only the Gelu DELTA over the Component base
 * contract: GELU-specific construction/validation, the build->forward->backward
 * path, the GELU numeric reference, and the stateless parameter contract. It
 * does NOT re-test the inherited base machinery — that is covered once in
 * Core/Component.cpp.
 *
 * CPU device, so this rides the MILA_ENABLE_CUDA=OFF CI gate. CUDA-instantiation
 * tests belong in Gelu.Cuda.cpp.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

#include "Common/GradientCheck.h"

import Mila;

namespace Mila::Tests::Dnn::Components::Activations::Gelu
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using GeluCpu = Mila::Dnn::Gelu<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        // GELU tanh-approximation reference (the only method GeluConfig supports).
        // Computed independently of the component under test.
        constexpr float kSqrt2OverPi = 0.7978845608f;
        constexpr float kGeluCoeff = 0.044715f;

        float geluReference( float x )
        {
            const float x_cubed = x * x * x;

            return 0.5f * x * (1.0f + std::tanh( kSqrt2OverPi * (x + kGeluCoeff * x_cubed) ));
        }

        float geluGradientReference( float x )
        {
            const float x_squared = x * x;
            const float arg = kSqrt2OverPi * (x + kGeluCoeff * x * x_squared);
            const float tanh_value = std::tanh( arg );
            const float sech_squared = 1.0f - tanh_value * tanh_value;
            const float d_arg = kSqrt2OverPi * (1.0f + 3.0f * kGeluCoeff * x_squared);

            return 0.5f * (1.0f + tanh_value) + 0.5f * x * sech_squared * d_arg;
        }

        static_assert( GeluCpu::getDeviceType() == DeviceType::Cpu );
        static_assert( GeluCpu::getPrecision() == TensorDataType::FP32 );
    }

    class GeluCpuTests : public ::testing::Test
    {
    protected:
        // Standalone Gelu (owns its CPU execution context), built in the given mode.
        std::unique_ptr<GeluCpu> builtGelu( const shape_t& shape, RuntimeMode mode )
        {
            auto gelu = std::make_unique<GeluCpu>( "gelu", GeluConfig(), Device::Cpu() );
            gelu->build( BuildContext( shape, mode ) );

            return gelu;
        }

        // Fill a tensor with a deterministic spread across roughly [-2, 2].
        static void fillSpread( TensorFp32& t )
        {
            auto* data = t.data();

            for ( size_t i = 0; i < t.size(); ++i )
            {
                data[ i ] = static_cast<float>( i ) / t.size() * 4.0f - 2.0f;
            }
        }
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( GeluCpuTests, Construct_StandaloneSucceeds )
    {
        GeluCpu gelu( "gelu", GeluConfig(), Device::Cpu() );

        EXPECT_EQ( gelu.getApproximationMethod(), ApproximationMethod::Tanh );
        EXPECT_EQ( gelu.getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( GeluCpuTests, Construct_InvalidConfigThrows )
    {
        // GeluConfig only supports Tanh; Exact must fail validation in the ctor.
        GeluConfig bad = GeluConfig().withApproximationMethod( ApproximationMethod::Exact );

        EXPECT_THROW( GeluCpu( "gelu", bad, Device::Cpu() ), std::invalid_argument );
    }

    TEST_F( GeluCpuTests, Construct_DeviceTypeMismatchThrows )
    {
        // A CUDA DeviceId on a CPU-typed component is rejected before any device
        // work, so this is safe on a host without a GPU.
        EXPECT_THROW( GeluCpu( "gelu", GeluConfig(), Device::Cuda( 0 ) ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle (Gelu preconditions)
    // ====================================================================

    TEST_F( GeluCpuTests, Forward_ThrowsBeforeBuild )
    {
        GeluCpu gelu( "gelu", GeluConfig(), Device::Cpu() );
        TensorFp32 input( Device::Cpu(), shape_t{ 2, 4 } );

        EXPECT_THROW( gelu.forward( input ), std::runtime_error );
    }

    TEST_F( GeluCpuTests, Backward_ThrowsBeforeBuild )
    {
        GeluCpu gelu( "gelu", GeluConfig(), Device::Cpu() );
        TensorFp32 input( Device::Cpu(), shape_t{ 2, 4 } );
        TensorFp32 output_grad( Device::Cpu(), shape_t{ 2, 4 } );

        EXPECT_THROW( gelu.backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( GeluCpuTests, Forward_MatchesReference )
    {
        const shape_t shape{ 2, 3, 4 };
        auto gelu = builtGelu( shape, RuntimeMode::Inference );

        TensorFp32 input( Device::Cpu(), shape );
        fillSpread( input );

        auto& output = gelu->forward( input );

        ASSERT_EQ( output.size(), input.size() );

        constexpr float tolerance = 1e-4f;

        for ( size_t i = 0; i < input.size(); ++i )
        {
            const float expected = geluReference( input.data()[ i ] );

            EXPECT_NEAR( output.data()[ i ], expected, tolerance )
                << "forward mismatch at index " << i << " input=" << input.data()[ i ];
        }
    }

    // ====================================================================
    // F. Backward (numeric vs analytic gradient)
    // ====================================================================

    TEST_F( GeluCpuTests, Backward_MatchesGradientReference )
    {
        const shape_t shape{ 2, 3, 4 };

        // Training build allocates the input-gradient buffer the backward pass needs.
        auto gelu = builtGelu( shape, RuntimeMode::Training );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape );
        fillSpread( input );

        for ( size_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = 1.0f;
        }

        gelu->forward( input );
        auto& input_grad = gelu->backward( input, output_grad );

        ASSERT_EQ( input_grad.size(), input.size() );

        constexpr float tolerance = 1e-3f;

        for ( size_t i = 0; i < input.size(); ++i )
        {
            const float expected = geluGradientReference( input.data()[ i ] ) * output_grad.data()[ i ];

            EXPECT_NEAR( input_grad.data()[ i ], expected, tolerance )
                << "backward mismatch at index " << i << " input=" << input.data()[ i ];
        }
    }

    TEST_F( GeluCpuTests, Backward_ChainsNonUniformOutputGradient )
    {
        const shape_t shape{ 2, 3, 4 };
        auto gelu = builtGelu( shape, RuntimeMode::Training );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape );
        fillSpread( input );

        for ( size_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = static_cast<float>( i + 1 ) * 0.1f;
        }

        gelu->forward( input );
        auto& input_grad = gelu->backward( input, output_grad );

        constexpr float tolerance = 1e-3f;

        for ( size_t i = 0; i < input.size(); ++i )
        {
            const float expected = geluGradientReference( input.data()[ i ] ) * output_grad.data()[ i ];

            EXPECT_NEAR( input_grad.data()[ i ], expected, tolerance )
                << "chain-rule mismatch at index " << i;
        }
    }

    // Reference application of the finite-difference gradient-check archetype
    // (Specifications/Testing.md). Unlike Backward_MatchesGradientReference above
    // -- which checks against a hand-derived analytic GELU derivative -- this
    // verifies backward() against a numeric gradient of the component's own
    // forward(), so it needs no per-component math and generalizes verbatim to
    // every leaf with the forward(input)->output& signature.
    TEST_F( GeluCpuTests, Backward_MatchesNumericGradient )
    {
        const shape_t shape{ 2, 3, 4 };
        auto gelu = builtGelu( shape, RuntimeMode::Training );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape );
        fillSpread( input );

        for ( size_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = static_cast<float>( i + 1 ) * 0.1f;
        }

        gelu->forward( input );
        auto& input_grad = gelu->backward( input, output_grad );

        // Snapshot the analytic gradient: the finite-difference probe re-runs
        // forward(), which may reuse component-internal buffers.
        std::vector<float> analytic( input_grad.data(), input_grad.data() + input_grad.size() );

        const auto numeric = Mila::Tests::Common::centralDifferenceGradient(
            input.data(), input.size(),
            output_grad.data(), output_grad.size(),
            [&]() -> const float* { return gelu->forward( input ).data(); },
            1e-2f );

        Mila::Tests::Common::expectGradientsClose( analytic.data(), numeric, 1e-3f, 1e-2f, "Gelu dX" );
    }

    // ====================================================================
    // G. Parameters & Gradients (Gelu is stateless)
    // ====================================================================

    TEST_F( GeluCpuTests, Parameters_AreEmpty )
    {
        auto gelu = builtGelu( shape_t{ 2, 4 }, RuntimeMode::Inference );

        EXPECT_EQ( gelu->parameterCount(), 0 );
        EXPECT_TRUE( gelu->getParameters().empty() );
        EXPECT_TRUE( gelu->getGradients().empty() );
    }

    // H. Serialization (save_/round-trip) — deferred to the serialization pass;
    // requires a ModelArchive fixture. GeluConfig metadata round-trip is covered
    // in GeluConfig.cpp.

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( GeluCpuTests, ToString_NamesComponent )
    {
        GeluCpu gelu( "my_gelu", GeluConfig(), Device::Cpu() );

        const std::string text = gelu.toString();

        EXPECT_NE( text.find( "Gelu" ), std::string::npos );
        EXPECT_NE( text.find( "my_gelu" ), std::string::npos );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TEST_F( GeluCpuTests, GetType_IsGelu )
    {
        GeluCpu gelu( "gelu", GeluConfig(), Device::Cpu() );

        EXPECT_EQ( gelu.getType(), ComponentType::Gelu );
    }
}
