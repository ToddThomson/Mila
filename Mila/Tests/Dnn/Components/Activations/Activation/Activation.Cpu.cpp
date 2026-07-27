/**
 * @file Activation.Cpu.cpp
 * @brief Concrete-component tests for Activation<DeviceType::Cpu, FP32, TFn>.
 *
 * Applies the typed-sweep methodology (Specifications/Testing.md): one suite swept
 * over the elementwise functions, each carrying its own independent numeric
 * reference. Tests only the Activation DELTA over the Component base contract --
 * the function-selection mechanism, the build->forward->backward path against a
 * per-function reference, and the stateless parameter contract.
 *
 * CPU device, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <string>

import Mila;

namespace Mila::Tests::Dnn::Components::Activations::Activation
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

    // ====================================================================
    // Per-function traits: the ActivationType to instantiate, plus independent
    // forward / derivative references computed without the component under test.
    // ====================================================================

    struct GeluTrait
    {
        static constexpr ActivationType type = ActivationType::Gelu;
        static constexpr float kScale = 0.7978845608f;
        static constexpr float kCoeff = 0.044715f;

        static float fwd( float x )
        {
            return 0.5f * x * (1.0f + std::tanh( kScale * (x + kCoeff * x * x * x) ));
        }

        static float df( float x )
        {
            float x2 = x * x;
            float arg = kScale * (x + kCoeff * x * x2);
            float t = std::tanh( arg );
            float darg = kScale * (1.0f + 3.0f * kCoeff * x2);
            return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * darg;
        }
    };

    struct SiluTrait
    {
        static constexpr ActivationType type = ActivationType::Silu;

        static float fwd( float x ) { return x / (1.0f + std::exp( -x )); }
        static float df( float x )
        {
            float s = 1.0f / (1.0f + std::exp( -x ));
            return s * (1.0f + x * (1.0f - s));
        }
    };

    struct ReluTrait
    {
        static constexpr ActivationType type = ActivationType::Relu;

        static float fwd( float x ) { return x > 0.0f ? x : 0.0f; }
        static float df( float x ) { return x > 0.0f ? 1.0f : 0.0f; }
    };

    struct TanhTrait
    {
        static constexpr ActivationType type = ActivationType::Tanh;

        static float fwd( float x ) { return std::tanh( x ); }
        static float df( float x )
        {
            float t = std::tanh( x );
            return 1.0f - t * t;
        }
    };

    struct SigmoidTrait
    {
        static constexpr ActivationType type = ActivationType::Sigmoid;

        static float fwd( float x ) { return 1.0f / (1.0f + std::exp( -x )); }
        static float df( float x )
        {
            float s = 1.0f / (1.0f + std::exp( -x ));
            return s * (1.0f - s);
        }
    };

    struct LeakyReluTrait
    {
        static constexpr ActivationType type = ActivationType::LeakyRelu;
        static constexpr float kAlpha = 0.01f;

        static float fwd( float x ) { return x > 0.0f ? x : kAlpha * x; }
        static float df( float x ) { return x > 0.0f ? 1.0f : kAlpha; }
    };

    struct MishTrait
    {
        static constexpr ActivationType type = ActivationType::Mish;

        static float softplus( float x ) { return x > 20.0f ? x : std::log1p( std::exp( x ) ); }
        static float fwd( float x ) { return x * std::tanh( softplus( x ) ); }
        static float df( float x )
        {
            float t = std::tanh( softplus( x ) );
            float s = 1.0f / (1.0f + std::exp( -x ));
            return t + x * (1.0f - t * t) * s;
        }
    };

    using ActivationTraits = ::testing::Types<
        GeluTrait, SiluTrait, ReluTrait, TanhTrait, SigmoidTrait, LeakyReluTrait, MishTrait>;

    template<typename TTrait>
    class ActivationCpuTests : public ::testing::Test
    {
    protected:
        using ActivationType_ = Mila::Dnn::Activation<DeviceType::Cpu, TensorDataType::FP32, TTrait::type>;

        std::unique_ptr<ActivationType_> built( const shape_t& shape, RuntimeMode mode )
        {
            auto act = std::make_unique<ActivationType_>( "act", ActivationConfig( TTrait::type ), Device::Cpu() );
            act->build( BuildContext( shape, mode ) );

            return act;
        }

        static void fillSpread( TensorFp32& t )
        {
            auto* data = t.data();

            for ( dim_t i = 0; i < t.size(); ++i )
            {
                data[ i ] = static_cast<float>( i ) / t.size() * 4.0f - 2.0f;
            }
        }
    };

    TYPED_TEST_SUITE( ActivationCpuTests, ActivationTraits );

    TYPED_TEST( ActivationCpuTests, GetType_IsActivation )
    {
        typename TestFixture::ActivationType_ act( "act", ActivationConfig( TypeParam::type ), Device::Cpu() );

        EXPECT_EQ( act.getType(), ComponentType::Activation );
        EXPECT_EQ( act.getActivationType(), TypeParam::type );
    }

    TYPED_TEST( ActivationCpuTests, Forward_ThrowsBeforeBuild )
    {
        typename TestFixture::ActivationType_ act( "act", ActivationConfig( TypeParam::type ), Device::Cpu() );
        TensorFp32 input( Device::Cpu(), shape_t{ 2, 4 } );

        EXPECT_THROW( act.forward( input ), std::runtime_error );
    }

    TYPED_TEST( ActivationCpuTests, Forward_MatchesReference )
    {
        const shape_t shape{ 2, 3, 4 };
        auto act = this->built( shape, RuntimeMode::Inference );

        TensorFp32 input( Device::Cpu(), shape );
        this->fillSpread( input );

        auto& output = act->forward( input );

        ASSERT_EQ( output.size(), input.size() );

        constexpr float tolerance = 1e-4f;

        for ( dim_t i = 0; i < input.size(); ++i )
        {
            const float expected = TypeParam::fwd( input.data()[ i ] );

            EXPECT_NEAR( output.data()[ i ], expected, tolerance )
                << "forward mismatch at index " << i << " input=" << input.data()[ i ];
        }
    }

    TYPED_TEST( ActivationCpuTests, Backward_MatchesGradientReference )
    {
        const shape_t shape{ 2, 3, 4 };
        auto act = this->built( shape, RuntimeMode::Training );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape );
        this->fillSpread( input );

        for ( dim_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = static_cast<float>( i + 1 ) * 0.1f;
        }

        act->forward( input );
        auto& input_grad = act->backward( input, output_grad );

        ASSERT_EQ( input_grad.size(), input.size() );

        constexpr float tolerance = 2e-3f;

        for ( dim_t i = 0; i < input.size(); ++i )
        {
            const float expected = TypeParam::df( input.data()[ i ] ) * output_grad.data()[ i ];

            EXPECT_NEAR( input_grad.data()[ i ], expected, tolerance )
                << "backward mismatch at index " << i << " input=" << input.data()[ i ];
        }
    }

    TYPED_TEST( ActivationCpuTests, Parameters_AreEmpty )
    {
        auto act = this->built( shape_t{ 2, 4 }, RuntimeMode::Inference );

        EXPECT_EQ( act->parameterCount(), 0 );
        EXPECT_TRUE( act->getParameters().empty() );
        EXPECT_TRUE( act->getGradients().empty() );
    }
}
