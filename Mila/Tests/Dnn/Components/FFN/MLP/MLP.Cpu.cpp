/**
 * @file MLP.Cpu.cpp
 * @brief Concrete-component tests for MLP<DeviceType::Cpu, FP32>.
 *
 * MLP is a COMPOSITE: fc1 (Linear) -> Gelu -> fc2 (Linear), mapping
 * in_features -> hidden_size -> in_features. Per the concrete-component
 * archetype (Specifications/Testing.md) this tests only the MLP DELTA over the
 * Component / CompositeComponent base contract: construction/validation, the
 * build->forward->backward path, the composed forward numeric reference, the
 * stack gradient-flow check, and parameter/gradient aggregation across children.
 * The inherited base machinery is covered once in Core/Component.cpp and the
 * container contract once in Core/CompositeComponent.cpp -- not retested here.
 *
 * Forward reference: out = Linear2( GELU_tanh( Linear1(x) ) ), computed host-side
 * from the same known child weights. The leaf numerics (matmul, GELU) are proven
 * in Linear.Cpu.cpp / Gelu.Cpu.cpp; here the point is that the composition wires
 * them together correctly. Backward is verified by central finite differences of
 * the scalar loss <output, g> against MLP::backward's input gradient -- the
 * gradient actually flows through the whole stack.
 *
 * CPU device, so this rides the MILA_ENABLE_CUDA=OFF CI gate. CUDA-instantiation
 * tests (and the FP32/BF16 precision sweep) belong in MLP.Cuda.cpp.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::FFN
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using MlpCpu = Mila::Dnn::MLP<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        // Deterministic child parameter values, independent of the component under
        // test. fc1 and fc2 use distinct generators so a swapped-layer wiring bug
        // would change the output.
        float weight1Value( int64_t out_index, int64_t in_index )
        {
            return 0.05f * static_cast<float>( out_index + 1 ) - 0.03f * static_cast<float>( in_index );
        }

        float bias1Value( int64_t out_index )
        {
            return 0.1f * static_cast<float>( out_index ) - 0.05f;
        }

        float weight2Value( int64_t out_index, int64_t in_index )
        {
            return 0.04f * static_cast<float>( in_index + 1 ) - 0.02f * static_cast<float>( out_index );
        }

        float bias2Value( int64_t out_index )
        {
            return 0.2f - 0.07f * static_cast<float>( out_index );
        }

        // GELU tanh-approximation (the only method GeluConfig supports), matching
        // the reference in Gelu.Cpu.cpp.
        constexpr float kSqrt2OverPi = 0.7978845608f;
        constexpr float kGeluCoeff = 0.044715f;

        float geluReference( float x )
        {
            const float x_cubed = x * x * x;

            return 0.5f * x * ( 1.0f + std::tanh( kSqrt2OverPi * ( x + kGeluCoeff * x_cubed ) ) );
        }

        // Host reference for the full MLP forward: out = Linear2( GELU( Linear1(x) ) ).
        // W1 is [hidden, in], W2 is [in, hidden]; X and Y are [batch, in].
        void referenceForward(
            const float* X,
            const float* W1, const float* B1,
            const float* W2, const float* B2,
            int64_t batch, int64_t in_features, int64_t hidden_size,
            std::vector<float>& Y )
        {
            std::vector<float> hidden( static_cast<size_t>( batch * hidden_size ) );

            for ( int64_t b = 0; b < batch; ++b )
            {
                for ( int64_t h = 0; h < hidden_size; ++h )
                {
                    double acc = B1 ? static_cast<double>( B1[ h ] ) : 0.0;

                    for ( int64_t i = 0; i < in_features; ++i )
                    {
                        acc += static_cast<double>( X[ b * in_features + i ] ) *
                            static_cast<double>( W1[ h * in_features + i ] );
                    }

                    hidden[ b * hidden_size + h ] = geluReference( static_cast<float>( acc ) );
                }
            }

            Y.assign( static_cast<size_t>( batch * in_features ), 0.0f );

            for ( int64_t b = 0; b < batch; ++b )
            {
                for ( int64_t o = 0; o < in_features; ++o )
                {
                    double acc = B2 ? static_cast<double>( B2[ o ] ) : 0.0;

                    for ( int64_t h = 0; h < hidden_size; ++h )
                    {
                        acc += static_cast<double>( hidden[ b * hidden_size + h ] ) *
                            static_cast<double>( W2[ o * hidden_size + h ] );
                    }

                    Y[ b * in_features + o ] = static_cast<float>( acc );
                }
            }
        }

        static_assert( MlpCpu::getDeviceType() == DeviceType::Cpu );
        static_assert( MlpCpu::getPrecision() == TensorDataType::FP32 );
    }

    class MLPCpuTests : public ::testing::Test
    {
    protected:
        std::unique_ptr<MlpCpu> builtMlp(
            const shape_t& shape, int64_t in_features, int64_t hidden_size,
            bool has_bias, RuntimeMode mode )
        {
            MLPConfig config( in_features, hidden_size );
            config.withBias( has_bias );

            auto mlp = std::make_unique<MlpCpu>( "mlp", config, Device::Cpu() );
            mlp->build( BuildContext( shape, mode, false ) );

            return mlp;
        }

        // Overwrite the (allocated) child weights and biases with deterministic
        // values. Children are added fc1, gelu, fc2 -- so getComponents()[0] is
        // fc1 and [2] is fc2; the Gelu at [1] is stateless.
        static void setKnownParameters(
            MlpCpu& mlp, int64_t in_features, int64_t hidden_size, bool has_bias )
        {
            auto children = mlp.getComponents();
            ASSERT_GE( children.size(), 3u );

            auto fc1_params = children[ 0 ]->getParameters();
            auto fc2_params = children[ 2 ]->getParameters();

            float* w1 = static_cast<float*>( fc1_params[ 0 ]->rawData() );
            for ( int64_t h = 0; h < hidden_size; ++h )
            {
                for ( int64_t i = 0; i < in_features; ++i )
                {
                    w1[ h * in_features + i ] = weight1Value( h, i );
                }
            }

            float* w2 = static_cast<float*>( fc2_params[ 0 ]->rawData() );
            for ( int64_t o = 0; o < in_features; ++o )
            {
                for ( int64_t h = 0; h < hidden_size; ++h )
                {
                    w2[ o * hidden_size + h ] = weight2Value( o, h );
                }
            }

            if ( has_bias )
            {
                float* b1 = static_cast<float*>( fc1_params[ 1 ]->rawData() );
                for ( int64_t h = 0; h < hidden_size; ++h )
                {
                    b1[ h ] = bias1Value( h );
                }

                float* b2 = static_cast<float*>( fc2_params[ 1 ]->rawData() );
                for ( int64_t o = 0; o < in_features; ++o )
                {
                    b2[ o ] = bias2Value( o );
                }
            }
        }

        static void buildKnownReferenceWeights(
            int64_t in_features, int64_t hidden_size, bool has_bias,
            std::vector<float>& w1, std::vector<float>& b1,
            std::vector<float>& w2, std::vector<float>& b2 )
        {
            w1.resize( static_cast<size_t>( hidden_size * in_features ) );
            w2.resize( static_cast<size_t>( in_features * hidden_size ) );

            for ( int64_t h = 0; h < hidden_size; ++h )
            {
                for ( int64_t i = 0; i < in_features; ++i )
                {
                    w1[ h * in_features + i ] = weight1Value( h, i );
                }
            }

            for ( int64_t o = 0; o < in_features; ++o )
            {
                for ( int64_t h = 0; h < hidden_size; ++h )
                {
                    w2[ o * hidden_size + h ] = weight2Value( o, h );
                }
            }

            if ( has_bias )
            {
                b1.resize( static_cast<size_t>( hidden_size ) );
                for ( int64_t h = 0; h < hidden_size; ++h )
                {
                    b1[ h ] = bias1Value( h );
                }

                b2.resize( static_cast<size_t>( in_features ) );
                for ( int64_t o = 0; o < in_features; ++o )
                {
                    b2[ o ] = bias2Value( o );
                }
            }
        }

        static void fillSpread( TensorFp32& t )
        {
            auto* data = t.data();

            for ( size_t i = 0; i < t.size(); ++i )
            {
                data[ i ] = static_cast<float>( i ) / t.size() * 2.0f - 1.0f;
            }
        }

        static int64_t leadingProduct( const shape_t& shape )
        {
            int64_t batch = 1;

            for ( size_t i = 0; i + 1 < shape.size(); ++i )
            {
                batch *= shape[ i ];
            }

            return batch;
        }
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( MLPCpuTests, Construct_StandaloneSucceeds )
    {
        MLPConfig config( 16, 64 );
        MlpCpu mlp( "mlp", config, Device::Cpu() );

        EXPECT_EQ( mlp.getName(), "mlp" );
        EXPECT_EQ( mlp.getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( MLPCpuTests, Construct_InvalidConfigThrows_ZeroInputFeatures )
    {
        MLPConfig bad( 0, 64 );

        EXPECT_THROW( MlpCpu( "mlp", bad, Device::Cpu() ), std::invalid_argument );
    }

    TEST_F( MLPCpuTests, Construct_InvalidConfigThrows_ZeroHiddenSize )
    {
        MLPConfig bad( 16, 0 );

        EXPECT_THROW( MlpCpu( "mlp", bad, Device::Cpu() ), std::invalid_argument );
    }

    TEST_F( MLPCpuTests, Construct_DeviceTypeMismatchThrows )
    {
        MLPConfig config( 16, 64 );

        EXPECT_THROW( MlpCpu( "mlp", config, Device::Cuda( 0 ) ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle
    // ====================================================================

    TEST_F( MLPCpuTests, IsBuilt_FalseBeforeBuild )
    {
        MLPConfig config( 16, 64 );
        MlpCpu mlp( "mlp", config, Device::Cpu() );

        EXPECT_FALSE( mlp.isBuilt() );
    }

    TEST_F( MLPCpuTests, Build_SetsIsBuilt )
    {
        auto mlp = builtMlp( shape_t{ 2, 16 }, 16, 64, true, RuntimeMode::Inference );

        EXPECT_TRUE( mlp->isBuilt() );
    }

    TEST_F( MLPCpuTests, Build_ThrowsOnInputFeatureMismatch )
    {
        MLPConfig config( 16, 64 );
        MlpCpu mlp( "mlp", config, Device::Cpu() );

        // Last dim (32) != configured input_features (16).
        EXPECT_THROW( mlp.build( BuildContext( shape_t{ 2, 32 }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    TEST_F( MLPCpuTests, Forward_ThrowsBeforeBuild )
    {
        MLPConfig config( 16, 64 );
        MlpCpu mlp( "mlp", config, Device::Cpu() );
        TensorFp32 input( Device::Cpu(), shape_t{ 2, 16 } );

        EXPECT_THROW( mlp.forward( input ), std::runtime_error );
    }

    TEST_F( MLPCpuTests, Backward_ThrowsBeforeBuild )
    {
        MLPConfig config( 16, 64 );
        MlpCpu mlp( "mlp", config, Device::Cpu() );
        TensorFp32 input( Device::Cpu(), shape_t{ 2, 16 } );
        TensorFp32 output_grad( Device::Cpu(), shape_t{ 2, 16 } );

        EXPECT_THROW( mlp.backward( input, output_grad ), std::runtime_error );
    }

    TEST_F( MLPCpuTests, Backward_ThrowsWhenForwardNotCalled )
    {
        const int64_t in_features = 3;
        const int64_t hidden_size = 4;
        const shape_t shape{ 2, in_features };

        auto mlp = builtMlp( shape, in_features, hidden_size, true, RuntimeMode::Training );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape );
        fillSpread( input );

        EXPECT_THROW( mlp->backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // C. Execution Context / Device
    // ====================================================================

    TEST_F( MLPCpuTests, Synchronize_Succeeds )
    {
        MLPConfig config( 16, 64 );
        MlpCpu mlp( "mlp", config, Device::Cpu() );

        EXPECT_NO_THROW( mlp.synchronize() );
    }

    // ====================================================================
    // D. Runtime + Training Mode
    // ====================================================================

    TEST_F( MLPCpuTests, BuiltForTraining_IsTrainingMode )
    {
        auto mlp = builtMlp( shape_t{ 2, 16 }, 16, 64, true, RuntimeMode::Training );

        EXPECT_TRUE( mlp->isTrainingMode() );
        EXPECT_FALSE( mlp->isInferenceMode() );
    }

    TEST_F( MLPCpuTests, BuiltForInference_IsInferenceMode )
    {
        auto mlp = builtMlp( shape_t{ 2, 16 }, 16, 64, true, RuntimeMode::Inference );

        EXPECT_TRUE( mlp->isInferenceMode() );
        EXPECT_FALSE( mlp->isTrainingMode() );
    }

    TEST_F( MLPCpuTests, SetTrainingMode_TogglesWithoutThrowWhenBuiltForTraining )
    {
        auto mlp = builtMlp( shape_t{ 2, 16 }, 16, 64, true, RuntimeMode::Training );

        EXPECT_NO_THROW( mlp->setTrainingMode( TrainingMode::Eval ) );
        EXPECT_NO_THROW( mlp->setTrainingMode( TrainingMode::Normal ) );
    }

    TEST_F( MLPCpuTests, SetTrainingMode_ThrowsWhenBuiltForInference )
    {
        auto mlp = builtMlp( shape_t{ 2, 16 }, 16, 64, true, RuntimeMode::Inference );

        EXPECT_THROW( mlp->setTrainingMode( TrainingMode::Eval ), std::runtime_error );
    }

    TEST_F( MLPCpuTests, Backward_ThrowsWhenBuiltForInference )
    {
        const int64_t in_features = 3;
        const int64_t hidden_size = 4;
        const shape_t shape{ 2, in_features };

        auto mlp = builtMlp( shape, in_features, hidden_size, true, RuntimeMode::Inference );
        setKnownParameters( *mlp, in_features, hidden_size, true );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape );
        fillSpread( input );

        mlp->forward( input );

        EXPECT_THROW( mlp->backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // E. Forward (numeric vs composed reference)
    // ====================================================================

    TEST_F( MLPCpuTests, Forward_PreservesInputShape )
    {
        const shape_t shape{ 2, 5, 8 };
        auto mlp = builtMlp( shape, 8, 16, true, RuntimeMode::Inference );

        TensorFp32 input( Device::Cpu(), shape );
        fillSpread( input );

        auto& output = mlp->forward( input );

        EXPECT_EQ( output.shape(), shape );
        EXPECT_EQ( output.size(), input.size() );
    }

    TEST_F( MLPCpuTests, Forward_MatchesComposedReference )
    {
        const int64_t in_features = 3;
        const int64_t hidden_size = 4;
        const shape_t shape{ 2, 3, in_features };

        auto mlp = builtMlp( shape, in_features, hidden_size, true, RuntimeMode::Inference );
        setKnownParameters( *mlp, in_features, hidden_size, true );

        TensorFp32 input( Device::Cpu(), shape );
        fillSpread( input );

        auto& output = mlp->forward( input );

        const int64_t batch = leadingProduct( shape );
        ASSERT_EQ( output.size(), static_cast<size_t>( batch * in_features ) );

        std::vector<float> w1, b1, w2, b2;
        buildKnownReferenceWeights( in_features, hidden_size, true, w1, b1, w2, b2 );

        std::vector<float> expected;
        referenceForward( input.data(), w1.data(), b1.data(), w2.data(), b2.data(),
            batch, in_features, hidden_size, expected );

        constexpr float tolerance = 1e-4f;

        for ( size_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], expected[ i ], tolerance )
                << "forward mismatch at index " << i;
        }
    }

    TEST_F( MLPCpuTests, Forward_MatchesComposedReferenceWithoutBias )
    {
        const int64_t in_features = 3;
        const int64_t hidden_size = 4;
        const shape_t shape{ 2, in_features };

        auto mlp = builtMlp( shape, in_features, hidden_size, false, RuntimeMode::Inference );
        setKnownParameters( *mlp, in_features, hidden_size, false );

        TensorFp32 input( Device::Cpu(), shape );
        fillSpread( input );

        auto& output = mlp->forward( input );

        const int64_t batch = leadingProduct( shape );

        std::vector<float> w1, b1, w2, b2;
        buildKnownReferenceWeights( in_features, hidden_size, false, w1, b1, w2, b2 );

        std::vector<float> expected;
        referenceForward( input.data(), w1.data(), nullptr, w2.data(), nullptr,
            batch, in_features, hidden_size, expected );

        constexpr float tolerance = 1e-4f;

        for ( size_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], expected[ i ], tolerance )
                << "no-bias forward mismatch at index " << i;
        }
    }

    // ====================================================================
    // F. Backward (gradient flows through the stack)
    // ====================================================================

    TEST_F( MLPCpuTests, Backward_InputGradientMatchesFiniteDifference )
    {
        const int64_t in_features = 3;
        const int64_t hidden_size = 4;
        const shape_t shape{ 2, in_features };
        const int64_t batch = leadingProduct( shape );

        auto mlp = builtMlp( shape, in_features, hidden_size, true, RuntimeMode::Training );
        setKnownParameters( *mlp, in_features, hidden_size, true );

        TensorFp32 input( Device::Cpu(), shape );
        fillSpread( input );

        // Fixed upstream gradient g = dL/doutput for the scalar loss L = <output, g>.
        std::vector<float> g( static_cast<size_t>( batch * in_features ) );
        for ( size_t i = 0; i < g.size(); ++i )
        {
            g[ i ] = 0.1f * static_cast<float>( i + 1 );
        }

        TensorFp32 output_grad( Device::Cpu(), shape );
        for ( size_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = g[ i ];
        }

        // Analytic input gradient via backward.
        mlp->forward( input );
        auto& input_grad = mlp->backward( input, output_grad );
        ASSERT_EQ( input_grad.shape(), shape );

        std::vector<float> analytic( input_grad.size() );
        for ( size_t i = 0; i < analytic.size(); ++i )
        {
            analytic[ i ] = input_grad.data()[ i ];
        }

        // Host reference weights for the finite-difference forward evaluations.
        std::vector<float> w1, b1, w2, b2;
        buildKnownReferenceWeights( in_features, hidden_size, true, w1, b1, w2, b2 );

        auto lossAt = [&]( const std::vector<float>& x ) -> double
            {
                std::vector<float> y;
                referenceForward( x.data(), w1.data(), b1.data(), w2.data(), b2.data(),
                    batch, in_features, hidden_size, y );

                double loss = 0.0;
                for ( size_t i = 0; i < y.size(); ++i )
                {
                    loss += static_cast<double>( y[ i ] ) * static_cast<double>( g[ i ] );
                }

                return loss;
            };

        std::vector<float> base( input.size() );
        for ( size_t i = 0; i < base.size(); ++i )
        {
            base[ i ] = input.data()[ i ];
        }

        constexpr float eps = 1e-3f;
        constexpr float tolerance = 1e-2f;

        for ( size_t k = 0; k < base.size(); ++k )
        {
            std::vector<float> plus = base;
            std::vector<float> minus = base;
            plus[ k ] += eps;
            minus[ k ] -= eps;

            double numeric = ( lossAt( plus ) - lossAt( minus ) ) / ( 2.0 * eps );

            EXPECT_NEAR( analytic[ k ], static_cast<float>( numeric ), tolerance )
                << "input-gradient finite-difference mismatch at index " << k;
        }
    }

    TEST_F( MLPCpuTests, Backward_ProducesNonZeroParameterGradients )
    {
        const int64_t in_features = 3;
        const int64_t hidden_size = 4;
        const shape_t shape{ 2, in_features };

        auto mlp = builtMlp( shape, in_features, hidden_size, true, RuntimeMode::Training );
        setKnownParameters( *mlp, in_features, hidden_size, true );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape );
        fillSpread( input );
        for ( size_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = 0.1f * static_cast<float>( i + 1 );
        }

        mlp->zeroGradients();
        mlp->forward( input );
        mlp->backward( input, output_grad );

        auto grads = mlp->getGradients();
        ASSERT_EQ( grads.size(), 4u );

        bool any_nonzero = false;
        for ( const auto& grad : grads )
        {
            const float* g = static_cast<const float*>( grad->rawData() );
            for ( size_t i = 0; i < grad->size(); ++i )
            {
                if ( g[ i ] != 0.0f )
                {
                    any_nonzero = true;
                    break;
                }
            }
        }

        EXPECT_TRUE( any_nonzero ) << "expected non-zero parameter gradients after backward";
    }

    // ====================================================================
    // G. Parameters & Gradients
    // ====================================================================

    TEST_F( MLPCpuTests, ParameterCount_WithBias )
    {
        const int64_t in_features = 16;
        const int64_t hidden_size = 64;
        auto mlp = builtMlp( shape_t{ 2, in_features }, in_features, hidden_size, true, RuntimeMode::Inference );

        const size_t expected =
            static_cast<size_t>( in_features * hidden_size + hidden_size ) +  // fc1 weight + bias
            static_cast<size_t>( hidden_size * in_features + in_features );    // fc2 weight + bias

        EXPECT_EQ( mlp->parameterCount(), expected );
    }

    TEST_F( MLPCpuTests, ParameterCount_WithoutBias )
    {
        const int64_t in_features = 16;
        const int64_t hidden_size = 64;
        auto mlp = builtMlp( shape_t{ 2, in_features }, in_features, hidden_size, false, RuntimeMode::Inference );

        const size_t expected =
            static_cast<size_t>( in_features * hidden_size ) +
            static_cast<size_t>( hidden_size * in_features );

        EXPECT_EQ( mlp->parameterCount(), expected );
    }

    TEST_F( MLPCpuTests, GetParameters_FourWithBias )
    {
        auto mlp = builtMlp( shape_t{ 2, 16 }, 16, 64, true, RuntimeMode::Inference );

        // fc1 weight+bias, fc2 weight+bias (Gelu is stateless).
        EXPECT_EQ( mlp->getParameters().size(), 4u );
    }

    TEST_F( MLPCpuTests, GetParameters_TwoWithoutBias )
    {
        auto mlp = builtMlp( shape_t{ 2, 16 }, 16, 64, false, RuntimeMode::Inference );

        EXPECT_EQ( mlp->getParameters().size(), 2u );
    }

    TEST_F( MLPCpuTests, GetGradients_EmptyWhenBuiltForInference )
    {
        auto mlp = builtMlp( shape_t{ 2, 16 }, 16, 64, true, RuntimeMode::Inference );

        EXPECT_TRUE( mlp->getGradients().empty() );
    }

    TEST_F( MLPCpuTests, GetGradients_PresentWhenBuiltForTraining )
    {
        auto mlp = builtMlp( shape_t{ 2, 16 }, 16, 64, true, RuntimeMode::Training );

        EXPECT_EQ( mlp->getGradients().size(), 4u );
    }

    TEST_F( MLPCpuTests, GetComponents_ReturnsThreeChildren )
    {
        auto mlp = builtMlp( shape_t{ 2, 16 }, 16, 64, true, RuntimeMode::Inference );

        // fc1, gelu, fc2.
        EXPECT_EQ( mlp->getComponents().size(), 3u );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( MLPCpuTests, ToString_DescribesComponent )
    {
        MLPConfig config( 16, 64 );
        MlpCpu mlp( "my_mlp", config, Device::Cpu() );

        const std::string text = mlp.toString();

        EXPECT_NE( text.find( "MLP" ), std::string::npos );
        EXPECT_NE( text.find( "my_mlp" ), std::string::npos );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TEST_F( MLPCpuTests, GetType_IsMlp )
    {
        MLPConfig config( 16, 64 );
        MlpCpu mlp( "mlp", config, Device::Cpu() );

        EXPECT_EQ( mlp.getType(), ComponentType::Mlp );
    }
}
