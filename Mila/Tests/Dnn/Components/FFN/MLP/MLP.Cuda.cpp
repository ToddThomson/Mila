/**
 * @file MLP.Cuda.cpp
 * @brief Concrete-component tests for MLP<DeviceType::Cuda, FP32>.
 *
 * The CUDA mirror of MLP.Cpu.cpp. MLP is a COMPOSITE: fc1 (Linear) -> Gelu -> fc2
 * (Linear), mapping in_features -> hidden_size -> in_features. Per the concrete-
 * component archetype (Specifications/Testing.md) this tests the MLP DELTA over the
 * Component / CompositeComponent base: construction/validation, the build lifecycle
 * and shape contract, the RuntimeMode/TrainingMode axis, forward shape/finiteness,
 * that gradients FLOW through the stack, and parameter/gradient aggregation.
 *
 * The composed forward NUMERIC reference is owned by MLP.Cpu.cpp; the leaf numerics
 * are owned by Linear.Cuda.cpp / Gelu.Cuda.cpp. Here the CUDA point is the wiring:
 * the GPT-2 MLP runs on CUDA in the Bard training sample (batch GEMM via cuBLASLt,
 * the BACKLOG:83 bias-epilogue path), so realistic feature sizes are used (cuBLASLt
 * has no algorithm for toy dims) and known small weights are uploaded so the forward
 * is finite and the gradient is non-zero.
 *
 * CUDA device tests -- skipped when no CUDA device is present.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::FFN
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using MlpCuda = Mila::Dnn::MLP<DeviceType::Cuda, TensorDataType::FP32>;
        using DeviceTensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        // Realistic feature sizes -- cuBLASLt has no algorithm for toy GEMM dims.
        constexpr int64_t kInFeatures = 64;
        constexpr int64_t kHidden = 128;

        bool allFinite( const HostTensor& t )
        {
            const auto* data = t.data();

            for ( dim_t i = 0; i < t.size(); ++i )
            {
                if ( !std::isfinite( data[ i ] ) )
                {
                    return false;
                }
            }

            return true;
        }

        bool anyNonZero( const HostTensor& t )
        {
            const auto* data = t.data();

            for ( dim_t i = 0; i < t.size(); ++i )
            {
                if ( data[ i ] != 0.0f )
                {
                    return true;
                }
            }

            return false;
        }

        static_assert( MlpCuda::getDeviceType() == DeviceType::Cuda );
        static_assert( MlpCuda::getPrecision() == TensorDataType::FP32 );
    }

    class MLPCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }
        }

        std::unique_ptr<MlpCuda> builtMlp(
            const shape_t& shape, int64_t in_features, int64_t hidden_size,
            bool has_bias, RuntimeMode mode )
        {
            MLPConfig config( in_features, hidden_size );
            config.withBias( has_bias );

            auto mlp = std::make_unique<MlpCuda>( "mlp", config, Device::Cuda( 0 ) );
            mlp->build( BuildContext( shape, mode, false ) );

            return mlp;
        }

        // Upload small deterministic weights (zero biases) so forward is finite and
        // the gradient is non-zero. Children are added fc1, gelu, fc2.
        void setSmallParameters( MlpCuda& mlp, bool has_bias )
        {
            auto children = mlp.getComponents();
            ASSERT_GE( children.size(), 3u );

            auto fc1 = children[ 0 ]->getParameters();
            auto fc2 = children[ 2 ]->getParameters();

            uploadSmall( fc1[ 0 ], 7, 3.0f );
            uploadSmall( fc2[ 0 ], 5, 2.0f );

            if ( has_bias )
            {
                uploadZero( fc1[ 1 ] );
                uploadZero( fc2[ 1 ] );
            }

            mlp.synchronize();
        }

        // Spread-filled device tensor (host-staged then copied).
        DeviceTensor spreadInput( const shape_t& shape )
        {
            HostTensor host( Device::Cpu(), shape );

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = static_cast<float>( i ) / host.size() * 2.0f - 1.0f;
            }

            DeviceTensor device( Device::Cuda( 0 ), shape );
            copy( host, device );

            return device;
        }

        // Device -> host copy for value assertions.
        HostTensor toHost( const DeviceTensor& device, MlpCuda& mlp )
        {
            HostTensor host( Device::Cpu(), device.shape() );
            copy( device, host );
            mlp.synchronize();

            return host;
        }

    private:
        template<typename Param>
        void uploadSmall( Param* param, int modulus, float center )
        {
            HostTensor host( Device::Cpu(), param->shape() );

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = 0.01f * ( static_cast<float>( i % modulus ) - center );
            }

            copy( host, *static_cast<DeviceTensor*>( param ) );
        }

        template<typename Param>
        void uploadZero( Param* param )
        {
            HostTensor host( Device::Cpu(), param->shape() );

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = 0.0f;
            }

            copy( host, *static_cast<DeviceTensor*>( param ) );
        }
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( MLPCudaTests, Construct_StandaloneSucceeds )
    {
        MLPConfig config( 16, 64 );
        MlpCuda mlp( "mlp", config, Device::Cuda( 0 ) );

        EXPECT_EQ( mlp.getName(), "mlp" );
        EXPECT_EQ( mlp.getDeviceId().type, DeviceType::Cuda );
    }

    TEST_F( MLPCudaTests, Construct_InvalidConfigThrows_ZeroInputFeatures )
    {
        MLPConfig bad( 0, 64 );

        EXPECT_THROW( MlpCuda( "mlp", bad, Device::Cuda( 0 ) ), std::invalid_argument );
    }

    TEST_F( MLPCudaTests, Construct_InvalidConfigThrows_ZeroHiddenSize )
    {
        MLPConfig bad( 16, 0 );

        EXPECT_THROW( MlpCuda( "mlp", bad, Device::Cuda( 0 ) ), std::invalid_argument );
    }

    TEST_F( MLPCudaTests, Construct_DeviceTypeMismatchThrows )
    {
        MLPConfig config( 16, 64 );

        EXPECT_THROW( MlpCuda( "mlp", config, Device::Cpu() ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle
    // ====================================================================

    TEST_F( MLPCudaTests, IsBuilt_FalseBeforeBuild )
    {
        MLPConfig config( 16, 64 );
        MlpCuda mlp( "mlp", config, Device::Cuda( 0 ) );

        EXPECT_FALSE( mlp.isBuilt() );
    }

    TEST_F( MLPCudaTests, Build_SetsIsBuilt )
    {
        auto mlp = builtMlp( shape_t{ 2, kInFeatures }, kInFeatures, kHidden, true, RuntimeMode::Inference );

        EXPECT_TRUE( mlp->isBuilt() );
    }

    TEST_F( MLPCudaTests, Build_ThrowsOnInputFeatureMismatch )
    {
        MLPConfig config( kInFeatures, kHidden );
        MlpCuda mlp( "mlp", config, Device::Cuda( 0 ) );

        // Last dim (2 * kInFeatures) != configured input_features.
        EXPECT_THROW( mlp.build( BuildContext( shape_t{ 2, 2 * kInFeatures }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    TEST_F( MLPCudaTests, Forward_ThrowsBeforeBuild )
    {
        MLPConfig config( kInFeatures, kHidden );
        MlpCuda mlp( "mlp", config, Device::Cuda( 0 ) );
        DeviceTensor input( Device::Cuda( 0 ), shape_t{ 2, kInFeatures } );

        EXPECT_THROW( mlp.forward( input ), std::runtime_error );
    }

    TEST_F( MLPCudaTests, Backward_ThrowsBeforeBuild )
    {
        MLPConfig config( kInFeatures, kHidden );
        MlpCuda mlp( "mlp", config, Device::Cuda( 0 ) );
        DeviceTensor input( Device::Cuda( 0 ), shape_t{ 2, kInFeatures } );
        DeviceTensor output_grad( Device::Cuda( 0 ), shape_t{ 2, kInFeatures } );

        EXPECT_THROW( mlp.backward( input, output_grad ), std::runtime_error );
    }

    TEST_F( MLPCudaTests, Backward_ThrowsWhenForwardNotCalled )
    {
        const shape_t shape{ 4, kInFeatures };
        auto mlp = builtMlp( shape, kInFeatures, kHidden, true, RuntimeMode::Training );

        auto input = spreadInput( shape );
        DeviceTensor output_grad( Device::Cuda( 0 ), shape );

        EXPECT_THROW( mlp->backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // C. Execution Context / Device
    // ====================================================================

    TEST_F( MLPCudaTests, Synchronize_Succeeds )
    {
        MLPConfig config( 16, 64 );
        MlpCuda mlp( "mlp", config, Device::Cuda( 0 ) );

        EXPECT_NO_THROW( mlp.synchronize() );
    }

    // ====================================================================
    // D. Runtime + Training Mode
    // ====================================================================

    TEST_F( MLPCudaTests, BuiltForTraining_IsTrainingMode )
    {
        auto mlp = builtMlp( shape_t{ 2, kInFeatures }, kInFeatures, kHidden, true, RuntimeMode::Training );

        EXPECT_TRUE( mlp->isTrainingMode() );
        EXPECT_FALSE( mlp->isInferenceMode() );
    }

    TEST_F( MLPCudaTests, BuiltForInference_IsInferenceMode )
    {
        auto mlp = builtMlp( shape_t{ 2, kInFeatures }, kInFeatures, kHidden, true, RuntimeMode::Inference );

        EXPECT_TRUE( mlp->isInferenceMode() );
        EXPECT_FALSE( mlp->isTrainingMode() );
    }

    TEST_F( MLPCudaTests, SetTrainingMode_TogglesWithoutThrowWhenBuiltForTraining )
    {
        auto mlp = builtMlp( shape_t{ 2, kInFeatures }, kInFeatures, kHidden, true, RuntimeMode::Training );

        EXPECT_NO_THROW( mlp->setTrainingMode( TrainingMode::Eval ) );
        EXPECT_NO_THROW( mlp->setTrainingMode( TrainingMode::Normal ) );
    }

    TEST_F( MLPCudaTests, SetTrainingMode_ThrowsWhenBuiltForInference )
    {
        auto mlp = builtMlp( shape_t{ 2, kInFeatures }, kInFeatures, kHidden, true, RuntimeMode::Inference );

        EXPECT_THROW( mlp->setTrainingMode( TrainingMode::Eval ), std::runtime_error );
    }

    TEST_F( MLPCudaTests, Backward_ThrowsWhenBuiltForInference )
    {
        const shape_t shape{ 4, kInFeatures };
        auto mlp = builtMlp( shape, kInFeatures, kHidden, true, RuntimeMode::Inference );
        setSmallParameters( *mlp, true );

        auto input = spreadInput( shape );
        DeviceTensor output_grad( Device::Cuda( 0 ), shape );

        mlp->forward( input );

        EXPECT_THROW( mlp->backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // E. Forward (shape + finiteness)
    // ====================================================================

    TEST_F( MLPCudaTests, Forward_PreservesInputShape )
    {
        const shape_t shape{ 2, 4, kInFeatures };
        auto mlp = builtMlp( shape, kInFeatures, kHidden, true, RuntimeMode::Inference );
        setSmallParameters( *mlp, true );

        auto input = spreadInput( shape );
        auto& output = mlp->forward( input );

        EXPECT_EQ( output.shape(), shape );
        EXPECT_EQ( output.size(), input.size() );
    }

    TEST_F( MLPCudaTests, Forward_ProducesFiniteOutput )
    {
        const shape_t shape{ 4, kInFeatures };
        auto mlp = builtMlp( shape, kInFeatures, kHidden, true, RuntimeMode::Inference );
        setSmallParameters( *mlp, true );

        auto input = spreadInput( shape );
        auto& output = mlp->forward( input );

        EXPECT_TRUE( allFinite( toHost( output, *mlp ) ) );
    }

    // ====================================================================
    // F. Backward (gradient flows through the stack)
    // ====================================================================

    TEST_F( MLPCudaTests, Backward_ReturnsFiniteNonZeroInputGradient )
    {
        const shape_t shape{ 4, kInFeatures };
        auto mlp = builtMlp( shape, kInFeatures, kHidden, true, RuntimeMode::Training );
        setSmallParameters( *mlp, true );

        auto input = spreadInput( shape );

        HostTensor host_grad( Device::Cpu(), shape );
        for ( dim_t i = 0; i < host_grad.size(); ++i )
        {
            host_grad.data()[ i ] = 0.1f * static_cast<float>( ( i % 7 ) + 1 );
        }

        DeviceTensor output_grad( Device::Cuda( 0 ), shape );
        copy( host_grad, output_grad );

        mlp->forward( input );
        auto& input_grad = mlp->backward( input, output_grad );

        EXPECT_EQ( input_grad.shape(), shape );

        const HostTensor host_input_grad = toHost( input_grad, *mlp );
        EXPECT_TRUE( allFinite( host_input_grad ) );
        EXPECT_TRUE( anyNonZero( host_input_grad ) ) << "expected non-zero gradient flowing through the stack";
    }

    // ====================================================================
    // G. Parameters & Gradients
    // ====================================================================

    TEST_F( MLPCudaTests, ParameterCount_WithBias )
    {
        auto mlp = builtMlp( shape_t{ 2, kInFeatures }, kInFeatures, kHidden, true, RuntimeMode::Inference );

        const size_t expected =
            static_cast<size_t>( kInFeatures * kHidden + kHidden ) +  // fc1 weight + bias
            static_cast<size_t>( kHidden * kInFeatures + kInFeatures ); // fc2 weight + bias

        EXPECT_EQ( mlp->parameterCount(), expected );
    }

    TEST_F( MLPCudaTests, ParameterCount_WithoutBias )
    {
        auto mlp = builtMlp( shape_t{ 2, kInFeatures }, kInFeatures, kHidden, false, RuntimeMode::Inference );

        const size_t expected =
            static_cast<size_t>( kInFeatures * kHidden ) +
            static_cast<size_t>( kHidden * kInFeatures );

        EXPECT_EQ( mlp->parameterCount(), expected );
    }

    TEST_F( MLPCudaTests, GetParameters_FourWithBias )
    {
        auto mlp = builtMlp( shape_t{ 2, kInFeatures }, kInFeatures, kHidden, true, RuntimeMode::Inference );

        // fc1 weight+bias, fc2 weight+bias (Gelu is stateless).
        EXPECT_EQ( mlp->getParameters().size(), 4u );
    }

    TEST_F( MLPCudaTests, GetParameters_TwoWithoutBias )
    {
        auto mlp = builtMlp( shape_t{ 2, kInFeatures }, kInFeatures, kHidden, false, RuntimeMode::Inference );

        EXPECT_EQ( mlp->getParameters().size(), 2u );
    }

    TEST_F( MLPCudaTests, GetGradients_EmptyWhenBuiltForInference )
    {
        auto mlp = builtMlp( shape_t{ 2, kInFeatures }, kInFeatures, kHidden, true, RuntimeMode::Inference );

        EXPECT_TRUE( mlp->getGradients().empty() );
    }

    TEST_F( MLPCudaTests, GetGradients_PresentWhenBuiltForTraining )
    {
        auto mlp = builtMlp( shape_t{ 2, kInFeatures }, kInFeatures, kHidden, true, RuntimeMode::Training );

        EXPECT_EQ( mlp->getGradients().size(), 4u );
    }

    TEST_F( MLPCudaTests, GetComponents_ReturnsThreeChildren )
    {
        auto mlp = builtMlp( shape_t{ 2, kInFeatures }, kInFeatures, kHidden, true, RuntimeMode::Inference );

        // fc1, gelu, fc2.
        EXPECT_EQ( mlp->getComponents().size(), 3u );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( MLPCudaTests, ToString_DescribesComponent )
    {
        MLPConfig config( 16, 64 );
        MlpCuda mlp( "my_mlp", config, Device::Cuda( 0 ) );

        const std::string text = mlp.toString();

        EXPECT_NE( text.find( "MLP" ), std::string::npos );
        EXPECT_NE( text.find( "my_mlp" ), std::string::npos );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TEST_F( MLPCudaTests, GetType_IsMlp )
    {
        MLPConfig config( 16, 64 );
        MlpCuda mlp( "mlp", config, Device::Cuda( 0 ) );

        EXPECT_EQ( mlp.getType(), ComponentType::Mlp );
    }
}
