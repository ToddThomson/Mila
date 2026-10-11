/**
 * @file MLP.Cuda.cpp
 * @brief Concrete-component tests for MLP<DeviceType::Cuda, FP32>.
 *
 * MLP is a composite: fc1 (Linear) -> Gelu -> fc2 (Linear), mapping in_features -> hidden_size -> in_features.
 * Forward and backward are checked against a double-precision host reference built from known parameters;
 * feature sizes are realistic because cuBLASLt has no algorithm for toy GEMM shapes.
 *
 * CUDA device tests -- skipped when no CUDA device is present.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

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

        // Known parameter values; fc1 and fc2 use distinct generators so swapped layers change the output.
        float weight1Value( int64_t out_index, int64_t in_index )
        {
            return 0.1f * ( static_cast<float>( ( out_index * 13 + in_index * 7 ) % 17 ) - 8.0f ) / 17.0f;
        }

        float bias1Value( int64_t out_index )
        {
            return 0.02f * ( static_cast<float>( out_index % 5 ) - 2.0f );
        }

        float weight2Value( int64_t out_index, int64_t in_index )
        {
            return 0.1f * ( static_cast<float>( ( out_index * 11 + in_index * 5 ) % 19 ) - 9.0f ) / 19.0f;
        }

        float bias2Value( int64_t out_index )
        {
            return 0.05f * ( static_cast<float>( out_index % 3 ) - 1.0f );
        }

        constexpr double kSqrt2OverPi = 0.7978845608;
        constexpr double kGeluCoefficient = 0.044715;

        double geluReference( double x )
        {
            return 0.5 * x * ( 1.0 + std::tanh( kSqrt2OverPi * ( x + kGeluCoefficient * x * x * x ) ) );
        }

        double geluGradientReference( double x )
        {
            const double x_squared = x * x;
            const double tanh_value = std::tanh( kSqrt2OverPi * ( x + kGeluCoefficient * x * x_squared ) );
            const double sech_squared = 1.0 - tanh_value * tanh_value;

            return 0.5 * ( 1.0 + tanh_value ) + 0.5 * x * sech_squared * kSqrt2OverPi * ( 1.0 + 3.0 * kGeluCoefficient * x_squared );
        }

        // out = fc2( gelu( fc1( x ) ) ). Fills H (pre-activation) and A (post-activation); returns Y.
        std::vector<double> referenceForward(
            const float* x, int64_t rows, bool has_bias, std::vector<double>& h, std::vector<double>& a )
        {
            h.assign( static_cast<size_t>( rows * kHidden ), 0.0 );
            a.assign( static_cast<size_t>( rows * kHidden ), 0.0 );
            std::vector<double> y( static_cast<size_t>( rows * kInFeatures ), 0.0 );

            for ( int64_t r = 0; r < rows; ++r )
            {
                for ( int64_t j = 0; j < kHidden; ++j )
                {
                    double accumulator = has_bias ? bias1Value( j ) : 0.0;

                    for ( int64_t i = 0; i < kInFeatures; ++i )
                    {
                        accumulator += static_cast<double>( x[ r * kInFeatures + i ] ) * weight1Value( j, i );
                    }

                    h[ r * kHidden + j ] = accumulator;
                    a[ r * kHidden + j ] = geluReference( accumulator );
                }

                for ( int64_t k = 0; k < kInFeatures; ++k )
                {
                    double accumulator = has_bias ? bias2Value( k ) : 0.0;

                    for ( int64_t j = 0; j < kHidden; ++j )
                    {
                        accumulator += a[ r * kHidden + j ] * weight2Value( k, j );
                    }

                    y[ r * kInFeatures + k ] = accumulator;
                }
            }

            return y;
        }
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

        // Upload the known parameters the host reference uses.
        void setKnownParameters( MlpCuda& mlp, bool has_bias )
        {
            auto children = mlp.getComponents();
            ASSERT_GE( children.size(), 3u );

            auto fc1 = children[ 0 ]->getParameters();
            auto fc2 = children[ 2 ]->getParameters();

            HostTensor w1( Device::Cpu(), shape_t{ kHidden, kInFeatures } );
            HostTensor b1( Device::Cpu(), shape_t{ kHidden } );
            HostTensor w2( Device::Cpu(), shape_t{ kInFeatures, kHidden } );
            HostTensor b2( Device::Cpu(), shape_t{ kInFeatures } );

            for ( int64_t j = 0; j < kHidden; ++j )
            {
                for ( int64_t i = 0; i < kInFeatures; ++i )
                {
                    w1.data()[ j * kInFeatures + i ] = weight1Value( j, i );
                    w2.data()[ i * kHidden + j ] = weight2Value( i, j );
                }

                b1.data()[ j ] = bias1Value( j );
            }

            for ( int64_t k = 0; k < kInFeatures; ++k )
            {
                b2.data()[ k ] = bias2Value( k );
            }

            copy( w1, *static_cast<DeviceTensor*>( fc1[ 0 ] ) );
            copy( w2, *static_cast<DeviceTensor*>( fc2[ 0 ] ) );

            if ( has_bias )
            {
                copy( b1, *static_cast<DeviceTensor*>( fc1[ 1 ] ) );
                copy( b2, *static_cast<DeviceTensor*>( fc2[ 1 ] ) );
            }

            mlp.synchronize();
        }

        HostTensor spreadHost( const shape_t& shape )
        {
            HostTensor host( Device::Cpu(), shape );

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = static_cast<float>( i ) / host.size() * 2.0f - 1.0f;
            }

            return host;
        }

        HostTensor toHost( const ITensor* device, MlpCuda& mlp )
        {
            return toHost( *static_cast<const DeviceTensor*>( device ), mlp );
        }

        // Forward against the host reference, with or without biases.
        void expectForwardMatchesReference( bool has_bias )
        {
            const shape_t shape{ 2, 4, kInFeatures };
            auto mlp = builtMlp( shape, kInFeatures, kHidden, has_bias, RuntimeMode::Inference );
            setKnownParameters( *mlp, has_bias );

            auto host_in = spreadHost( shape );
            DeviceTensor input( Device::Cuda( 0 ), shape );
            copy( host_in, input );

            auto output = toHost( mlp->forward( input ), *mlp );

            std::vector<double> h, a;
            const auto expected = referenceForward( host_in.data(), 2 * 4, has_bias, h, a );

            ASSERT_EQ( output.size(), static_cast<dim_t>( expected.size() ) );

            for ( dim_t i = 0; i < output.size(); ++i )
            {
                EXPECT_NEAR( output.data()[ i ], static_cast<float>( expected[ i ] ), 1e-3f )
                    << "forward mismatch at index " << i;
            }
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

    TEST_F( MLPCudaTests, Forward_MatchesComposedReference )
    {
        expectForwardMatchesReference( true );
    }

    TEST_F( MLPCudaTests, Forward_MatchesComposedReferenceWithoutBias )
    {
        expectForwardMatchesReference( false );
    }

    // The input gradient and all four parameter gradients against the chain rule on the host.
    TEST_F( MLPCudaTests, Backward_MatchesReferenceGradients )
    {
        constexpr int64_t kRows = 4;
        const shape_t shape{ kRows, kInFeatures };

        auto mlp = builtMlp( shape, kInFeatures, kHidden, true, RuntimeMode::Training );
        setKnownParameters( *mlp, true );

        auto host_in = spreadHost( shape );

        // Non-uniform and signed, so a layer that ignored its output gradient fails.
        HostTensor host_grad( Device::Cpu(), shape );
        for ( dim_t i = 0; i < host_grad.size(); ++i )
        {
            host_grad.data()[ i ] = 0.05f * static_cast<float>( ( i % 7 ) + 1 ) - 0.15f;
        }

        DeviceTensor input( Device::Cuda( 0 ), shape );
        DeviceTensor output_grad( Device::Cuda( 0 ), shape );
        copy( host_in, input );
        copy( host_grad, output_grad );

        mlp->zeroGradients();
        mlp->forward( input );
        auto input_grad = toHost( mlp->backward( input, output_grad ), *mlp );

        std::vector<double> h, a;
        referenceForward( host_in.data(), kRows, true, h, a );

        const float* x = host_in.data();
        const float* dy = host_grad.data();

        // dA = dY W2 ; dH = dA * gelu'(H)
        std::vector<double> dh( static_cast<size_t>( kRows * kHidden ), 0.0 );

        for ( int64_t r = 0; r < kRows; ++r )
        {
            for ( int64_t j = 0; j < kHidden; ++j )
            {
                double da = 0.0;

                for ( int64_t k = 0; k < kInFeatures; ++k )
                {
                    da += static_cast<double>( dy[ r * kInFeatures + k ] ) * weight2Value( k, j );
                }

                dh[ r * kHidden + j ] = da * geluGradientReference( h[ r * kHidden + j ] );
            }
        }

        auto tolerance = []( double expected ) { return 1e-3f + 1e-3f * static_cast<float>( std::fabs( expected ) ); };

        for ( int64_t r = 0; r < kRows; ++r )
        {
            for ( int64_t i = 0; i < kInFeatures; ++i )
            {
                double expected = 0.0;

                for ( int64_t j = 0; j < kHidden; ++j )
                {
                    expected += dh[ r * kHidden + j ] * weight1Value( j, i );
                }

                EXPECT_NEAR( input_grad.data()[ r * kInFeatures + i ], static_cast<float>( expected ), tolerance( expected ) )
                    << "dX mismatch at row " << r << " feature " << i;
            }
        }

        // getGradients() is [fc1 weight, fc1 bias, fc2 weight, fc2 bias].
        auto grads = mlp->getGradients();
        ASSERT_EQ( grads.size(), 4u );

        auto dw1 = toHost( grads[ 0 ], *mlp );
        auto db1 = toHost( grads[ 1 ], *mlp );
        auto dw2 = toHost( grads[ 2 ], *mlp );
        auto db2 = toHost( grads[ 3 ], *mlp );

        for ( int64_t j = 0; j < kHidden; ++j )
        {
            double expected_bias = 0.0;

            for ( int64_t r = 0; r < kRows; ++r )
            {
                expected_bias += dh[ r * kHidden + j ];
            }

            EXPECT_NEAR( db1.data()[ j ], static_cast<float>( expected_bias ), tolerance( expected_bias ) ) << "fc1 dB at " << j;

            for ( int64_t i = 0; i < kInFeatures; ++i )
            {
                double expected_weight = 0.0;

                for ( int64_t r = 0; r < kRows; ++r )
                {
                    expected_weight += dh[ r * kHidden + j ] * static_cast<double>( x[ r * kInFeatures + i ] );
                }

                EXPECT_NEAR( dw1.data()[ j * kInFeatures + i ], static_cast<float>( expected_weight ), tolerance( expected_weight ) )
                    << "fc1 dW at " << j << ", " << i;
            }
        }

        for ( int64_t k = 0; k < kInFeatures; ++k )
        {
            double expected_bias = 0.0;

            for ( int64_t r = 0; r < kRows; ++r )
            {
                expected_bias += static_cast<double>( dy[ r * kInFeatures + k ] );
            }

            EXPECT_NEAR( db2.data()[ k ], static_cast<float>( expected_bias ), tolerance( expected_bias ) ) << "fc2 dB at " << k;

            for ( int64_t j = 0; j < kHidden; ++j )
            {
                double expected_weight = 0.0;

                for ( int64_t r = 0; r < kRows; ++r )
                {
                    expected_weight += static_cast<double>( dy[ r * kInFeatures + k ] ) * a[ r * kHidden + j ];
                }

                EXPECT_NEAR( dw2.data()[ k * kHidden + j ], static_cast<float>( expected_weight ), tolerance( expected_weight ) )
                    << "fc2 dW at " << k << ", " << j;
            }
        }
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
