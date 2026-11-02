/**
 * @file Network.Cuda.cpp
 * @brief The training spine on CUDA: a composite network's forward, backward, optimizer wiring and training loop.
 *
 * A Linear -> Gelu -> Linear network (the MnistClassifier shape) checked against a double-precision host
 * reference: backward through the composite lands each layer's parameter gradients, createOptimizer wires AdamW
 * to all of them, and a training loop reduces a regression loss. Feature sizes are realistic because cuBLASLt
 * finds no algorithm for toy GEMM shapes.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Modeling
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Optimizers;
    using namespace Mila::Dnn::Serialization;

    namespace
    {
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
            const double d_argument = kSqrt2OverPi * ( 1.0 + 3.0 * kGeluCoefficient * x_squared );

            return 0.5 * ( 1.0 + tanh_value ) + 0.5 * x * sech_squared * d_argument;
        }

        // Deterministic parameter values, independent of the component under test.
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

        template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
        class MlpUnderTest : public Network<TDeviceType, TPrecision>
        {
        public:
            using Base = Network<TDeviceType, TPrecision>;
            using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
            using TensorType = Tensor<TPrecision, MR>;
            using LinearType = Linear<TDeviceType, TPrecision>;
            using GeluType = Gelu<TDeviceType, TPrecision>;

            MlpUnderTest( const std::string& name,
                int64_t in_features, int64_t hidden_features, int64_t out_features,
                DeviceId device_id )
                : Base( name ),
                owned_context_( createExecutionContext( device_id ) ),
                in_features_( in_features ),
                hidden_features_( hidden_features ),
                out_features_( out_features )
            {
                if ( device_id.type != TDeviceType )
                {
                    throw std::invalid_argument(
                        std::format( "MlpUnderTest: device type mismatch: expected {}, got {}",
                            deviceTypeToString( TDeviceType ),
                            deviceTypeToString( device_id.type ) ) );
                }

                createGraph();

                this->setExecutionContext( owned_context_.get() );
            }

            TensorType& forward( const TensorType& input )
            {
                if ( !this->isBuilt() )
                {
                    throw std::runtime_error( "MlpUnderTest: must be built before forward" );
                }

                auto& hidden = fc1_->forward( input );
                fc1_out_ = &hidden;

                auto& activated = act_->forward( hidden );
                act_out_ = &activated;

                return fc2_->forward( activated );
            }

            TensorType& backward( const TensorType& input, const TensorType& output_grad )
            {
                if ( !this->isBuilt() )
                {
                    throw std::runtime_error( "MlpUnderTest: must be built before backward" );
                }

                auto& activated_grad = fc2_->backward( *act_out_, output_grad );
                auto& hidden_grad = act_->backward( *fc1_out_, activated_grad );

                return fc1_->backward( input, hidden_grad );
            }

            void zeroGradients() override
            {
                if ( !this->isBuilt() )
                {
                    return;
                }

                fc1_->zeroGradients();
                act_->zeroGradients();
                fc2_->zeroGradients();
            }

            MemoryStats getMemoryStats() const override
            {
                return {};
            }

            IExecutionContext* context() const { return owned_context_.get(); }
            const std::shared_ptr<LinearType>& linear1() const { return fc1_; }
            const std::shared_ptr<LinearType>& linear2() const { return fc2_; }

        protected:
            void onBuilding( const BuildContext& context ) override
            {
                const int64_t batch = context.inputShape()[ 0 ];

                fc1_ = this->template getComponentAs<LinearType>( this->getName() + ".fc1" );
                fc1_->build( context.withShape( shape_t{ batch, in_features_ } ) );

                act_ = this->template getComponentAs<GeluType>( this->getName() + ".act" );
                act_->build( context.withShape( shape_t{ batch, hidden_features_ } ) );

                fc2_ = this->template getComponentAs<LinearType>( this->getName() + ".fc2" );
                fc2_->build( context.withShape( shape_t{ batch, hidden_features_ } ) );
            }

            void save_( ModelArchive&, SerializationMode ) const override
            {
            }

        private:
            void createGraph()
            {
                this->addComponent( std::make_shared<LinearType>(
                    this->getName() + ".fc1",
                    LinearConfig( in_features_, hidden_features_ ).withBias( true ),
                    std::nullopt ) );

                this->addComponent( std::make_shared<GeluType>(
                    this->getName() + ".act", GeluConfig(), std::nullopt ) );

                this->addComponent( std::make_shared<LinearType>(
                    this->getName() + ".fc2",
                    LinearConfig( hidden_features_, out_features_ ).withBias( true ),
                    std::nullopt ) );
            }

            std::unique_ptr<IExecutionContext> owned_context_{ nullptr };

            int64_t in_features_;
            int64_t hidden_features_;
            int64_t out_features_;

            std::shared_ptr<LinearType> fc1_{ nullptr };
            std::shared_ptr<GeluType> act_{ nullptr };
            std::shared_ptr<LinearType> fc2_{ nullptr };

            TensorType* fc1_out_{ nullptr };
            TensorType* act_out_{ nullptr };
        };

        using MlpCuda = MlpUnderTest<DeviceType::Cuda, TensorDataType::FP32>;
        using DeviceFp32 = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
    }

    class MlpNetworkCudaTests : public ::testing::Test
    {
    protected:
        static constexpr int64_t kBatch = 8;
        static constexpr int64_t kIn = 64;
        static constexpr int64_t kHidden = 32;
        static constexpr int64_t kOut = 16;

        static constexpr float kForwardTolerance = 1e-3f;
        static constexpr float kBackwardAbsoluteTolerance = 1e-3f;
        static constexpr float kBackwardRelativeTolerance = 1e-3f;

        void SetUp() override
        {
            try
            {
                probe_context_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                probe_context_ = nullptr;
            }

            if ( !probe_context_ )
            {
                GTEST_SKIP() << "CUDA device not available";
            }
        }

        std::unique_ptr<MlpCuda> builtMlp( RuntimeMode mode )
        {
            auto net = std::make_unique<MlpCuda>( "mlp", kIn, kHidden, kOut, Device::Cuda( 0 ) );
            net->build( BuildContext( shape_t{ kBatch, kIn }, mode, false ) );

            return net;
        }

        static void upload( MlpCuda& net, const HostFp32& host, ITensor* device )
        {
            copy( host, *static_cast<DeviceFp32*>( device ), net.context() );
        }

        static void setKnownParameters( MlpCuda& net )
        {
            HostFp32 w1( Device::Cpu(), shape_t{ kHidden, kIn } );
            HostFp32 b1( Device::Cpu(), shape_t{ kHidden } );

            for ( int64_t j = 0; j < kHidden; ++j )
            {
                for ( int64_t i = 0; i < kIn; ++i )
                {
                    w1.data()[ j * kIn + i ] = weight1Value( j, i );
                }

                b1.data()[ j ] = bias1Value( j );
            }

            HostFp32 w2( Device::Cpu(), shape_t{ kOut, kHidden } );
            HostFp32 b2( Device::Cpu(), shape_t{ kOut } );

            for ( int64_t k = 0; k < kOut; ++k )
            {
                for ( int64_t j = 0; j < kHidden; ++j )
                {
                    w2.data()[ k * kHidden + j ] = weight2Value( k, j );
                }

                b2.data()[ k ] = bias2Value( k );
            }

            auto fc1_params = net.linear1()->getParameters();
            upload( net, w1, fc1_params[ 0 ] );
            upload( net, b1, fc1_params[ 1 ] );

            auto fc2_params = net.linear2()->getParameters();
            upload( net, w2, fc2_params[ 0 ] );
            upload( net, b2, fc2_params[ 1 ] );

            net.context()->synchronize();
        }

        static HostFp32 spreadInput()
        {
            HostFp32 host( Device::Cpu(), shape_t{ kBatch, kIn } );

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = static_cast<float>( i ) / host.size() * 4.0f - 2.0f;
            }

            return host;
        }

        static DeviceFp32 toDevice( MlpCuda& net, const HostFp32& host )
        {
            DeviceFp32 device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, net.context() );
            net.context()->synchronize();

            return device;
        }

        static HostFp32 toHostFp32( MlpCuda& net, const ITensor* device )
        {
            auto host = toHost<TensorDataType::FP32>( *static_cast<const DeviceFp32*>( device ), net.context() );
            net.context()->synchronize();

            return host;
        }

        // Returns Y[kBatch * kOut]; fills H and A (pre- and post-activation).
        static std::vector<double> referenceForward( const float* x, std::vector<double>& h, std::vector<double>& a )
        {
            h.assign( kBatch * kHidden, 0.0 );
            a.assign( kBatch * kHidden, 0.0 );
            std::vector<double> y( kBatch * kOut, 0.0 );

            for ( int64_t b = 0; b < kBatch; ++b )
            {
                for ( int64_t j = 0; j < kHidden; ++j )
                {
                    double accumulator = bias1Value( j );

                    for ( int64_t i = 0; i < kIn; ++i )
                    {
                        accumulator += static_cast<double>( x[ b * kIn + i ] ) * weight1Value( j, i );
                    }

                    h[ b * kHidden + j ] = accumulator;
                    a[ b * kHidden + j ] = geluReference( accumulator );
                }

                for ( int64_t k = 0; k < kOut; ++k )
                {
                    double accumulator = bias2Value( k );

                    for ( int64_t j = 0; j < kHidden; ++j )
                    {
                        accumulator += a[ b * kHidden + j ] * weight2Value( k, j );
                    }

                    y[ b * kOut + k ] = accumulator;
                }
            }

            return y;
        }

        static float backwardTolerance( double expected )
        {
            return kBackwardAbsoluteTolerance + kBackwardRelativeTolerance * static_cast<float>( std::fabs( expected ) );
        }

        std::unique_ptr<IExecutionContext> probe_context_;
    };

    // ====================================================================
    // A. Construction & Build
    // ====================================================================

    TEST_F( MlpNetworkCudaTests, Construct_StandaloneSucceeds )
    {
        MlpCuda net( "mlp", kIn, kHidden, kOut, Device::Cuda( 0 ) );

        EXPECT_EQ( net.getName(), "mlp" );
        EXPECT_EQ( net.getDeviceId().type, DeviceType::Cuda );
        EXPECT_EQ( net.childCount(), 3u );
    }

    TEST_F( MlpNetworkCudaTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW( MlpCuda( "mlp", kIn, kHidden, kOut, Device::Cpu() ), std::invalid_argument );
    }

    TEST_F( MlpNetworkCudaTests, Forward_ThrowsBeforeBuild )
    {
        MlpCuda net( "mlp", kIn, kHidden, kOut, Device::Cuda( 0 ) );
        DeviceFp32 input( Device::Cuda( 0 ), shape_t{ kBatch, kIn } );

        EXPECT_THROW( net.forward( input ), std::runtime_error );
    }

    TEST_F( MlpNetworkCudaTests, Build_PropagatesToChildrenAndAggregatesParameters )
    {
        auto net = builtMlp( RuntimeMode::Inference );

        EXPECT_TRUE( net->isBuilt() );
        EXPECT_TRUE( net->linear1()->isBuilt() );
        EXPECT_TRUE( net->linear2()->isBuilt() );

        const dim_t expected = ( kIn * kHidden + kHidden ) + ( kHidden * kOut + kOut );
        EXPECT_EQ( net->parameterCount(), expected );
    }

    // ====================================================================
    // B. Runtime mode
    // ====================================================================

    TEST_F( MlpNetworkCudaTests, Backward_ThrowsWhenBuiltForInference )
    {
        auto net = builtMlp( RuntimeMode::Inference );
        setKnownParameters( *net );

        auto input = toDevice( *net, spreadInput() );
        DeviceFp32 output_grad( Device::Cuda( 0 ), shape_t{ kBatch, kOut } );

        net->forward( input );

        EXPECT_THROW( net->backward( input, output_grad ), std::runtime_error );
    }

    TEST_F( MlpNetworkCudaTests, GetGradients_EmptyWhenBuiltForInference )
    {
        auto net = builtMlp( RuntimeMode::Inference );

        EXPECT_TRUE( net->getGradients().empty() );
    }

    TEST_F( MlpNetworkCudaTests, GetParameters_AggregatesChildrenInOrder )
    {
        auto net = builtMlp( RuntimeMode::Training );

        auto parameters = net->getParameters();
        ASSERT_EQ( parameters.size(), 4u );
        EXPECT_EQ( net->getGradients().size(), 4u );

        EXPECT_EQ( parameters[ 0 ]->shape(), ( shape_t{ kHidden, kIn } ) );
        EXPECT_EQ( parameters[ 1 ]->shape(), ( shape_t{ kHidden } ) );
        EXPECT_EQ( parameters[ 2 ]->shape(), ( shape_t{ kOut, kHidden } ) );
        EXPECT_EQ( parameters[ 3 ]->shape(), ( shape_t{ kOut } ) );
    }

    // ====================================================================
    // C. Forward and backward against the host reference
    // ====================================================================

    TEST_F( MlpNetworkCudaTests, Forward_MatchesReference )
    {
        auto net = builtMlp( RuntimeMode::Inference );
        setKnownParameters( *net );

        auto host_input = spreadInput();
        auto input = toDevice( *net, host_input );

        auto& device_output = net->forward( input );
        auto output = toHostFp32( *net, &device_output );

        std::vector<double> h, a;
        const auto expected = referenceForward( host_input.data(), h, a );

        ASSERT_EQ( output.size(), static_cast<dim_t>( expected.size() ) );

        for ( dim_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], static_cast<float>( expected[ i ] ), kForwardTolerance )
                << "forward mismatch at index " << i;
        }
    }

    TEST_F( MlpNetworkCudaTests, Backward_PropagatesGradientsThroughGraph )
    {
        auto net = builtMlp( RuntimeMode::Training );
        setKnownParameters( *net );

        auto host_input = spreadInput();
        auto input = toDevice( *net, host_input );

        // Non-uniform, so a layer that ignored its output gradient fails.
        HostFp32 host_output_grad( Device::Cpu(), shape_t{ kBatch, kOut } );

        for ( dim_t i = 0; i < host_output_grad.size(); ++i )
        {
            host_output_grad.data()[ i ] = 0.05f * static_cast<float>( ( i % 7 ) + 1 ) - 0.15f;
        }

        auto output_grad = toDevice( *net, host_output_grad );
        const float* dy = host_output_grad.data();

        net->forward( input );
        net->zeroGradients();
        auto& device_input_grad = net->backward( input, output_grad );
        auto input_grad = toHostFp32( *net, &device_input_grad );

        std::vector<double> h, a;
        referenceForward( host_input.data(), h, a );

        // dA = dY W2 ; dH = dA * gelu'(H) ; dX = dH W1
        std::vector<double> dh( kBatch * kHidden, 0.0 );

        for ( int64_t b = 0; b < kBatch; ++b )
        {
            for ( int64_t j = 0; j < kHidden; ++j )
            {
                double da = 0.0;

                for ( int64_t k = 0; k < kOut; ++k )
                {
                    da += static_cast<double>( dy[ b * kOut + k ] ) * weight2Value( k, j );
                }

                dh[ b * kHidden + j ] = da * geluGradientReference( h[ b * kHidden + j ] );
            }
        }

        ASSERT_EQ( input_grad.size(), kBatch * kIn );

        for ( int64_t b = 0; b < kBatch; ++b )
        {
            for ( int64_t i = 0; i < kIn; ++i )
            {
                double expected = 0.0;

                for ( int64_t j = 0; j < kHidden; ++j )
                {
                    expected += dh[ b * kHidden + j ] * weight1Value( j, i );
                }

                EXPECT_NEAR( input_grad.data()[ b * kIn + i ], static_cast<float>( expected ), backwardTolerance( expected ) )
                    << "input-gradient mismatch at b=" << b << " i=" << i;
            }
        }

        auto fc1_grads = net->linear1()->getGradients();
        auto dw1 = toHostFp32( *net, fc1_grads[ 0 ] );
        auto db1 = toHostFp32( *net, fc1_grads[ 1 ] );

        for ( int64_t j = 0; j < kHidden; ++j )
        {
            double expected_bias = 0.0;

            for ( int64_t b = 0; b < kBatch; ++b )
            {
                expected_bias += dh[ b * kHidden + j ];
            }

            EXPECT_NEAR( db1.data()[ j ], static_cast<float>( expected_bias ), backwardTolerance( expected_bias ) )
                << "dB1 mismatch at j=" << j;

            for ( int64_t i = 0; i < kIn; ++i )
            {
                double expected_weight = 0.0;

                for ( int64_t b = 0; b < kBatch; ++b )
                {
                    expected_weight += dh[ b * kHidden + j ] * static_cast<double>( host_input.data()[ b * kIn + i ] );
                }

                EXPECT_NEAR( dw1.data()[ j * kIn + i ], static_cast<float>( expected_weight ), backwardTolerance( expected_weight ) )
                    << "dW1 mismatch at j=" << j << " i=" << i;
            }
        }

        auto fc2_grads = net->linear2()->getGradients();
        auto dw2 = toHostFp32( *net, fc2_grads[ 0 ] );
        auto db2 = toHostFp32( *net, fc2_grads[ 1 ] );

        for ( int64_t k = 0; k < kOut; ++k )
        {
            double expected_bias = 0.0;

            for ( int64_t b = 0; b < kBatch; ++b )
            {
                expected_bias += static_cast<double>( dy[ b * kOut + k ] );
            }

            EXPECT_NEAR( db2.data()[ k ], static_cast<float>( expected_bias ), backwardTolerance( expected_bias ) )
                << "dB2 mismatch at k=" << k;

            for ( int64_t j = 0; j < kHidden; ++j )
            {
                double expected_weight = 0.0;

                for ( int64_t b = 0; b < kBatch; ++b )
                {
                    expected_weight += static_cast<double>( dy[ b * kOut + k ] ) * a[ b * kHidden + j ];
                }

                EXPECT_NEAR( dw2.data()[ k * kHidden + j ], static_cast<float>( expected_weight ), backwardTolerance( expected_weight ) )
                    << "dW2 mismatch at k=" << k << " j=" << j;
            }
        }
    }

    // ====================================================================
    // D. Optimizer wiring and the training loop
    // ====================================================================

    TEST_F( MlpNetworkCudaTests, CreateOptimizer_RegistersAllParameters )
    {
        auto net = builtMlp( RuntimeMode::Training );

        auto optimizer = net->createOptimizer<AdamWOptimizer<DeviceType::Cuda, TensorDataType::FP32>>(
            AdamWConfig().withLearningRate( 0.01f ) );

        EXPECT_EQ( optimizer->getParameterCount(), 4u );
    }

    TEST_F( MlpNetworkCudaTests, TrainingLoop_ReducesRegressionLoss )
    {
        auto net = builtMlp( RuntimeMode::Training );
        setKnownParameters( *net );

        auto optimizer = net->createOptimizer<AdamWOptimizer<DeviceType::Cuda, TensorDataType::FP32>>(
            AdamWConfig().withLearningRate( 0.01f ).withWeightDecay( 0.0f ) );

        auto input = toDevice( *net, spreadInput() );

        std::vector<float> target( kBatch * kOut );

        for ( size_t i = 0; i < target.size(); ++i )
        {
            target[ i ] = 0.3f * std::sin( 0.7f * static_cast<float>( i ) );
        }

        HostFp32 host_output_grad( Device::Cpu(), shape_t{ kBatch, kOut } );
        DeviceFp32 output_grad( Device::Cuda( 0 ), shape_t{ kBatch, kOut } );

        // Loss = sum (Y - T)^2 ; dL/dY = 2 (Y - T).
        auto computeLossAndGrad = [&]() {
            auto& device_output = net->forward( input );
            auto output = toHostFp32( *net, &device_output );

            double loss = 0.0;

            for ( dim_t i = 0; i < output.size(); ++i )
            {
                const double difference = static_cast<double>( output.data()[ i ] ) - target[ i ];
                loss += difference * difference;
                host_output_grad.data()[ i ] = static_cast<float>( 2.0 * difference );
            }

            copy( host_output_grad, output_grad, net->context() );
            net->context()->synchronize();

            return loss;
        };

        const double initial_loss = computeLossAndGrad();

        for ( int step = 0; step < 100; ++step )
        {
            computeLossAndGrad();
            net->zeroGradients();
            net->backward( input, output_grad );
            optimizer->step();
        }

        const double final_loss = computeLossAndGrad();

        EXPECT_LT( final_loss, 0.5 * initial_loss )
            << "the training loop should reduce the regression loss (initial=" << initial_loss
            << ", final=" << final_loss << ")";
    }
}
