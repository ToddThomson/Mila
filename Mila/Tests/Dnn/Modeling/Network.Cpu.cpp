/**
 * @file Network.Cpu.cpp
 * @brief Concrete-network forward/backward gradient-flow tests (CPU, FP32).
 *
 * Net-new coverage: the authored suite validated component forward passes against
 * HuggingFace but never exercised a composite Network's forward()/backward()
 * CHAINING with real child components. This is the MNIST training spine in
 * miniature -- a Linear -> Gelu -> Linear MLP (the same shape MnistClassifier
 * builds) driven end to end:
 *   - composite forward chains component-owned activations,
 *   - composite backward chains child backward() in reverse and lands gradients
 *     in each child's parameter-gradient buffers,
 *   - Network::createOptimizer wires AdamW to the aggregated parameters/gradients,
 *   - zeroGradients / a multi-step training loop reduce a regression loss.
 *
 * The numeric references (forward, input gradient, and per-layer parameter
 * gradients) are computed independently in double precision. This is the oracle
 * the Training Revival milestone depends on: it proves the gradient FLOW through
 * a composite, which the green leaf tests (Linear.Cpu / Gelu.Cpu) cannot.
 *
 * Single precision (CPU is FP32-only); kept as a plain fixture rather than a
 * precision sweep because this is a network-level integration test, not a leaf
 * component. CPU device, so it rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <stdexcept>
#include <format>

import Mila;

namespace Mila::Tests::Dnn::Modeling
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Optimizers;
    using namespace Mila::Dnn::Serialization;

    namespace
    {
        // GELU tanh-approximation reference and its derivative (matches Gelu.Cpu.cpp),
        // computed independently of the component under test.
        constexpr float kSqrt2OverPi = 0.7978845608f;
        constexpr float kGeluCoeff = 0.044715f;

        double geluReference( double x )
        {
            const double x_cubed = x * x * x;

            return 0.5 * x * ( 1.0 + std::tanh( kSqrt2OverPi * ( x + kGeluCoeff * x_cubed ) ) );
        }

        double geluGradientReference( double x )
        {
            const double x_squared = x * x;
            const double arg = kSqrt2OverPi * ( x + kGeluCoeff * x * x_squared );
            const double tanh_value = std::tanh( arg );
            const double sech_squared = 1.0 - tanh_value * tanh_value;
            const double d_arg = kSqrt2OverPi * ( 1.0 + 3.0 * kGeluCoeff * x_squared );

            return 0.5 * ( 1.0 + tanh_value ) + 0.5 * x * sech_squared * d_arg;
        }

        // Deterministic parameter values, independent of the component under test.
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
            return 0.02f * static_cast<float>( in_index + 1 ) - 0.04f * static_cast<float>( out_index );
        }

        float bias2Value( int64_t out_index )
        {
            return 0.2f - 0.1f * static_cast<float>( out_index );
        }

        // ================================================================
        // Concrete Network under test: Linear(in->hidden) -> Gelu -> Linear(hidden->out).
        //
        // Mirrors the MnistClassifier construction pattern: owns its
        // ExecutionContext, builds the graph context-independently, propagates the
        // context to children, and builds each child with its own correctly-sized
        // BuildContext (Linear validates input_shape.back() == in_features).
        // ================================================================
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

                auto& h = fc1_->forward( input );
                fc1_out_ = &h;

                auto& a = act_->forward( h );
                act_out_ = &a;

                auto& y = fc2_->forward( a );

                return y;
            }

            // Returns the component-owned input gradient produced by the first layer.
            TensorType& backward( const TensorType& input, const TensorType& output_grad )
            {
                if ( !this->isBuilt() )
                {
                    throw std::runtime_error( "MlpUnderTest: must be built before backward" );
                }

                auto& da = fc2_->backward( *act_out_, output_grad );
                auto& dh = act_->backward( *fc1_out_, da );
                auto& dx = fc1_->backward( input, dh );

                return dx;
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

            // Test accessors (valid after build()).
            const std::shared_ptr<LinearType>& linear1() const { return fc1_; }
            const std::shared_ptr<LinearType>& linear2() const { return fc2_; }

        protected:
            void onBuilding( const BuildContext& context ) override
            {
                const auto& input_shape = context.inputShape();
                const int64_t batch = input_shape[ 0 ];

                fc1_ = this->template getComponentAs<LinearType>( this->getName() + ".fc1" );
                fc1_->build( context.withShape( shape_t{ batch, in_features_ } ) );

                act_ = this->template getComponentAs<GeluType>( this->getName() + ".act" );
                act_->build( context.withShape( shape_t{ batch, hidden_features_ } ) );

                fc2_ = this->template getComponentAs<LinearType>( this->getName() + ".fc2" );
                fc2_->build( context.withShape( shape_t{ batch, hidden_features_ } ) );
            }

            void save_( ModelArchive&, SerializationMode ) const override
            {
                // Serialization round-trip is covered by the serialization pass; this
                // concrete network only needs a valid override to satisfy the contract.
            }

        private:
            void createGraph()
            {
                auto fc1 = std::make_shared<LinearType>(
                    this->getName() + ".fc1",
                    LinearConfig( in_features_, hidden_features_ ).withBias( true ),
                    std::nullopt );
                this->addComponent( fc1 );

                auto act = std::make_shared<GeluType>(
                    this->getName() + ".act", GeluConfig(), std::nullopt );
                this->addComponent( act );

                auto fc2 = std::make_shared<LinearType>(
                    this->getName() + ".fc2",
                    LinearConfig( hidden_features_, out_features_ ).withBias( true ),
                    std::nullopt );
                this->addComponent( fc2 );
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

        using MlpCpu = MlpUnderTest<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
    }

    class MlpNetworkCpuTests : public ::testing::Test
    {
    protected:
        static constexpr int64_t kBatch = 2;
        static constexpr int64_t kIn = 3;
        static constexpr int64_t kHidden = 4;
        static constexpr int64_t kOut = 2;

        std::unique_ptr<MlpCpu> builtMlp( RuntimeMode mode )
        {
            auto net = std::make_unique<MlpCpu>( "mlp", kIn, kHidden, kOut, Device::Cpu() );
            net->build( BuildContext( shape_t{ kBatch, kIn }, mode, false ) );

            return net;
        }

        static void setKnownParameters( MlpCpu& net )
        {
            auto fc1_params = net.linear1()->getParameters();
            float* w1 = static_cast<float*>( fc1_params[ 0 ]->rawData() );
            float* b1 = static_cast<float*>( fc1_params[ 1 ]->rawData() );

            for ( int64_t j = 0; j < kHidden; ++j )
            {
                for ( int64_t i = 0; i < kIn; ++i )
                {
                    w1[ j * kIn + i ] = weight1Value( j, i );
                }

                b1[ j ] = bias1Value( j );
            }

            auto fc2_params = net.linear2()->getParameters();
            float* w2 = static_cast<float*>( fc2_params[ 0 ]->rawData() );
            float* b2 = static_cast<float*>( fc2_params[ 1 ]->rawData() );

            for ( int64_t k = 0; k < kOut; ++k )
            {
                for ( int64_t j = 0; j < kHidden; ++j )
                {
                    w2[ k * kHidden + j ] = weight2Value( k, j );
                }

                b2[ k ] = bias2Value( k );
            }
        }

        static void fillSpread( TensorFp32& t )
        {
            for ( size_t i = 0; i < t.size(); ++i )
            {
                t.data()[ i ] = static_cast<float>( i ) / t.size() * 4.0f - 2.0f;
            }
        }

        // Reference forward: returns Y[kBatch*kOut]; fills H and A (pre/post activation).
        static std::vector<double> referenceForward(
            const float* X, std::vector<double>& H, std::vector<double>& A )
        {
            H.assign( kBatch * kHidden, 0.0 );
            A.assign( kBatch * kHidden, 0.0 );
            std::vector<double> Y( kBatch * kOut, 0.0 );

            for ( int64_t b = 0; b < kBatch; ++b )
            {
                for ( int64_t j = 0; j < kHidden; ++j )
                {
                    double acc = bias1Value( j );
                    for ( int64_t i = 0; i < kIn; ++i )
                    {
                        acc += static_cast<double>( X[ b * kIn + i ] ) * weight1Value( j, i );
                    }
                    H[ b * kHidden + j ] = acc;
                    A[ b * kHidden + j ] = geluReference( acc );
                }

                for ( int64_t k = 0; k < kOut; ++k )
                {
                    double acc = bias2Value( k );
                    for ( int64_t j = 0; j < kHidden; ++j )
                    {
                        acc += A[ b * kHidden + j ] * weight2Value( k, j );
                    }
                    Y[ b * kOut + k ] = acc;
                }
            }

            return Y;
        }
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( MlpNetworkCpuTests, Construct_StandaloneSucceeds )
    {
        MlpCpu net( "mlp", kIn, kHidden, kOut, Device::Cpu() );

        EXPECT_EQ( net.getName(), "mlp" );
        EXPECT_EQ( net.getDeviceId().type, DeviceType::Cpu );
        EXPECT_EQ( net.childCount(), 3u );
    }

    TEST_F( MlpNetworkCpuTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW( MlpCpu( "mlp", kIn, kHidden, kOut, Device::Cuda( 0 ) ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle
    // ====================================================================

    TEST_F( MlpNetworkCpuTests, Forward_ThrowsBeforeBuild )
    {
        MlpCpu net( "mlp", kIn, kHidden, kOut, Device::Cpu() );
        TensorFp32 input( Device::Cpu(), shape_t{ kBatch, kIn } );

        EXPECT_THROW( net.forward( input ), std::runtime_error );
    }

    TEST_F( MlpNetworkCpuTests, Build_PropagatesToChildrenAndAggregatesParameters )
    {
        auto net = builtMlp( RuntimeMode::Inference );

        EXPECT_TRUE( net->isBuilt() );
        EXPECT_TRUE( net->linear1()->isBuilt() );
        EXPECT_TRUE( net->linear2()->isBuilt() );

        // fc1: in*hidden + hidden ; fc2: hidden*out + out ; gelu: 0.
        const size_t expected = static_cast<size_t>( kIn * kHidden + kHidden )
            + static_cast<size_t>( kHidden * kOut + kOut );
        EXPECT_EQ( net->parameterCount(), expected );
    }

    // ====================================================================
    // D. Runtime + Training Mode
    // ====================================================================

    TEST_F( MlpNetworkCpuTests, Backward_ThrowsWhenBuiltForInference )
    {
        auto net = builtMlp( RuntimeMode::Inference );
        setKnownParameters( *net );

        TensorFp32 input( Device::Cpu(), shape_t{ kBatch, kIn } );
        TensorFp32 output_grad( Device::Cpu(), shape_t{ kBatch, kOut } );
        fillSpread( input );

        net->forward( input );

        EXPECT_THROW( net->backward( input, output_grad ), std::runtime_error );
    }

    TEST_F( MlpNetworkCpuTests, GetGradients_EmptyWhenBuiltForInference )
    {
        auto net = builtMlp( RuntimeMode::Inference );

        EXPECT_TRUE( net->getGradients().empty() );
    }

    TEST_F( MlpNetworkCpuTests, GetParameters_AggregatesChildrenInOrder )
    {
        auto net = builtMlp( RuntimeMode::Training );

        // fc1 (weight,bias) + fc2 (weight,bias); gelu contributes none.
        EXPECT_EQ( net->getParameters().size(), 4u );
        EXPECT_EQ( net->getGradients().size(), 4u );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( MlpNetworkCpuTests, Forward_MatchesReference )
    {
        auto net = builtMlp( RuntimeMode::Inference );
        setKnownParameters( *net );

        TensorFp32 input( Device::Cpu(), shape_t{ kBatch, kIn } );
        fillSpread( input );

        auto& output = net->forward( input );

        std::vector<double> H, A;
        std::vector<double> expected = referenceForward( input.data(), H, A );

        ASSERT_EQ( output.size(), expected.size() );

        for ( size_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], static_cast<float>( expected[ i ] ), 1e-4f )
                << "forward mismatch at index " << i;
        }
    }

    // ====================================================================
    // F. Backward (numeric vs analytic gradient through the whole graph)
    // ====================================================================

    TEST_F( MlpNetworkCpuTests, Backward_PropagatesGradientsThroughGraph )
    {
        auto net = builtMlp( RuntimeMode::Training );
        setKnownParameters( *net );

        TensorFp32 input( Device::Cpu(), shape_t{ kBatch, kIn } );
        TensorFp32 output_grad( Device::Cpu(), shape_t{ kBatch, kOut } );
        fillSpread( input );

        for ( size_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = 0.1f * static_cast<float>( i + 1 );
        }

        net->forward( input );
        auto& input_grad = net->backward( input, output_grad );

        // Reference activations.
        std::vector<double> H, A;
        referenceForward( input.data(), H, A );

        // dA[b,j] = sum_k dY[b,k] * W2[k,j]
        // dH[b,j] = dA[b,j] * gelu'(H[b,j])
        // dX[b,i] = sum_j dH[b,j] * W1[j,i]
        // dW2[k,j] = sum_b dY[b,k] * A[b,j] ; dB2[k] = sum_b dY[b,k]
        // dW1[j,i] = sum_b dH[b,j] * X[b,i] ; dB1[j] = sum_b dH[b,j]
        std::vector<double> dH( kBatch * kHidden, 0.0 );
        std::vector<double> expected_dx( kBatch * kIn, 0.0 );

        for ( int64_t b = 0; b < kBatch; ++b )
        {
            for ( int64_t j = 0; j < kHidden; ++j )
            {
                double da = 0.0;
                for ( int64_t k = 0; k < kOut; ++k )
                {
                    da += static_cast<double>( output_grad.data()[ b * kOut + k ] ) * weight2Value( k, j );
                }
                dH[ b * kHidden + j ] = da * geluGradientReference( H[ b * kHidden + j ] );
            }

            for ( int64_t i = 0; i < kIn; ++i )
            {
                double acc = 0.0;
                for ( int64_t j = 0; j < kHidden; ++j )
                {
                    acc += dH[ b * kHidden + j ] * weight1Value( j, i );
                }
                expected_dx[ b * kIn + i ] = acc;
            }
        }

        ASSERT_EQ( input_grad.size(), expected_dx.size() );
        for ( size_t i = 0; i < input_grad.size(); ++i )
        {
            EXPECT_NEAR( input_grad.data()[ i ], static_cast<float>( expected_dx[ i ] ), 1e-3f )
                << "input-gradient mismatch at index " << i;
        }

        // Per-layer parameter gradients.
        auto fc1_grads = net->linear1()->getGradients();
        const float* dw1 = static_cast<const float*>( fc1_grads[ 0 ]->rawData() );
        const float* db1 = static_cast<const float*>( fc1_grads[ 1 ]->rawData() );

        for ( int64_t j = 0; j < kHidden; ++j )
        {
            double db = 0.0;
            for ( int64_t b = 0; b < kBatch; ++b )
            {
                db += dH[ b * kHidden + j ];
            }
            EXPECT_NEAR( db1[ j ], static_cast<float>( db ), 1e-3f ) << "dB1 mismatch at j=" << j;

            for ( int64_t i = 0; i < kIn; ++i )
            {
                double dw = 0.0;
                for ( int64_t b = 0; b < kBatch; ++b )
                {
                    dw += dH[ b * kHidden + j ] * static_cast<double>( input.data()[ b * kIn + i ] );
                }
                EXPECT_NEAR( dw1[ j * kIn + i ], static_cast<float>( dw ), 1e-3f )
                    << "dW1 mismatch at j=" << j << " i=" << i;
            }
        }

        auto fc2_grads = net->linear2()->getGradients();
        const float* dw2 = static_cast<const float*>( fc2_grads[ 0 ]->rawData() );
        const float* db2 = static_cast<const float*>( fc2_grads[ 1 ]->rawData() );

        for ( int64_t k = 0; k < kOut; ++k )
        {
            double db = 0.0;
            for ( int64_t b = 0; b < kBatch; ++b )
            {
                db += static_cast<double>( output_grad.data()[ b * kOut + k ] );
            }
            EXPECT_NEAR( db2[ k ], static_cast<float>( db ), 1e-3f ) << "dB2 mismatch at k=" << k;

            for ( int64_t j = 0; j < kHidden; ++j )
            {
                double dw = 0.0;
                for ( int64_t b = 0; b < kBatch; ++b )
                {
                    dw += static_cast<double>( output_grad.data()[ b * kOut + k ] ) * A[ b * kHidden + j ];
                }
                EXPECT_NEAR( dw2[ k * kHidden + j ], static_cast<float>( dw ), 1e-3f )
                    << "dW2 mismatch at k=" << k << " j=" << j;
            }
        }
    }

    // ====================================================================
    // G. Optimizer wiring + training spine (end-to-end convergence oracle)
    // ====================================================================

    TEST_F( MlpNetworkCpuTests, CreateOptimizer_RegistersAllParameters )
    {
        auto net = builtMlp( RuntimeMode::Training );

        auto optimizer = net->createOptimizer<AdamWOptimizer<DeviceType::Cpu, TensorDataType::FP32>>(
            AdamWConfig().withLearningRate( 0.01f ) );

        // One parameter group per parameter tensor (fc1 weight/bias + fc2 weight/bias).
        EXPECT_EQ( optimizer->getParameterCount(), 4u );
    }

    TEST_F( MlpNetworkCpuTests, TrainingLoop_ReducesRegressionLoss )
    {
        auto net = builtMlp( RuntimeMode::Training );
        setKnownParameters( *net );

        auto optimizer = net->createOptimizer<AdamWOptimizer<DeviceType::Cpu, TensorDataType::FP32>>(
            AdamWConfig().withLearningRate( 0.02f ).withWeightDecay( 0.0f ) );

        TensorFp32 input( Device::Cpu(), shape_t{ kBatch, kIn } );
        fillSpread( input );

        // Fixed regression target. Loss = sum (Y - T)^2 ; dL/dY = 2 (Y - T).
        const std::vector<float> target{ 0.3f, -0.2f, 0.5f, 0.1f };

        TensorFp32 output_grad( Device::Cpu(), shape_t{ kBatch, kOut } );

        auto computeLossAndGrad = [&]() {
            auto& y = net->forward( input );

            double loss = 0.0;
            for ( size_t i = 0; i < y.size(); ++i )
            {
                const double d = static_cast<double>( y.data()[ i ] ) - target[ i ];
                loss += d * d;
                output_grad.data()[ i ] = static_cast<float>( 2.0 * d );
            }

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

        EXPECT_LT( final_loss, initial_loss )
            << "the training loop should reduce the regression loss (initial=" << initial_loss
            << ", final=" << final_loss << ")";
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( MlpNetworkCpuTests, ToString_NamesNetwork )
    {
        MlpCpu net( "mlp", kIn, kHidden, kOut, Device::Cpu() );

        EXPECT_NE( net.toString().find( "mlp" ), std::string::npos );
    }
}
