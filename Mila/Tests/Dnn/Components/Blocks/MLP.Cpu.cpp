#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <exception>
#include <cstdint>
#include <ostream>
#include <algorithm>

import Mila;

namespace Modules::Blocks::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    template<TensorDataType TPrecision>
    struct MLPCpuTestData
    {
        MLPConfig config;
        std::shared_ptr<MLP<DeviceType::Cpu, TPrecision>> mlp;
        shape_t input_shape;
        int64_t input_features;
        int64_t hidden_size;
        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> exec_context;

        MLPCpuTestData()
            : config( 1, 1 ), input_features( 0 ), hidden_size( 0 )
        {
        }

        static MLPCpuTestData Create(
            const std::string& name,
            const shape_t& input_shape,
            int64_t input_features,
            int64_t hidden_size,
            bool has_bias = true,
            ActivationType activation = ActivationType::Gelu,
            bool use_layer_norm = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            MLPCpuTestData data;
            data.input_shape = input_shape;
            data.input_features = input_features;
            data.hidden_size = hidden_size;

            data.config = MLPConfig( input_features, hidden_size );
            data.config.withBias( has_bias )
                .withActivation( activation )
                .withLayerNorm( use_layer_norm )
                .withName( name )
                .withPrecisionPolicy( precision );

            data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            data.mlp = std::make_shared<MLP<DeviceType::Cpu, TPrecision>>( data.exec_context, data.config );

            return data;
        }

        static MLPCpuTestData CreateWithContext(
            const std::string& name,
            const shape_t& input_shape,
            int64_t input_features,
            int64_t hidden_size,
            std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
            bool has_bias = true,
            ActivationType activation = ActivationType::Gelu,
            bool use_layer_norm = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            MLPCpuTestData data;
            data.input_shape = input_shape;
            data.input_features = input_features;
            data.hidden_size = hidden_size;

            data.config = MLPConfig( input_features, hidden_size );
            data.config.withBias( has_bias )
                .withActivation( activation )
                .withLayerNorm( use_layer_norm )
                .withName( name )
                .withPrecisionPolicy( precision );

            data.exec_context = context;
            data.mlp = std::make_shared<MLP<DeviceType::Cpu, TPrecision>>( context, data.config );

            return data;
        }
    };

    class MLPCpuTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            batch_size_ = 2;
            sequence_length_ = 16;
            input_features_ = 768;
            hidden_size_ = 3072;
        }

        int64_t batch_size_{ 0 };
        int64_t sequence_length_{ 0 };
        int64_t input_features_{ 0 };
        int64_t hidden_size_{ 0 };
    };

    TEST_F( MLPCpuTests, GetName )
    {
        // Setup
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        // Execution & Assertions
        ASSERT_NE( data.mlp, nullptr );
        EXPECT_EQ( data.mlp->getName(), "small_mlp_cpu" );
    }

    TEST_F( MLPCpuTests, DeviceType )
    {
        // Setup
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        // Execution & Assertions
        ASSERT_NE( data.exec_context, nullptr );
        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( MLPCpuTests, IsBuilt_BeforeBuild )
    {
        // Setup
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        // Execution & Assertions
        ASSERT_NE( data.mlp, nullptr );
        EXPECT_FALSE( data.mlp->isBuilt() );
    }

    TEST_F( MLPCpuTests, IsBuilt_AfterBuild )
    {
        // Setup
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        // Execution
        ASSERT_NE( data.mlp, nullptr );
        EXPECT_NO_THROW( data.mlp->build( data.input_shape ) );

        // Assertions
        EXPECT_TRUE( data.mlp->isBuilt() );
    }

    TEST_F( MLPCpuTests, Build )
    {
        // Setup
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        // Execution & Assertions
        ASSERT_NE( data.mlp, nullptr );
        EXPECT_NO_THROW( data.mlp->build( data.input_shape ) );
        EXPECT_TRUE( data.mlp->isBuilt() );
    }

    TEST_F( MLPCpuTests, ParameterCount )
    {
        // Setup
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        // Execution & Assertions
        ASSERT_NE( data.mlp, nullptr );

        int64_t input_features = data.config.getInputFeatures();
        int64_t hidden_size = data.config.getHiddenSize();
        bool has_bias = data.config.hasBias();

        size_t expected_fc1_params = input_features * hidden_size;
        size_t expected_fc2_params = hidden_size * input_features;

        if ( has_bias )
        {
            expected_fc1_params += hidden_size;
            expected_fc2_params += input_features;
        }

        size_t expected_total_params = expected_fc1_params + expected_fc2_params;

        EXPECT_EQ( data.mlp->parameterCount(), expected_total_params );
    }

    TEST_F( MLPCpuTests, Forward )
    {
        // Setup
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        // Build and allocate
        data.mlp->build( data.input_shape );

        TensorType input( data.exec_context->getDevice(), data.input_shape );
        TensorType output( data.exec_context->getDevice(), data.input_shape );

        // Initialize input and run
        random( input, -1.0f, 1.0f );

        EXPECT_NO_THROW( data.mlp->forward( input, output ) );

        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    TEST_F( MLPCpuTests, ToString )
    {
        // Setup
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        // Execution
        ASSERT_NE( data.mlp, nullptr );
        std::string output = data.mlp->toString();

        // Assertion
        EXPECT_NE( output.find( "MLP: small_mlp_cpu" ), std::string::npos );
    }

    TEST_F( MLPCpuTests, TrainingMode_Default )
    {
        // Setup
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        // Assertion
        ASSERT_NE( data.mlp, nullptr );
        EXPECT_FALSE( data.mlp->isTraining() );
    }

    TEST_F( MLPCpuTests, TrainingMode_Enabled )
    {
        // Setup
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "training_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        // Execution: enable training
        ASSERT_NE( data.mlp, nullptr );
        data.mlp->setTraining( true );

        // Assertion
        EXPECT_TRUE( data.mlp->isTraining() );
    }

    TEST_F( MLPCpuTests, GetNamedComponents_Returns_ChildComponents )
    {
        // Setup: create MLP and inspect registered subcomponents
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );

        auto components = data.mlp->getNamedComponents();

        EXPECT_GE( components.size(), 3u );

        const std::string base = data.mlp->getName();

        EXPECT_NE( components.find( base + ".fc1" ), components.end() );
        EXPECT_NE( components.find( base + ".act" ), components.end() );
        EXPECT_NE( components.find( base + ".fc2" ), components.end() );

        if ( data.config.useLayerNorm() )
        {
            EXPECT_NE( components.find( base + ".norm" ), components.end() );
        }
    }

    TEST_F( MLPCpuTests, SaveLoad )
    {
        // Setup
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );

        // Save/load tests are not implemented in the core yet; ensure no crash if present.
        // Kept intentionally minimal per original test.
    }

    TEST_F( MLPCpuTests, NoBias_ParameterCount )
    {
        // Setup (no bias)
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "no_bias_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            false );

        ASSERT_NE( data.mlp, nullptr );

        // Execution & Assertions (same calculation as ParameterCount)
        int64_t input_features = data.config.getInputFeatures();
        int64_t hidden_size = data.config.getHiddenSize();
        bool has_bias = data.config.hasBias();

        size_t expected_fc1_params = input_features * hidden_size;
        size_t expected_fc2_params = hidden_size * input_features;

        if ( has_bias )
        {
            expected_fc1_params += hidden_size;
            expected_fc2_params += input_features;
        }

        size_t expected_total_params = expected_fc1_params + expected_fc2_params;

        EXPECT_EQ( data.mlp->parameterCount(), expected_total_params );
    }

    TEST_F( MLPCpuTests, NoBias_Forward )
    {
        // Setup (no bias)
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "no_bias_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            false );

        ASSERT_NE( data.mlp, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        data.mlp->build( data.input_shape );

        TensorType input( data.exec_context->getDevice(), data.input_shape );
        TensorType output( data.exec_context->getDevice(), data.input_shape );

        random( input, -1.0f, 1.0f );

        EXPECT_NO_THROW( data.mlp->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    TEST_F( MLPCpuTests, LayerNorm_Forward )
    {
        // Setup (with layer norm)
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "layer_norm_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            true,
            ActivationType::Gelu,
            true );

        ASSERT_NE( data.mlp, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        data.mlp->build( data.input_shape );

        TensorType input( data.exec_context->getDevice(), data.input_shape );
        TensorType output( data.exec_context->getDevice(), data.input_shape );

        random( input, -1.0f, 1.0f );

        EXPECT_NO_THROW( data.mlp->forward( input, output ) );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    TEST_F( MLPCpuTests, LayerNorm_SubModules )
    {
        // Setup (with layer norm)
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "layer_norm_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            true,
            ActivationType::Gelu,
            true );

        ASSERT_NE( data.mlp, nullptr );

        auto components = data.mlp->getNamedComponents();

        const std::string base = data.mlp->getName();

        EXPECT_NE( components.find( base + ".fc1" ), components.end() );
        EXPECT_NE( components.find( base + ".act" ), components.end() );
        EXPECT_NE( components.find( base + ".fc2" ), components.end() );
        EXPECT_NE( components.find( base + ".norm" ), components.end() );
    }

    TEST_F( MLPCpuTests, Training_Forward )
    {
        // Setup (training mode)
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "training_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        data.mlp->setTraining( true );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        data.mlp->build( data.input_shape );

        TensorType input( data.exec_context->getDevice(), data.input_shape );
        TensorType output( data.exec_context->getDevice(), data.input_shape );

        random( input, -1.0f, 1.0f );

        EXPECT_NO_THROW( data.mlp->forward( input, output ) );
    }

    TEST_F( MLPCpuTests, WithContext_Construction )
    {
        // Setup: explicit context
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        auto data = MLPCpuTestData<TensorDataType::FP32>::CreateWithContext(
            "context_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            ctx );

        ASSERT_NE( data.mlp, nullptr );

        // Assertions
        EXPECT_EQ( data.mlp->getName(), "context_mlp_cpu" );
        EXPECT_EQ( data.exec_context, ctx );
    }

    TEST_F( MLPCpuTests, EdgeCase_MinimalShape )
    {
        // Setup
        shape_t shape = { 1, 1, 8 };
        int64_t hidden = 16;

        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "minimal_cpu", shape, 8, hidden );

        ASSERT_NE( data.mlp, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        data.mlp->build( data.input_shape );

        TensorType input( data.exec_context->getDevice(), data.input_shape );
        TensorType output( data.exec_context->getDevice(), data.input_shape );

        random( input, -1.0f, 1.0f );

        EXPECT_NO_THROW( data.mlp->forward( input, output ) );
    }

    TEST_F( MLPCpuTests, EdgeCase_MediumShape )
    {
        // Setup
        shape_t shape = { 1, 2, 512 };
        int64_t hidden = 2048;

        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "medium_cpu", shape, 512, hidden );

        ASSERT_NE( data.mlp, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        data.mlp->build( data.input_shape );

        TensorType input( data.exec_context->getDevice(), data.input_shape );
        TensorType output( data.exec_context->getDevice(), data.input_shape );

        random( input, -1.0f, 1.0f );

        EXPECT_NO_THROW( data.mlp->forward( input, output ) );
    }

    TEST_F( MLPCpuTests, Error_InvalidConfiguration_ZeroInputFeatures )
    {
        MLPConfig invalid_config( 0, 1024 );

        auto cpu_exec = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        EXPECT_THROW(
            (MLP<DeviceType::Cpu, TensorDataType::FP32>( cpu_exec, invalid_config )),
            std::invalid_argument
        );
    }

    TEST_F( MLPCpuTests, Error_InvalidConfiguration_ZeroHiddenSize )
    {
        MLPConfig invalid_config( 768, 0 );

        auto cpu_exec = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        EXPECT_THROW(
            (MLP<DeviceType::Cpu, TensorDataType::FP32>( cpu_exec, invalid_config )),
            std::invalid_argument
        );
    }

    TEST_F( MLPCpuTests, Error_NullExecutionContext )
    {
        MLPConfig config( 768, 3072 );
        config.withName( "null_context_test" );

        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> null_ctx;

        EXPECT_THROW(
            (MLP<DeviceType::Cpu, TensorDataType::FP32>( null_ctx, config )),
            std::invalid_argument
        );
    }

    TEST_F( MLPCpuTests, Error_ForwardBeforeBuild )
    {
        shape_t test_shape = { 2, 16, 768 };

        MLPConfig config( 768, 3072 );
        config.withName( "unbuild_test" );

        auto cpu_exec = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto mlp = std::make_shared<MLP<DeviceType::Cpu, TensorDataType::FP32>>( cpu_exec, config );

        CpuTensor<TensorDataType::FP32> input( cpu_exec->getDevice(), test_shape );
        CpuTensor<TensorDataType::FP32> output( cpu_exec->getDevice(), test_shape );

        EXPECT_THROW(
            mlp->forward( input, output ),
            std::runtime_error
        );
    }

    TEST_F( MLPCpuTests, Synchronize )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );

        EXPECT_NO_THROW( data.mlp->synchronize() );
    }

    TEST_F( MLPCpuTests, SetTrainingMode )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );

        EXPECT_FALSE( data.mlp->isTraining() );

        data.mlp->setTraining( true );
        EXPECT_TRUE( data.mlp->isTraining() );

        data.mlp->setTraining( false );
        EXPECT_FALSE( data.mlp->isTraining() );
    }

    TEST_F( MLPCpuTests, MultipleForwardCalls )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        data.mlp->build( data.input_shape );

        CpuTensor<TensorDataType::FP32> input( data.exec_context->getDevice(), data.input_shape );
        CpuTensor<TensorDataType::FP32> output( data.exec_context->getDevice(), data.input_shape );

        for ( int iter = 0; iter < 10; ++iter )
        {
            random( input, -1.0f, 1.0f );

            EXPECT_NO_THROW( data.mlp->forward( input, output ) );
        }
    }
}