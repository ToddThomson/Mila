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

namespace CompositeComponents_Tests
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

        MLPCpuTestData()
            : config( 1, 1 ), input_features( 0 ), hidden_size( 0 )
        {}

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
                .withPrecisionPolicy( precision );

            // Construct MLP in standalone mode (owns its ExecutionContext)
            data.mlp = std::make_shared<MLP<DeviceType::Cpu, TPrecision>>( name, data.config, Device::Cpu() );

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
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );
        EXPECT_EQ( data.mlp->getName(), "small_mlp_cpu" );
    }

    TEST_F( MLPCpuTests, DeviceType )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );
        auto device = data.mlp->getDeviceId();

        EXPECT_EQ( device.type, DeviceType::Cpu );
    }

    TEST_F( MLPCpuTests, IsBuilt_BeforeBuild )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );
        EXPECT_FALSE( data.mlp->isBuilt() );
    }

    TEST_F( MLPCpuTests, IsBuilt_AfterBuild )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );
        EXPECT_NO_THROW( data.mlp->build( data.input_shape ) );
        EXPECT_TRUE( data.mlp->isBuilt() );
    }

    TEST_F( MLPCpuTests, Build )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );
        EXPECT_NO_THROW( data.mlp->build( data.input_shape ) );
        EXPECT_TRUE( data.mlp->isBuilt() );
    }

    TEST_F( MLPCpuTests, ParameterCount )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );

        data.mlp->build( data.input_shape );

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
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        data.mlp->build( data.input_shape );

        TensorType input( data.mlp->getDeviceId(), data.input_shape );

        random( input, -1.0f, 1.0f );

        TensorType* out_ptr = nullptr;

        EXPECT_NO_THROW( { auto& out_ref = data.mlp->forward( input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );

        auto* out_tensor = out_ptr;
        ASSERT_NE( out_tensor, nullptr );

        EXPECT_EQ( out_tensor->size(), input.size() );
        EXPECT_EQ( out_tensor->shape(), input.shape() );
    }

    TEST_F( MLPCpuTests, ToString )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );
        std::string output = data.mlp->toString();

        EXPECT_NE( output.find( "MLP: small_mlp_cpu" ), std::string::npos );
    }

    TEST_F( MLPCpuTests, TrainingMode_Default )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );
        EXPECT_FALSE( data.mlp->isTraining() );
    }

    TEST_F( MLPCpuTests, TrainingMode_Enabled )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "training_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );
        data.mlp->build( data.input_shape );
        data.mlp->setTraining( true );

        EXPECT_TRUE( data.mlp->isTraining() );
    }

    TEST_F( MLPCpuTests, GetComponents_Returns_ChildComponents )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "small_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );

        auto components = data.mlp->getComponents();

        EXPECT_GE( components.size(), 3u );
    }

    TEST_F( MLPCpuTests, NoBias_ParameterCount )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "no_bias_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            false );

        ASSERT_NE( data.mlp, nullptr );

        data.mlp->build( data.input_shape );

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
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "no_bias_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            false );

        ASSERT_NE( data.mlp, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        data.mlp->build( data.input_shape );

        TensorType input( data.mlp->getDeviceId(), data.input_shape );

        random( input, -1.0f, 1.0f );

        TensorType* out_ptr = nullptr;

        EXPECT_NO_THROW( { auto& out_ref = data.mlp->forward( input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );

        auto* out_tensor = out_ptr;
        ASSERT_NE( out_tensor, nullptr );

        EXPECT_EQ( out_tensor->size(), input.size() );
    }

    TEST_F( MLPCpuTests, LayerNorm_Forward )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "layer_norm_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            true,
            ActivationType::Gelu,
            true );

        ASSERT_NE( data.mlp, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        data.mlp->build( data.input_shape );

        TensorType input( data.mlp->getDeviceId(), data.input_shape );

        random( input, -1.0f, 1.0f );

        TensorType* out_ptr = nullptr;

        EXPECT_NO_THROW( { auto& out_ref = data.mlp->forward( input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );

        auto* out_tensor = out_ptr;
        ASSERT_NE( out_tensor, nullptr );

        EXPECT_EQ( out_tensor->shape(), input.shape() );
    }

    TEST_F( MLPCpuTests, LayerNorm_SubModules )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "layer_norm_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            true,
            ActivationType::Gelu,
            true );

        ASSERT_NE( data.mlp, nullptr );

        auto components = data.mlp->getComponents();
    }

    TEST_F( MLPCpuTests, Training_Forward )
    {
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "training_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );
        data.mlp->build( data.input_shape );

        data.mlp->setTraining( true );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        TensorType input( data.mlp->getDeviceId(), data.input_shape );

        random( input, -1.0f, 1.0f );

        TensorType* out_ptr = nullptr;

        EXPECT_NO_THROW( { auto& out_ref = data.mlp->forward( input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );
    }

    TEST_F( MLPCpuTests, WithContext_Construction )
    {
        // Constructing with DeviceId should create and bind an execution context.
        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "context_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp, nullptr );

        EXPECT_EQ( data.mlp->getName(), "context_mlp_cpu" );
        EXPECT_EQ( data.mlp->getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( MLPCpuTests, EdgeCase_MinimalShape )
    {
        shape_t shape = { 1, 1, 8 };
        int64_t hidden = 16;

        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "minimal_cpu", shape, 8, hidden );

        ASSERT_NE( data.mlp, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        data.mlp->build( data.input_shape );

        TensorType input( data.mlp->getDeviceId(), data.input_shape );

        random( input, -1.0f, 1.0f );

        TensorType* out_ptr = nullptr;

        EXPECT_NO_THROW( { auto& out_ref = data.mlp->forward( input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );
    }

    TEST_F( MLPCpuTests, EdgeCase_MediumShape )
    {
        shape_t shape = { 1, 2, 512 };
        int64_t hidden = 2048;

        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "medium_cpu", shape, 512, hidden );

        ASSERT_NE( data.mlp, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        data.mlp->build( data.input_shape );

        TensorType input( data.mlp->getDeviceId(), data.input_shape );

        random( input, -1.0f, 1.0f );

        TensorType* out_ptr = nullptr;

        EXPECT_NO_THROW( { auto& out_ref = data.mlp->forward( input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );
    }

    TEST_F( MLPCpuTests, Error_InvalidConfiguration_ZeroInputFeatures )
    {
        MLPConfig invalid_config( 0, 1024 );

        EXPECT_THROW(
            (MLP<DeviceType::Cpu, TensorDataType::FP32>( "invalid", invalid_config, Device::Cpu() )),
            std::invalid_argument
        );
    }

    TEST_F( MLPCpuTests, Error_InvalidConfiguration_ZeroHiddenSize )
    {
        MLPConfig invalid_config( 768, 0 );

        EXPECT_THROW(
            (MLP<DeviceType::Cpu, TensorDataType::FP32>( "invalid", invalid_config, Device::Cpu() )),
            std::invalid_argument
        );
    }

    TEST_F( MLPCpuTests, Error_NullExecutionContext )
    {
        // Shared-mode construction (no DeviceId) is allowed — but build should fail
        // because no execution context is set on the component or its children.
        MLPConfig config( 768, 3072 );

        auto mlp = std::make_shared<MLP<DeviceType::Cpu, TensorDataType::FP32>>( "null_context_test", config, std::nullopt );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), shape_t{ 2, 16, 768 } );

        EXPECT_THROW(
            mlp->build( shape_t{ 2, 16, 768 } ),
            std::runtime_error
        );
    }

    TEST_F( MLPCpuTests, Error_ForwardBeforeBuild )
    {
        shape_t test_shape = { 2, 16, 768 };

        MLPConfig config( 768, 3072 );

        auto mlp = std::make_shared<MLP<DeviceType::Cpu, TensorDataType::FP32>>( "unbuild_test", config, Device::Cpu() );

        CpuTensor<TensorDataType::FP32> input( mlp->getDeviceId(), test_shape );

        EXPECT_THROW(
            mlp->forward( input ),
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

        data.mlp->build( data.input_shape );

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

        data.mlp->build( data.input_shape );

        CpuTensor<TensorDataType::FP32> input( data.mlp->getDeviceId(), data.input_shape );

        for ( int iter = 0; iter < 5; ++iter )
        {
            random( input, -1.0f, 1.0f );

            Tensor<TensorDataType::FP32, CpuMemoryResource>* out_ptr = nullptr;
            EXPECT_NO_THROW( { auto& out_ref = data.mlp->forward( input ); out_ptr = &out_ref; } );
            ASSERT_NE( out_ptr, nullptr );
        }
    }
}