#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <exception>
#include <cstdint>

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
        std::shared_ptr<MLP<DeviceType::Cpu, TPrecision>> mlp_module;
        shape_t input_shape;
        int64_t input_features;
        int64_t hidden_size;
        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> exec_context;

        MLPCpuTestData() : config( 1, 1 ), input_features( 0 ), hidden_size( 0 )
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
            data.mlp_module = std::make_shared<MLP<DeviceType::Cpu, TPrecision>>( data.exec_context, data.config );

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
            data.mlp_module = std::make_shared<MLP<DeviceType::Cpu, TPrecision>>( context, data.config );

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

        MLPCpuTestData<TensorDataType::FP32>& SmallFp32Data()
        {
            if (!small_fp32_.mlp_module)
            {
                small_fp32_ = MLPCpuTestData<TensorDataType::FP32>::Create(
                    "small_mlp_cpu",
                    shape_t{ batch_size_, sequence_length_, input_features_ },
                    input_features_,
                    hidden_size_ );
            }
            return small_fp32_;
        }

        MLPCpuTestData<TensorDataType::FP32>& TrainingFp32Data()
        {
            if (!training_fp32_.mlp_module)
            {
                training_fp32_ = MLPCpuTestData<TensorDataType::FP32>::Create(
                    "training_mlp_cpu",
                    shape_t{ batch_size_, sequence_length_, input_features_ },
                    input_features_,
                    hidden_size_ );

                training_fp32_.mlp_module->setTraining( true );
            }

            return training_fp32_;
        }

        MLPCpuTestData<TensorDataType::FP32>& NoBiasFp32Data()
        {
            if (!no_bias_fp32_.mlp_module)
            {
                no_bias_fp32_ = MLPCpuTestData<TensorDataType::FP32>::Create(
                    "no_bias_mlp_cpu",
                    shape_t{ batch_size_, sequence_length_, input_features_ },
                    input_features_,
                    hidden_size_,
                    false );
            }
            return no_bias_fp32_;
        }

        MLPCpuTestData<TensorDataType::FP32>& LayerNormFp32Data()
        {
            if (!layer_norm_fp32_.mlp_module)
            {
                layer_norm_fp32_ = MLPCpuTestData<TensorDataType::FP32>::Create(
                    "layer_norm_mlp_cpu",
                    shape_t{ batch_size_, sequence_length_, input_features_ },
                    input_features_,
                    hidden_size_,
                    true,
                    ActivationType::Gelu,
                    true );
            }
            return layer_norm_fp32_;
        }

        int64_t batch_size_{ 0 };
        int64_t sequence_length_{ 0 };
        int64_t input_features_{ 0 };
        int64_t hidden_size_{ 0 };

        MLPCpuTestData<TensorDataType::FP32> small_fp32_;
        MLPCpuTestData<TensorDataType::FP32> training_fp32_;
        MLPCpuTestData<TensorDataType::FP32> no_bias_fp32_;
        MLPCpuTestData<TensorDataType::FP32> layer_norm_fp32_;
    };

    template<TensorDataType TPrecision>
    void TestGetName( const MLPCpuTestData<TPrecision>& data, const std::string& expected_name )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        EXPECT_EQ( data.mlp_module->getName(), expected_name );
    }

    template<TensorDataType TPrecision>
    void TestDeviceType( const MLPCpuTestData<TPrecision>& data )
    {
        ASSERT_NE( data.exec_context, nullptr );
        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
    }

    template<TensorDataType TPrecision>
    void TestIsBuilt( const MLPCpuTestData<TPrecision>& data, bool expected_built )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        EXPECT_EQ( data.mlp_module->isBuilt(), expected_built );
    }

    template<TensorDataType TPrecision>
    void TestBuild( MLPCpuTestData<TPrecision>& data )
    {
        ASSERT_NE( data.mlp_module, nullptr );

        EXPECT_NO_THROW( data.mlp_module->build( data.input_shape ) );
        EXPECT_TRUE( data.mlp_module->isBuilt() );

        data.mlp_module->build( data.input_shape );
        EXPECT_TRUE( data.mlp_module->isBuilt() );
    }

    template<TensorDataType TPrecision>
    void TestParameterCount( const MLPCpuTestData<TPrecision>& data )
    {
        ASSERT_NE( data.mlp_module, nullptr );

        int64_t input_features = data.config.getInputFeatures();
        int64_t hidden_size = data.config.getHiddenSize();
        bool has_bias = data.config.hasBias();

        size_t expected_fc1_params = input_features * hidden_size;
        size_t expected_fc2_params = hidden_size * input_features;

        if (has_bias)
        {
            expected_fc1_params += hidden_size;
            expected_fc2_params += input_features;
        }

        size_t expected_total_params = expected_fc1_params + expected_fc2_params;

        EXPECT_EQ( data.mlp_module->parameterCount(), expected_total_params );
    }

    template<TensorDataType TPrecision>
    void TestForward( MLPCpuTestData<TPrecision>& data )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        using TensorType = CpuTensor<TPrecision>;

        data.mlp_module->build( data.input_shape );

        TensorType input( data.exec_context->getDevice(), data.input_shape );
        TensorType output( data.exec_context->getDevice(), data.input_shape );

        random( input, -1.0f, 1.0f );

        EXPECT_NO_THROW( data.mlp_module->forward( input, output ) );

        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    template<TensorDataType TPrecision>
    void TestToString( const MLPCpuTestData<TPrecision>& data, const std::string& expected_substring )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        std::string output = data.mlp_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<TensorDataType TPrecision>
    void TestTrainingMode( const MLPCpuTestData<TPrecision>& data, bool expected_mode )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        EXPECT_EQ( data.mlp_module->isTraining(), expected_mode );
    }

    template<TensorDataType TPrecision>
    void TestSubModules( const MLPCpuTestData<TPrecision>& data )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        auto modules = data.mlp_module->getNamedModules();

        EXPECT_GE( modules.size(), 3 );
        EXPECT_NE( modules.find( "fc1" ), modules.end() );
        EXPECT_NE( modules.find( "activation" ), modules.end() );
        EXPECT_NE( modules.find( "fc2" ), modules.end() );

        if (data.config.useLayerNorm())
        {
            EXPECT_NE( modules.find( "norm" ), modules.end() );
        }
    }

    template<TensorDataType TPrecision>
    void TestSaveLoad( const MLPCpuTestData<TPrecision>& data )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        //ModelArchive archive;
		//SerializationMode mode = SerializationMode

  //      EXPECT_NO_THROW( data.mlp_module->save( archive ) );
  //      EXPECT_NO_THROW( data.mlp_module->load( archive ) );
    }

    TEST_F( MLPCpuTests, GetName )
    {
        TestGetName( SmallFp32Data(), "small_mlp_cpu" );
    }

    TEST_F( MLPCpuTests, DeviceType )
    {
        TestDeviceType( SmallFp32Data() );
    }

    TEST_F( MLPCpuTests, IsBuilt_BeforeBuild )
    {
        TestIsBuilt( SmallFp32Data(), false );
    }

    TEST_F( MLPCpuTests, IsBuilt_AfterBuild )
    {
        auto data = SmallFp32Data();

        EXPECT_FALSE( data.mlp_module->isBuilt() );

        data.mlp_module->build( data.input_shape );

        EXPECT_TRUE( data.mlp_module->isBuilt() );
    }

    TEST_F( MLPCpuTests, Build )
    {
        auto data = SmallFp32Data();
        TestBuild( data );
    }

    TEST_F( MLPCpuTests, ParameterCount )
    {
        TestParameterCount( SmallFp32Data() );
    }

    TEST_F( MLPCpuTests, Forward )
    {
        auto data = SmallFp32Data();
        TestForward( data );
    }

    TEST_F( MLPCpuTests, ToString )
    {
        TestToString( SmallFp32Data(), "MLP: small_mlp_cpu" );
    }

    TEST_F( MLPCpuTests, TrainingMode_Default )
    {
        TestTrainingMode( SmallFp32Data(), false );
    }

    TEST_F( MLPCpuTests, TrainingMode_Enabled )
    {
        TestTrainingMode( TrainingFp32Data(), true );
    }

    TEST_F( MLPCpuTests, SubModules )
    {
        TestSubModules( SmallFp32Data() );
    }

    TEST_F( MLPCpuTests, SaveLoad )
    {
        TestSaveLoad( SmallFp32Data() );
    }

    TEST_F( MLPCpuTests, NoBias_ParameterCount )
    {
        TestParameterCount( NoBiasFp32Data() );
    }

    TEST_F( MLPCpuTests, NoBias_Forward )
    {
        auto data = NoBiasFp32Data();
        TestForward( data );
    }

    TEST_F( MLPCpuTests, LayerNorm_Forward )
    {
        auto data = LayerNormFp32Data();
        TestForward( data );
    }

    TEST_F( MLPCpuTests, LayerNorm_SubModules )
    {
        TestSubModules( LayerNormFp32Data() );
    }

    TEST_F( MLPCpuTests, Training_Forward )
    {
        auto data = TrainingFp32Data();
        TestForward( data );
    }

    TEST_F( MLPCpuTests, WithContext_Construction )
    {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        auto data = MLPCpuTestData<TensorDataType::FP32>::CreateWithContext(
            "context_mlp_cpu",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            ctx );

        EXPECT_EQ( data.mlp_module->getName(), "context_mlp_cpu" );
        EXPECT_EQ( data.exec_context, ctx );
    }

    TEST_F( MLPCpuTests, EdgeCase_MinimalShape )
    {
        shape_t shape = { 1, 1, 8 };
        int64_t hidden = 16;

        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "minimal_cpu", shape, 8, hidden );

        TestForward( data );
    }

    TEST_F( MLPCpuTests, EdgeCase_MediumShape )
    {
        shape_t shape = { 1, 2, 512 };
        int64_t hidden = 2048;

        auto data = MLPCpuTestData<TensorDataType::FP32>::Create(
            "medium_cpu", shape, 512, hidden );

        TestForward( data );
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
        auto data = SmallFp32Data();

        EXPECT_NO_THROW( data.mlp_module->synchronize() );
    }

    TEST_F( MLPCpuTests, SetTrainingMode )
    {
        auto data = SmallFp32Data();

        EXPECT_FALSE( data.mlp_module->isTraining() );

        data.mlp_module->setTraining( true );
        EXPECT_TRUE( data.mlp_module->isTraining() );

        data.mlp_module->setTraining( false );
        EXPECT_FALSE( data.mlp_module->isTraining() );
    }

    TEST_F( MLPCpuTests, MultipleForwardCalls )
    {
        auto data = SmallFp32Data();
        data.mlp_module->build( data.input_shape );

        CpuTensor<TensorDataType::FP32> input( data.exec_context->getDevice(), data.input_shape );
        CpuTensor<TensorDataType::FP32> output( data.exec_context->getDevice(), data.input_shape );

        for (int iter = 0; iter < 10; ++iter)
        {
            random( input, -1.0f, 1.0f );

            EXPECT_NO_THROW( data.mlp_module->forward( input, output ) );
        }
    }
}