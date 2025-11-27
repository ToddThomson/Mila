#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <exception>
#include <cstdint>
#include <cuda_runtime.h>

import Mila;

namespace Modules::Blocks::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    template<TensorDataType TPrecision>
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    template<TensorDataType TPrecision>
    struct MLPCudaTestData
    {
        MLPConfig config;
        std::shared_ptr<MLP<DeviceType::Cuda, TPrecision>> mlp_module;
        shape_t input_shape;
        int64_t input_features;
        int64_t hidden_size;
        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> exec_context;

        MLPCudaTestData() : config( 1, 1 ), input_features( 0 ), hidden_size( 0 )
        {
        }

        static MLPCudaTestData Create(
            const std::string& name,
            const shape_t& input_shape,
            int64_t input_features,
            int64_t hidden_size,
            bool has_bias = true,
            ActivationType activation = ActivationType::Gelu,
            bool use_layer_norm = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            MLPCudaTestData data;
            data.input_shape = input_shape;
            data.input_features = input_features;
            data.hidden_size = hidden_size;

            data.config = MLPConfig( input_features, hidden_size );
            data.config.withBias( has_bias )
                .withActivation( activation )
                .withLayerNorm( use_layer_norm )
                .withName( name )
                .withPrecisionPolicy( precision );

            data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
            data.mlp_module = std::make_shared<MLP<DeviceType::Cuda, TPrecision>>( data.exec_context, data.config );

            return data;
        }

        static MLPCudaTestData CreateWithContext(
            const std::string& name,
            const shape_t& input_shape,
            int64_t input_features,
            int64_t hidden_size,
            std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
            bool has_bias = true,
            ActivationType activation = ActivationType::Gelu,
            bool use_layer_norm = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            MLPCudaTestData data;
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
            data.mlp_module = std::make_shared<MLP<DeviceType::Cuda, TPrecision>>( context, data.config );

            return data;
        }
    };

    class MLPCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = 0;
            cudaError_t error = cudaGetDeviceCount( &device_count );
            cuda_available_ = (error == cudaSuccess && device_count > 0);

            if ( !cuda_available_ )
            {
                return;
            }

            batch_size_ = 16;
            sequence_length_ = 64;
            input_features_ = 768;
            hidden_size_ = 3072;
        }

        bool cuda_available_{ false };

        int64_t batch_size_{ 0 };
        int64_t sequence_length_{ 0 };
        int64_t input_features_{ 0 };
        int64_t hidden_size_{ 0 };
    };

    TEST_F( MLPCudaTests, GetName )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "small_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );
        EXPECT_EQ( data.mlp_module->getName(), "small_mlp_cuda" );
    }

    TEST_F( MLPCudaTests, DeviceType )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "small_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.exec_context, nullptr );
        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );
    }

    TEST_F( MLPCudaTests, IsBuilt_BeforeBuild )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "small_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );
        EXPECT_FALSE( data.mlp_module->isBuilt() );
    }

    TEST_F( MLPCudaTests, IsBuilt_AfterBuild )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "small_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );
        EXPECT_NO_THROW( data.mlp_module->build( data.input_shape ) );
        EXPECT_TRUE( data.mlp_module->isBuilt() );
    }

    TEST_F( MLPCudaTests, Build )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "small_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );
        EXPECT_NO_THROW( data.mlp_module->build( data.input_shape ) );
        EXPECT_TRUE( data.mlp_module->isBuilt() );
    }

    TEST_F( MLPCudaTests, ParameterCount )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "small_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );

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

        EXPECT_EQ( data.mlp_module->parameterCount(), expected_total_params );
    }

    TEST_F( MLPCudaTests, Forward )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "small_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        using DeviceTensorType = CudaTensor<TensorDataType::FP32>;
        using HostTensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.mlp_module->build( data.input_shape ) );

        HostTensorType host_input( "CPU", data.input_shape );
        random( host_input, -1.0f, 1.0f );

        DeviceTensorType device_input( data.exec_context->getDevice(), data.input_shape );
        DeviceTensorType device_output( data.exec_context->getDevice(), data.input_shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.mlp_module->forward( device_input, device_output ) );

        EXPECT_EQ( device_output.size(), device_input.size() );
        EXPECT_EQ( device_output.shape(), device_input.shape() );

        HostTensorType host_output = toHost<TensorDataType::FP32>( device_output );
        EXPECT_EQ( host_output.size(), device_input.size() );
    }

    TEST_F( MLPCudaTests, ToString )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "small_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );
        std::string output = data.mlp_module->toString();
        EXPECT_NE( output.find( "MLP: small_mlp_cuda" ), std::string::npos );
    }

    TEST_F( MLPCudaTests, TrainingMode_Default )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "small_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );
        EXPECT_FALSE( data.mlp_module->isTraining() );
    }

    TEST_F( MLPCudaTests, TrainingMode_Enabled )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "training_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );
        data.mlp_module->setTraining( true );
        EXPECT_TRUE( data.mlp_module->isTraining() );
    }

    TEST_F( MLPCudaTests, GetNamedComponents_Returns_ChildComponents )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "small_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );

        auto modules = data.mlp_module->getNamedComponents();

        EXPECT_GE( modules.size(), 3u );

        const std::string base = data.mlp_module->getName();

        EXPECT_NE( modules.find( base + ".fc1" ), modules.end() );
        EXPECT_NE( modules.find( base + ".act" ), modules.end() );
        EXPECT_NE( modules.find( base + ".fc2" ), modules.end() );

        if ( data.config.useLayerNorm() )
        {
            EXPECT_NE( modules.find( base + ".norm" ), modules.end() );
        }
    }

    TEST_F( MLPCudaTests, SaveLoad )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "small_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );

        // Save/load not implemented; test kept minimal to avoid regressions.
    }

    TEST_F( MLPCudaTests, NoBias_ParameterCount )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "no_bias_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            false );

        ASSERT_NE( data.mlp_module, nullptr );

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

        EXPECT_EQ( data.mlp_module->parameterCount(), expected_total_params );
    }

    TEST_F( MLPCudaTests, NoBias_Forward )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "no_bias_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            false );

        ASSERT_NE( data.mlp_module, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        using DeviceTensorType = CudaTensor<TensorDataType::FP32>;
        using HostTensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.mlp_module->build( data.input_shape ) );

        HostTensorType host_input( "CPU", data.input_shape );
        random( host_input, -1.0f, 1.0f );

        DeviceTensorType device_input( data.exec_context->getDevice(), data.input_shape );
        DeviceTensorType device_output( data.exec_context->getDevice(), data.input_shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.mlp_module->forward( device_input, device_output ) );
        EXPECT_EQ( device_output.size(), device_input.size() );
    }

    TEST_F( MLPCudaTests, LayerNorm_Forward )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "layer_norm_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            true,
            ActivationType::Gelu,
            true );

        ASSERT_NE( data.mlp_module, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        using DeviceTensorType = CudaTensor<TensorDataType::FP32>;
        using HostTensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.mlp_module->build( data.input_shape ) );

        HostTensorType host_input( "CPU", data.input_shape );
        random( host_input, -1.0f, 1.0f );

        DeviceTensorType device_input( data.exec_context->getDevice(), data.input_shape );
        DeviceTensorType device_output( data.exec_context->getDevice(), data.input_shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.mlp_module->forward( device_input, device_output ) );
        EXPECT_EQ( device_output.shape(), device_input.shape() );
    }

    TEST_F( MLPCudaTests, LayerNorm_SubModules )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "layer_norm_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            true,
            ActivationType::Gelu,
            true );

        ASSERT_NE( data.mlp_module, nullptr );

        auto modules = data.mlp_module->getNamedComponents();
        const std::string base = data.mlp_module->getName();

        EXPECT_NE( modules.find( base + ".fc1" ), modules.end() );
        EXPECT_NE( modules.find( base + ".act" ), modules.end() );
        EXPECT_NE( modules.find( base + ".fc2" ), modules.end() );
        EXPECT_NE( modules.find( base + ".norm" ), modules.end() );
    }

    TEST_F( MLPCudaTests, Training_Forward )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "training_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        data.mlp_module->setTraining( true );

        using DeviceTensorType = CudaTensor<TensorDataType::FP32>;
        using HostTensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.mlp_module->build( data.input_shape ) );

        HostTensorType host_input( "CPU", data.input_shape );
        random( host_input, -1.0f, 1.0f );

        DeviceTensorType device_input( data.exec_context->getDevice(), data.input_shape );
        DeviceTensorType device_output( data.exec_context->getDevice(), data.input_shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.mlp_module->forward( device_input, device_output ) );
    }

    TEST_F( MLPCudaTests, FP16_Forward )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP16>::Create(
            "small_mlp_cuda_fp16",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        using DeviceTensorType = CudaTensor<TensorDataType::FP16>;
        using HostTensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.mlp_module->build( data.input_shape ) );

        HostTensorType host_input( "CPU", data.input_shape );
        random( host_input, -1.0f, 1.0f );

        DeviceTensorType device_input( data.exec_context->getDevice(), data.input_shape );
        DeviceTensorType device_output( data.exec_context->getDevice(), data.input_shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.mlp_module->forward( device_input, device_output ) );
    }

    TEST_F( MLPCudaTests, Precision_Policies_Run )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto perf = MLPCudaTestData<TensorDataType::FP32>::Create(
            "perf_precision_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            true,
            ActivationType::Gelu,
            false,
            ComputePrecision::Policy::Performance );

        auto acc = MLPCudaTestData<TensorDataType::FP32>::Create(
            "accuracy_precision_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            true,
            ActivationType::Gelu,
            false,
            ComputePrecision::Policy::Accuracy );

        auto native = MLPCudaTestData<TensorDataType::FP32>::Create(
            "native_precision_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            true,
            ActivationType::Gelu,
            false,
            ComputePrecision::Policy::Native );

        ASSERT_NE( perf.mlp_module, nullptr );
        ASSERT_NE( acc.mlp_module, nullptr );
        ASSERT_NE( native.mlp_module, nullptr );

        // Run simple forward to ensure construction and forward pipeline works
        {
            using HostTensorType = CpuTensor<TensorDataType::FP32>;
            using DeviceTensorType = CudaTensor<TensorDataType::FP32>;

            EXPECT_NO_THROW( perf.mlp_module->build( perf.input_shape ) );
            HostTensorType host_input( "CPU", perf.input_shape );
            random( host_input, -1.0f, 1.0f );
            DeviceTensorType device_input( perf.exec_context->getDevice(), perf.input_shape );
            DeviceTensorType device_output( perf.exec_context->getDevice(), perf.input_shape );
            copy( host_input, device_input );
            EXPECT_NO_THROW( perf.mlp_module->forward( device_input, device_output ) );
        }

        {
            using HostTensorType = CpuTensor<TensorDataType::FP32>;
            using DeviceTensorType = CudaTensor<TensorDataType::FP32>;

            EXPECT_NO_THROW( acc.mlp_module->build( acc.input_shape ) );
            HostTensorType host_input( "CPU", acc.input_shape );
            random( host_input, -1.0f, 1.0f );
            DeviceTensorType device_input( acc.exec_context->getDevice(), acc.input_shape );
            DeviceTensorType device_output( acc.exec_context->getDevice(), acc.input_shape );
            copy( host_input, device_input );
            EXPECT_NO_THROW( acc.mlp_module->forward( device_input, device_output ) );
        }

        {
            using HostTensorType = CpuTensor<TensorDataType::FP32>;
            using DeviceTensorType = CudaTensor<TensorDataType::FP32>;

            EXPECT_NO_THROW( native.mlp_module->build( native.input_shape ) );
            HostTensorType host_input( "CPU", native.input_shape );
            random( host_input, -1.0f, 1.0f );
            DeviceTensorType device_input( native.exec_context->getDevice(), native.input_shape );
            DeviceTensorType device_output( native.exec_context->getDevice(), native.input_shape );
            copy( host_input, device_input );
            EXPECT_NO_THROW( native.mlp_module->forward( device_input, device_output ) );
        }
    }

    TEST_F( MLPCudaTests, WithContext_Construction )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        auto data = MLPCudaTestData<TensorDataType::FP32>::CreateWithContext(
            "context_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            ctx );

        ASSERT_NE( data.mlp_module, nullptr );
        EXPECT_EQ( data.mlp_module->getName(), "context_mlp_cuda" );
        EXPECT_EQ( data.exec_context, ctx );
    }

    TEST_F( MLPCudaTests, EdgeCase_MinimalShape )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        shape_t shape = { 1, 1, 8 };
        int64_t hidden = 16;

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "minimal_cuda", shape, 8, hidden );

        ASSERT_NE( data.mlp_module, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        using DeviceTensorType = CudaTensor<TensorDataType::FP32>;
        using HostTensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.mlp_module->build( data.input_shape ) );

        HostTensorType host_input( "CPU", data.input_shape );
        random( host_input, -1.0f, 1.0f );

        DeviceTensorType device_input( data.exec_context->getDevice(), data.input_shape );
        DeviceTensorType device_output( data.exec_context->getDevice(), data.input_shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.mlp_module->forward( device_input, device_output ) );
    }

    TEST_F( MLPCudaTests, EdgeCase_MediumShape )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        shape_t shape = { 2, 2, 1024 };
        int64_t hidden = 2048;

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "medium_cuda", shape, 1024, hidden );

        ASSERT_NE( data.mlp_module, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        using DeviceTensorType = CudaTensor<TensorDataType::FP32>;
        using HostTensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.mlp_module->build( data.input_shape ) );

        HostTensorType host_input( "CPU", data.input_shape );
        random( host_input, -1.0f, 1.0f );

        DeviceTensorType device_input( data.exec_context->getDevice(), data.input_shape );
        DeviceTensorType device_output( data.exec_context->getDevice(), data.input_shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.mlp_module->forward( device_input, device_output ) );
    }

    TEST_F( MLPCudaTests, Error_InvalidConfiguration_ZeroInputFeatures )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        MLPConfig invalid_config( 0, 1024 );

        auto cuda_exec = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        EXPECT_THROW(
            (MLP<DeviceType::Cuda, TensorDataType::FP32>( cuda_exec, invalid_config )),
            std::invalid_argument
        );
    }

    TEST_F( MLPCudaTests, Error_InvalidConfiguration_ZeroHiddenSize )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        MLPConfig invalid_config( 768, 0 );

        auto cuda_exec = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        EXPECT_THROW(
            (MLP<DeviceType::Cuda, TensorDataType::FP32>( cuda_exec, invalid_config )),
            std::invalid_argument
        );
    }

    TEST_F( MLPCudaTests, Error_NullExecutionContext )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        MLPConfig config( 768, 3072 );
        config.withName( "null_context_test" );

        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> null_ctx;

        EXPECT_THROW(
            (MLP<DeviceType::Cuda, TensorDataType::FP32>( null_ctx, config )),
            std::invalid_argument
        );
    }

    TEST_F( MLPCudaTests, Error_ForwardBeforeBuild )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        shape_t test_shape = { 2, 16, 768 };

        MLPConfig config( 768, 3072 );
        config.withName( "unbuild_test" );

        auto cuda_exec = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto mlp = std::make_shared<MLP<DeviceType::Cuda, TensorDataType::FP32>>( cuda_exec, config );

        CudaTensor<TensorDataType::FP32> input( cuda_exec->getDevice(), test_shape );
        CudaTensor<TensorDataType::FP32> output( cuda_exec->getDevice(), test_shape );

        EXPECT_THROW(
            mlp->forward( input, output ),
            std::runtime_error
        );
    }

    TEST_F( MLPCudaTests, Synchronize )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "small_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );

        EXPECT_NO_THROW( data.mlp_module->synchronize() );
    }

    TEST_F( MLPCudaTests, SetTrainingMode )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "small_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );

        EXPECT_FALSE( data.mlp_module->isTraining() );

        data.mlp_module->setTraining( true );
        EXPECT_TRUE( data.mlp_module->isTraining() );

        data.mlp_module->setTraining( false );
        EXPECT_FALSE( data.mlp_module->isTraining() );
    }

    TEST_F( MLPCudaTests, MultipleForwardCalls )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "small_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_ );

        ASSERT_NE( data.mlp_module, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        EXPECT_NO_THROW( data.mlp_module->build( data.input_shape ) );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", data.input_shape );
        CudaTensor<TensorDataType::FP32> device_input( data.exec_context->getDevice(), data.input_shape );
        CudaTensor<TensorDataType::FP32> device_output( data.exec_context->getDevice(), data.input_shape );

        for ( int iter = 0; iter < 10; ++iter )
        {
            random( host_input, -1.0f, 1.0f );
            copy( host_input, device_input );

            EXPECT_NO_THROW( data.mlp_module->forward( device_input, device_output ) );
        }
    }

    TEST_F( MLPCudaTests, CpuCuda_OutputEquivalence )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        shape_t test_shape = { 1, 2, 768 };
        int64_t test_hidden_size = 1024;

        auto cpu_config = MLPConfig( 768, test_hidden_size );
        cpu_config.withName( "test_cpu_mlp" );
        auto cuda_config = MLPConfig( 768, test_hidden_size );
        cuda_config.withName( "test_cuda_mlp" );

        auto cpu_exec = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cuda_exec = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        auto cpu_mlp = std::make_shared<MLP<DeviceType::Cpu, TensorDataType::FP32>>( cpu_exec, cpu_config );
        auto cuda_mlp = std::make_shared<MLP<DeviceType::Cuda, TensorDataType::FP32>>( cuda_exec, cuda_config );

        cpu_mlp->build( test_shape );
        cuda_mlp->build( test_shape );

        CpuTensor<TensorDataType::FP32> host_input( cpu_exec->getDevice(), test_shape );

        for ( size_t i = 0; i < host_input.size(); ++i )
        {
            host_input.data()[i] = static_cast<float>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        CpuTensor<TensorDataType::FP32> cpu_output( cpu_exec->getDevice(), test_shape );
        cpu_mlp->forward( host_input, cpu_output );

        CudaTensor<TensorDataType::FP32> device_input( cuda_exec->getDevice(), test_shape );
        copy( host_input, device_input );

        CudaTensor<TensorDataType::FP32> cuda_output( cuda_exec->getDevice(), test_shape );
        cuda_mlp->forward( device_input, cuda_output );

        cuda_exec->synchronize();

        auto cuda_output_host = toHost<TensorDataType::FP32>( cuda_output, cuda_exec.get() );

        const float epsilon = 1e-3f;
        bool all_equal = true;

        for ( size_t i = 0; i < cpu_output.size(); ++i )
        {
            float diff = std::abs( static_cast<float>( cpu_output.data()[i] ) - static_cast<float>( cuda_output_host.data()[i] ) );
            if ( diff > epsilon )
            {
                all_equal = false;
                break;
            }
        }

        EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
    }
}