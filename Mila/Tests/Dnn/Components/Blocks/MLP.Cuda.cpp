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

            if (!cuda_available_)
            {
                return;
            }

            batch_size_ = 16;
            sequence_length_ = 64;
            input_features_ = 768;
            hidden_size_ = 3072;
        }

        MLPCudaTestData<TensorDataType::FP32>& SmallFp32Data()
        {
            if (!small_fp32_.mlp_module)
            {
                small_fp32_ = MLPCudaTestData<TensorDataType::FP32>::Create(
                    "small_mlp_cuda",
                    shape_t{ batch_size_, sequence_length_, input_features_ },
                    input_features_,
                    hidden_size_ );
            }
            return small_fp32_;
        }

        MLPCudaTestData<TensorDataType::FP32>& TrainingFp32Data()
        {
            if (!training_fp32_.mlp_module)
            {
                training_fp32_ = MLPCudaTestData<TensorDataType::FP32>::Create(
                    "training_mlp_cuda",
                    shape_t{ batch_size_, sequence_length_, input_features_ },
                    input_features_,
                    hidden_size_ );

                training_fp32_.mlp_module->setTraining( true );
            }
            return training_fp32_;
        }

        MLPCudaTestData<TensorDataType::FP32>& NoBiasFp32Data()
        {
            if (!no_bias_fp32_.mlp_module)
            {
                no_bias_fp32_ = MLPCudaTestData<TensorDataType::FP32>::Create(
                    "no_bias_mlp_cuda",
                    shape_t{ batch_size_, sequence_length_, input_features_ },
                    input_features_,
                    hidden_size_,
                    false );
            }
            return no_bias_fp32_;
        }

        MLPCudaTestData<TensorDataType::FP32>& LayerNormFp32Data()
        {
            if (!layer_norm_fp32_.mlp_module)
            {
                layer_norm_fp32_ = MLPCudaTestData<TensorDataType::FP32>::Create(
                    "layer_norm_mlp_cuda",
                    shape_t{ batch_size_, sequence_length_, input_features_ },
                    input_features_,
                    hidden_size_,
                    true,
                    ActivationType::Gelu,
                    true );
            }
            return layer_norm_fp32_;
        }

        MLPCudaTestData<TensorDataType::FP16>& SmallFp16Data()
        {
            if (!small_fp16_.mlp_module)
            {
                small_fp16_ = MLPCudaTestData<TensorDataType::FP16>::Create(
                    "small_mlp_cuda_fp16",
                    shape_t{ batch_size_, sequence_length_, input_features_ },
                    input_features_,
                    hidden_size_ );
            }
            return small_fp16_;
        }

        MLPCudaTestData<TensorDataType::FP32>& PerfPrecisionFp32Data()
        {
            if (!perf_precision_fp32_.mlp_module)
            {
                perf_precision_fp32_ = MLPCudaTestData<TensorDataType::FP32>::Create(
                    "perf_precision_mlp_cuda",
                    shape_t{ batch_size_, sequence_length_, input_features_ },
                    input_features_,
                    hidden_size_,
                    true,
                    ActivationType::Gelu,
                    false,
                    ComputePrecision::Policy::Performance );
            }
            return perf_precision_fp32_;
        }

        MLPCudaTestData<TensorDataType::FP32>& AccuracyPrecisionFp32Data()
        {
            if (!accuracy_precision_fp32_.mlp_module)
            {
                accuracy_precision_fp32_ = MLPCudaTestData<TensorDataType::FP32>::Create(
                    "accuracy_precision_mlp_cuda",
                    shape_t{ batch_size_, sequence_length_, input_features_ },
                    input_features_,
                    hidden_size_,
                    true,
                    ActivationType::Gelu,
                    false,
                    ComputePrecision::Policy::Accuracy );
            }
            return accuracy_precision_fp32_;
        }

        MLPCudaTestData<TensorDataType::FP32>& NativePrecisionFp32Data()
        {
            if (!native_precision_fp32_.mlp_module)
            {
                native_precision_fp32_ = MLPCudaTestData<TensorDataType::FP32>::Create(
                    "native_precision_mlp_cuda",
                    shape_t{ batch_size_, sequence_length_, input_features_ },
                    input_features_,
                    hidden_size_,
                    true,
                    ActivationType::Gelu,
                    false,
                    ComputePrecision::Policy::Native );
            }
            return native_precision_fp32_;
        }

        bool cuda_available_{ false };

        int64_t batch_size_{ 0 };
        int64_t sequence_length_{ 0 };
        int64_t input_features_{ 0 };
        int64_t hidden_size_{ 0 };

        MLPCudaTestData<TensorDataType::FP32> small_fp32_;
        MLPCudaTestData<TensorDataType::FP32> training_fp32_;
        MLPCudaTestData<TensorDataType::FP32> no_bias_fp32_;
        MLPCudaTestData<TensorDataType::FP32> layer_norm_fp32_;
        MLPCudaTestData<TensorDataType::FP16> small_fp16_;
        MLPCudaTestData<TensorDataType::FP32> perf_precision_fp32_;
        MLPCudaTestData<TensorDataType::FP32> accuracy_precision_fp32_;
        MLPCudaTestData<TensorDataType::FP32> native_precision_fp32_;
    };

    template<TensorDataType TPrecision>
    void TestGetName( const MLPCudaTestData<TPrecision>& data, const std::string& expected_name )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        EXPECT_EQ( data.mlp_module->getName(), expected_name );
    }

    template<TensorDataType TPrecision>
    void TestDeviceType( const MLPCudaTestData<TPrecision>& data )
    {
        ASSERT_NE( data.exec_context, nullptr );
        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );
    }

    template<TensorDataType TPrecision>
    void TestIsBuilt( const MLPCudaTestData<TPrecision>& data, bool expected_built )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        EXPECT_EQ( data.mlp_module->isBuilt(), expected_built );
    }

    template<TensorDataType TPrecision>
    void TestBuild( MLPCudaTestData<TPrecision>& data )
    {
        ASSERT_NE( data.mlp_module, nullptr );

        EXPECT_NO_THROW( data.mlp_module->build( data.input_shape ) );
        EXPECT_TRUE( data.mlp_module->isBuilt() );

        data.mlp_module->build( data.input_shape );
        EXPECT_TRUE( data.mlp_module->isBuilt() );
    }

    template<TensorDataType TPrecision>
    void TestParameterCount( const MLPCudaTestData<TPrecision>& data )
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
    void TestForward( MLPCudaTestData<TPrecision>& data )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        ASSERT_NE( data.exec_context, nullptr );

        using DeviceTensorType = CudaTensor<TPrecision>;
        using HostTensorType = CpuTensor<TensorDataType::FP32>;

        data.mlp_module->build( data.input_shape );

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

    template<TensorDataType TPrecision>
    void TestToString( const MLPCudaTestData<TPrecision>& data, const std::string& expected_substring )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        std::string output = data.mlp_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<TensorDataType TPrecision>
    void TestTrainingMode( const MLPCudaTestData<TPrecision>& data, bool expected_mode )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        EXPECT_EQ( data.mlp_module->isTraining(), expected_mode );
    }

    template<TensorDataType TPrecision>
    void TestSubModules( const MLPCudaTestData<TPrecision>& data )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        auto modules = data.mlp_module->getNamedComponents();

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
    void TestSaveLoad( const MLPCudaTestData<TPrecision>& data )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        //ModelArchive archive;
        //EXPECT_NO_THROW( data.mlp_module->save( archive ) );
        //EXPECT_NO_THROW( data.mlp_module->load( archive ) );
    }

    TEST_F( MLPCudaTests, GetName )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetName( SmallFp32Data(), "small_mlp_cuda" );
    }

    TEST_F( MLPCudaTests, DeviceType )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestDeviceType( SmallFp32Data() );
    }

    TEST_F( MLPCudaTests, IsBuilt_BeforeBuild )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestIsBuilt( SmallFp32Data(), false );
    }

    TEST_F( MLPCudaTests, IsBuilt_AfterBuild )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_FALSE( data.mlp_module->isBuilt() );

        data.mlp_module->build( data.input_shape );

        EXPECT_TRUE( data.mlp_module->isBuilt() );
    }

    TEST_F( MLPCudaTests, Build )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestBuild( data );
    }

    TEST_F( MLPCudaTests, ParameterCount )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestParameterCount( SmallFp32Data() );
    }

    TEST_F( MLPCudaTests, Forward )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestForward( data );
    }

    TEST_F( MLPCudaTests, ToString )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestToString( SmallFp32Data(), "MLP: small_mlp_cuda" );
    }

    TEST_F( MLPCudaTests, TrainingMode_Default )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestTrainingMode( SmallFp32Data(), false );
    }

    TEST_F( MLPCudaTests, TrainingMode_Enabled )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestTrainingMode( TrainingFp32Data(), true );
    }

    TEST_F( MLPCudaTests, SubModules )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestSubModules( SmallFp32Data() );
    }

    TEST_F( MLPCudaTests, SaveLoad )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestSaveLoad( SmallFp32Data() );
    }

    TEST_F( MLPCudaTests, NoBias_ParameterCount )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestParameterCount( NoBiasFp32Data() );
    }

    TEST_F( MLPCudaTests, NoBias_Forward )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = NoBiasFp32Data();
        TestForward( data );
    }

    TEST_F( MLPCudaTests, LayerNorm_Forward )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormFp32Data();
        TestForward( data );
    }

    TEST_F( MLPCudaTests, LayerNorm_SubModules )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestSubModules( LayerNormFp32Data() );
    }

    TEST_F( MLPCudaTests, Training_Forward )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = TrainingFp32Data();
        TestForward( data );
    }

    TEST_F( MLPCudaTests, FP16_Forward )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp16Data();
        TestForward( data );
    }

    TEST_F( MLPCudaTests, PerformancePrecision_Policy )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = PerfPrecisionFp32Data();
        TestForward( data );
    }

    TEST_F( MLPCudaTests, AccuracyPrecision_Policy )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = AccuracyPrecisionFp32Data();
        TestForward( data );
    }

    TEST_F( MLPCudaTests, NativePrecision_Policy )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = NativePrecisionFp32Data();
        TestForward( data );
    }

    TEST_F( MLPCudaTests, WithContext_Construction )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        auto data = MLPCudaTestData<TensorDataType::FP32>::CreateWithContext(
            "context_mlp_cuda",
            shape_t{ batch_size_, sequence_length_, input_features_ },
            input_features_,
            hidden_size_,
            ctx );

        EXPECT_EQ( data.mlp_module->getName(), "context_mlp_cuda" );
        EXPECT_EQ( data.exec_context, ctx );
    }

    TEST_F( MLPCudaTests, EdgeCase_MinimalShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 1, 1, 8 };
        int64_t hidden = 16;

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "minimal_cuda", shape, 8, hidden );

        TestForward( data );
    }

    TEST_F( MLPCudaTests, EdgeCase_MediumShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 2, 2, 1024 };
        int64_t hidden = 2048;

        auto data = MLPCudaTestData<TensorDataType::FP32>::Create(
            "medium_cuda", shape, 1024, hidden );

        TestForward( data );
    }

    TEST_F( MLPCudaTests, Error_InvalidConfiguration_ZeroInputFeatures )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        MLPConfig invalid_config( 0, 1024 );

        auto cuda_exec = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        EXPECT_THROW(
            (MLP<DeviceType::Cuda, TensorDataType::FP32>( cuda_exec, invalid_config )),
            std::invalid_argument
        );
    }

    TEST_F( MLPCudaTests, Error_InvalidConfiguration_ZeroHiddenSize )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        MLPConfig invalid_config( 768, 0 );

        auto cuda_exec = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        EXPECT_THROW(
            (MLP<DeviceType::Cuda, TensorDataType::FP32>( cuda_exec, invalid_config )),
            std::invalid_argument
        );
    }

    TEST_F( MLPCudaTests, Error_NullExecutionContext )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

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
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

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
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_NO_THROW( data.mlp_module->synchronize() );
    }

    TEST_F( MLPCudaTests, SetTrainingMode )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_FALSE( data.mlp_module->isTraining() );

        data.mlp_module->setTraining( true );
        EXPECT_TRUE( data.mlp_module->isTraining() );

        data.mlp_module->setTraining( false );
        EXPECT_FALSE( data.mlp_module->isTraining() );
    }

    TEST_F( MLPCudaTests, MultipleForwardCalls )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        data.mlp_module->build( data.input_shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", data.input_shape );
        CudaTensor<TensorDataType::FP32> device_input( data.exec_context->getDevice(), data.input_shape );
        CudaTensor<TensorDataType::FP32> device_output( data.exec_context->getDevice(), data.input_shape );

        for (int iter = 0; iter < 10; ++iter)
        {
            random( host_input, -1.0f, 1.0f );
            copy( host_input, device_input );

            EXPECT_NO_THROW( data.mlp_module->forward( device_input, device_output ) );
        }
    }

    TEST_F( MLPCudaTests, CpuCuda_OutputEquivalence )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

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

        for (size_t i = 0; i < host_input.size(); ++i)
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

        for (size_t i = 0; i < cpu_output.size(); ++i)
        {
            float diff = std::abs( static_cast<float>( cpu_output.data()[i] ) - static_cast<float>( cuda_output_host.data()[i] ) );
            if (diff > epsilon)
            {
                all_equal = false;
                break;
            }
        }

        EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
    }
}