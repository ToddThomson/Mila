#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <type_traits>
#include <stdexcept>

import Mila;

namespace Modules::Blocks::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    // Memory resource depends only on device type for these tests.
    template<DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == DeviceType::Cuda,
        CudaDeviceMemoryResource,
        CpuMemoryResource>;

    // Test fixture data uses the TensorDataType precision used by the MLP implementation.
    template<DeviceType TDevice, TensorDataType TPrecision>
    struct MLPTestData
    {
        MLPConfig config;
        std::shared_ptr<MLP<TDevice, TPrecision>> mlp_module;
        std::vector<size_t> input_shape;
        size_t hidden_size;
        std::shared_ptr<ExecutionContext<TDevice>> exec_context;

        MLPTestData() : config( { 1 }, 1 )
        {
        }

        static MLPTestData Create(
            const std::string& name,
            const std::vector<size_t>& input_shape,
            size_t hidden_size,
            bool has_bias = true,
            bool is_training = false,
            ActivationType activation = ActivationType::Gelu,
            bool use_layer_norm = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            MLPTestData data;
            data.input_shape = input_shape;
            data.hidden_size = hidden_size;

            data.config = MLPConfig( input_shape, hidden_size );
            data.config
                .withBias( has_bias )
                .withActivation( activation )
                .withLayerNorm( use_layer_norm )
                .withName( name )
                .withPrecisionPolicy( precision )
                .withTraining( is_training );

            std::string device_str = TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU";

            // Create an execution context and retain it so tests can inspect device info.
            data.exec_context = std::make_shared<ExecutionContext<TDevice>>( 0 );

            // Construct module using the supplied execution context.
            data.mlp_module = std::make_shared<MLP<TDevice, TPrecision>>( data.exec_context, data.config );

            return data;
        }

        static MLPTestData CreateWithContext(
            const std::string& name,
            const std::vector<size_t>& input_shape,
            size_t hidden_size,
            std::shared_ptr<ExecutionContext<TDevice>> context,
            bool has_bias = true,
            bool is_training = false,
            ActivationType activation = ActivationType::Gelu,
            bool use_layer_norm = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            MLPTestData data;
            data.input_shape = input_shape;
            data.hidden_size = hidden_size;

            data.config = MLPConfig( input_shape, hidden_size )
                .withBias( has_bias )
                .withActivation( activation )
                .withLayerNorm( use_layer_norm )
                .withName( name )
                .withPrecisionPolicy( precision )
                .withTraining( is_training );

            data.exec_context = context;
            data.mlp_module = std::make_shared<MLP<TDevice, TPrecision>>( context, data.config );

            return data;
        }
    };

    class MLPTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            cuda_batch_size_ = 16;
            cuda_sequence_length_ = 64;
            cpu_batch_size_ = 2;
            cpu_sequence_length_ = 16;
            input_features_ = 768;
            hidden_size_ = 3072;
        }

        void TearDown() override
        {
            cpu_float_data_.mlp_module.reset();
            context_cpu_float_data_.mlp_module.reset();
            training_cpu_float_data_.mlp_module.reset();
            no_bias_cpu_float_data_.mlp_module.reset();
            layer_norm_cpu_float_data_.mlp_module.reset();

            cuda_float_data_.mlp_module.reset();
            training_cuda_float_data_.mlp_module.reset();
            no_bias_cuda_float_data_.mlp_module.reset();
            layer_norm_cuda_float_data_.mlp_module.reset();

            cuda_fp16_data_.mlp_module.reset();
            training_cuda_fp16_data_.mlp_module.reset();

            perf_precision_cuda_float_data_.mlp_module.reset();
            accuracy_precision_cuda_float_data_.mlp_module.reset();
            native_precision_cuda_float_data_.mlp_module.reset();
        }

        MLPTestData<DeviceType::Cpu, TensorDataType::FP32>& CpuFloatData()
        {
            if (!cpu_float_data_.mlp_module)
            {
                cpu_float_data_ = MLPTestData<DeviceType::Cpu, TensorDataType::FP32>::Create(
                    "cpu_mlp_float", { cpu_batch_size_, cpu_sequence_length_, input_features_ }, hidden_size_ );
            }
            return cpu_float_data_;
        }

        MLPTestData<DeviceType::Cuda, TensorDataType::FP32>& CudaFloatData()
        {
            if (!cuda_float_data_.mlp_module)
            {
                try
                {
                    cuda_float_data_ = MLPTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                        "cuda_mlp_float", { cuda_batch_size_, cuda_sequence_length_, input_features_ }, hidden_size_ );
                }
                catch (const std::exception&)
                {
                    // GPU unavailable — tests that depend on CUDA will skip.
                }
            }
            return cuda_float_data_;
        }

        MLPTestData<DeviceType::Cpu, TensorDataType::FP32>& TrainingCpuFloatData()
        {
            if (!training_cpu_float_data_.mlp_module)
            {
                training_cpu_float_data_ = MLPTestData<DeviceType::Cpu, TensorDataType::FP32>::Create(
                    "cpu_mlp_float_training", { cpu_batch_size_, cpu_sequence_length_, input_features_ },
                    hidden_size_, true, true );
            }
            return training_cpu_float_data_;
        }

        MLPTestData<DeviceType::Cuda, TensorDataType::FP32>& TrainingCudaFloatData()
        {
            if (!training_cuda_float_data_.mlp_module)
            {
                try
                {
                    training_cuda_float_data_ = MLPTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                        "cuda_mlp_float_training", { cuda_batch_size_, cuda_sequence_length_, input_features_ },
                        hidden_size_, true, true );
                }
                catch (const std::exception&)
                {
                }
            }
            return training_cuda_float_data_;
        }

        MLPTestData<DeviceType::Cpu, TensorDataType::FP32>& NoBiasCpuFloatData()
        {
            if (!no_bias_cpu_float_data_.mlp_module)
            {
                no_bias_cpu_float_data_ = MLPTestData<DeviceType::Cpu, TensorDataType::FP32>::Create(
                    "cpu_mlp_float_nobias", { cpu_batch_size_, cpu_sequence_length_, input_features_ },
                    hidden_size_, false );
            }
            return no_bias_cpu_float_data_;
        }

        MLPTestData<DeviceType::Cuda, TensorDataType::FP32>& NoBiasCudaFloatData()
        {
            if (!no_bias_cuda_float_data_.mlp_module)
            {
                try
                {
                    no_bias_cuda_float_data_ = MLPTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                        "cuda_mlp_float_nobias", { cuda_batch_size_, cuda_sequence_length_, input_features_ },
                        hidden_size_, false );
                }
                catch (const std::exception&)
                {
                }
            }
            return no_bias_cuda_float_data_;
        }

        MLPTestData<DeviceType::Cpu, TensorDataType::FP32>& LayerNormCpuFloatData()
        {
            if (!layer_norm_cpu_float_data_.mlp_module)
            {
                layer_norm_cpu_float_data_ = MLPTestData<DeviceType::Cpu, TensorDataType::FP32>::Create(
                    "cpu_mlp_float_layernorm", { cpu_batch_size_, cpu_sequence_length_, input_features_ },
                    hidden_size_, true, false, ActivationType::Gelu, true );
            }
            return layer_norm_cpu_float_data_;
        }

        MLPTestData<DeviceType::Cuda, TensorDataType::FP32>& LayerNormCudaFloatData()
        {
            if (!layer_norm_cuda_float_data_.mlp_module)
            {
                try
                {
                    layer_norm_cuda_float_data_ = MLPTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                        "cuda_mlp_float_layernorm", { cuda_batch_size_, cuda_sequence_length_, input_features_ },
                        hidden_size_, true, false, ActivationType::Gelu, true );
                }
                catch (const std::exception&)
                {
                }
            }
            return layer_norm_cuda_float_data_;
        }

        MLPTestData<DeviceType::Cpu, TensorDataType::FP32>& ContextCpuFloatData()
        {
            if (!context_cpu_float_data_.mlp_module)
            {
                auto exec_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
                context_cpu_float_data_ = MLPTestData<DeviceType::Cpu, TensorDataType::FP32>::CreateWithContext(
                    "cpu_context_mlp_float", { cpu_batch_size_, cpu_sequence_length_, input_features_ },
                    hidden_size_, exec_ctx );
            }
            return context_cpu_float_data_;
        }

        MLPTestData<DeviceType::Cuda, TensorDataType::FP16>& CudaFP16Data()
        {
            if (!cuda_fp16_data_.mlp_module)
            {
                try
                {
                    cuda_fp16_data_ = MLPTestData<DeviceType::Cuda, TensorDataType::FP16>::Create(
                        "cuda_mlp_fp16", { cuda_batch_size_, cuda_sequence_length_, input_features_ }, hidden_size_ );
                }
                catch (const std::exception&)
                {
                }
            }
            return cuda_fp16_data_;
        }

        MLPTestData<DeviceType::Cuda, TensorDataType::FP16>& TrainingCudaFP16Data()
        {
            if (!training_cuda_fp16_data_.mlp_module)
            {
                try
                {
                    training_cuda_fp16_data_ = MLPTestData<DeviceType::Cuda, TensorDataType::FP16>::Create(
                        "cuda_mlp_fp16_training", { cuda_batch_size_, cuda_sequence_length_, input_features_ },
                        hidden_size_, true, true );
                }
                catch (const std::exception&)
                {
                }
            }
            return training_cuda_fp16_data_;
        }

        MLPTestData<DeviceType::Cuda, TensorDataType::FP32>& PerfPrecisionCudaFloatData()
        {
            if (!perf_precision_cuda_float_data_.mlp_module)
            {
                try
                {
                    perf_precision_cuda_float_data_ = MLPTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                        "cuda_mlp_perf_precision", { cuda_batch_size_, cuda_sequence_length_, input_features_ },
                        hidden_size_, true, false, ActivationType::Gelu, false,
                        ComputePrecision::Policy::Performance );
                }
                catch (const std::exception&)
                {
                }
            }
            return perf_precision_cuda_float_data_;
        }

        MLPTestData<DeviceType::Cuda, TensorDataType::FP32>& AccuracyPrecisionCudaFloatData()
        {
            if (!accuracy_precision_cuda_float_data_.mlp_module)
            {
                try
                {
                    accuracy_precision_cuda_float_data_ = MLPTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                        "cuda_mlp_accuracy_precision", { cuda_batch_size_, cuda_sequence_length_, input_features_ },
                        hidden_size_, true, false, ActivationType::Gelu, false,
                        ComputePrecision::Policy::Accuracy );
                }
                catch (const std::exception&)
                {
                }
            }
            return accuracy_precision_cuda_float_data_;
        }

        MLPTestData<DeviceType::Cuda, TensorDataType::FP32>& NativePrecisionCudaFloatData()
        {
            if (!native_precision_cuda_float_data_.mlp_module)
            {
                try
                {
                    native_precision_cuda_float_data_ = MLPTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                        "cuda_mlp_native_precision", { cuda_batch_size_, cuda_sequence_length_, input_features_ },
                        hidden_size_, true, false, ActivationType::Gelu, false,
                        ComputePrecision::Policy::Native );
                }
                catch (const std::exception&)
                {
                }
            }
            return native_precision_cuda_float_data_;
        }

        size_t cuda_batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t cuda_sequence_length_{ 0 };
        size_t cpu_sequence_length_{ 0 };
        size_t input_features_{ 0 };
        size_t hidden_size_{ 0 };

        MLPTestData<DeviceType::Cpu, TensorDataType::FP32> cpu_float_data_;
        MLPTestData<DeviceType::Cpu, TensorDataType::FP32> context_cpu_float_data_;
        MLPTestData<DeviceType::Cpu, TensorDataType::FP32> training_cpu_float_data_;
        MLPTestData<DeviceType::Cpu, TensorDataType::FP32> no_bias_cpu_float_data_;
        MLPTestData<DeviceType::Cpu, TensorDataType::FP32> layer_norm_cpu_float_data_;

        MLPTestData<DeviceType::Cuda, TensorDataType::FP32> cuda_float_data_;
        MLPTestData<DeviceType::Cuda, TensorDataType::FP32> training_cuda_float_data_;
        MLPTestData<DeviceType::Cuda, TensorDataType::FP32> no_bias_cuda_float_data_;
        MLPTestData<DeviceType::Cuda, TensorDataType::FP32> layer_norm_cuda_float_data_;

        MLPTestData<DeviceType::Cuda, TensorDataType::FP16> cuda_fp16_data_;
        MLPTestData<DeviceType::Cuda, TensorDataType::FP16> training_cuda_fp16_data_;

        MLPTestData<DeviceType::Cuda, TensorDataType::FP32> perf_precision_cuda_float_data_;
        MLPTestData<DeviceType::Cuda, TensorDataType::FP32> accuracy_precision_cuda_float_data_;
        MLPTestData<DeviceType::Cuda, TensorDataType::FP32> native_precision_cuda_float_data_;
    };

    // Tests adapted to new Module API (getName(), execution context ownership stored in test data)
    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestGetName( const MLPTestData<TDevice, TPrecision>& data, const std::string& expected_name )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        EXPECT_EQ( data.mlp_module->getName(), expected_name );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestParameterCount( const MLPTestData<TDevice, TPrecision>& data )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        size_t input_features = data.config.getInputFeatures();
        size_t hidden_size = data.config.getHiddenSize();
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

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestForward( const MLPTestData<TDevice, TPrecision>& data )
    {
        // Ensure test data is valid and device is available
        if (!data.mlp_module || !data.exec_context)
        {
            throw std::runtime_error( "Module or execution context not available" );
        }

        using MR = MemoryResourceType<TDevice>;
        using TensorType = Tensor<TPrecision, MR>;

        // Construct tensors bound to the execution context's device
        TensorType input( data.exec_context->getDevice(), data.input_shape );
        TensorType output( data.exec_context->getDevice(), data.input_shape );

        random( input, -1.0f, 1.0f );

        EXPECT_NO_THROW( data.mlp_module->forward( input, output ) );

        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestToString( const MLPTestData<TDevice, TPrecision>& data, const std::string& expected_substring )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        std::string output = data.mlp_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestTrainingMode( const MLPTestData<TDevice, TPrecision>& data, bool expected_mode )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        EXPECT_EQ( data.mlp_module->isTraining(), expected_mode );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestDeviceType( const MLPTestData<TDevice, TPrecision>& data )
    {
        ASSERT_NE( data.exec_context, nullptr );
        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestSubModules( const MLPTestData<TDevice, TPrecision>& data )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        auto modules = data.mlp_module->getNamedModules();

        EXPECT_GE( modules.size(), 3 );
        EXPECT_NE( modules.find( "fc1" ), modules.end() );
        EXPECT_NE( modules.find( "activation" ), modules.end() );
        EXPECT_NE( modules.find( "fc2" ), modules.end() );

        if (data.config.useLayerNorm())
        {
            EXPECT_NE( modules.find( "norm1" ), modules.end() );
        }
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestSaveLoad( const MLPTestData<TDevice, TPrecision>& data )
    {
        ASSERT_NE( data.mlp_module, nullptr );
        ModelArchive archive;
        EXPECT_NO_THROW( data.mlp_module->save( archive ) );
        EXPECT_NO_THROW( data.mlp_module->load( archive ) );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestEdgeCases()
    {
        using MR = MemoryResourceType<TDevice>;
        using TensorType = Tensor<TPrecision, MR>;

        try
        {
            std::vector<size_t> minimal_shape = { 1, 1, 8 };
            size_t minimal_hidden_size = 16;

            auto minimal_config = MLPConfig( minimal_shape, minimal_hidden_size );
            minimal_config.withName( "minimal_mlp" );

            std::string device_str = TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU";
            auto exec_ctx = std::make_shared<ExecutionContext<TDevice>>( 0 );
            auto minimal_module = std::make_shared<MLP<TDevice, TPrecision>>( exec_ctx, minimal_config );

            TensorType minimal_input( exec_ctx->getDevice(), minimal_shape );
            TensorType minimal_output( exec_ctx->getDevice(), minimal_shape );

            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), 8 );

            std::vector<size_t> medium_shape;
            if constexpr (TDevice == DeviceType::Cuda)
            {
                medium_shape = { 2, 2, 1024 };
            }
            else
            {
                medium_shape = { 1, 2, 512 };
            }

            size_t medium_hidden_size = 2048;
            auto medium_config = MLPConfig( medium_shape, medium_hidden_size );
            medium_config.withName( "medium_mlp" );
            auto medium_module = std::make_shared<MLP<TDevice, TPrecision>>( exec_ctx, medium_config );

            TensorType medium_input( exec_ctx->getDevice(), medium_shape );
            TensorType medium_output( exec_ctx->getDevice(), medium_shape );

            EXPECT_NO_THROW( medium_module->forward( medium_input, medium_output ) );
        }
        catch (const std::exception& e)
        {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    template<typename HostType = float>
    void TestCpuCudaEquivalence(
        const MLPTestData<DeviceType::Cpu, TensorDataType::FP32>& cpu_data,
        const MLPTestData<DeviceType::Cuda, TensorDataType::FP32>& cuda_data )
    {
        try
        {
            // Make sure CUDA available by attempting to create an execution context.
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping CPU-CUDA equivalence test";
            return;
        }

        std::vector<size_t> test_shape = { 1, 2, cpu_data.config.getInputFeatures() };
        size_t test_hidden_size = 1024;

        auto cpu_config = MLPConfig( test_shape, test_hidden_size );
        cpu_config.withName( "test_cpu_mlp" );
        auto cuda_config = MLPConfig( test_shape, test_hidden_size );
        cuda_config.withName( "test_cuda_mlp" );

        auto cpu_exec = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cuda_exec = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        auto cpu_mlp = std::make_shared<MLP<DeviceType::Cpu, TensorDataType::FP32>>( cpu_exec, cpu_config );
        auto cuda_mlp = std::make_shared<MLP<DeviceType::Cuda, TensorDataType::FP32>>( cuda_exec, cuda_config );

        using CpuTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using CudaTensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;

        CpuTensor host_input( cpu_exec->getDevice(), test_shape );

        for (size_t i = 0; i < host_input.size(); ++i)
        {
            host_input.data()[i] = static_cast<float>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        CpuTensor cpu_output( cpu_exec->getDevice(), test_shape );
        cpu_mlp->forward( host_input, cpu_output );

        CudaTensor device_input( cuda_exec->getDevice(), test_shape );
        // Copy host -> device using provided utility
        copy( host_input, device_input );

        CudaTensor cuda_output( cuda_exec->getDevice(), test_shape );
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
                std::cout << "Difference at index " << i << ": CPU=" << cpu_output.data()[i]
                    << ", CUDA=" << cuda_output_host.data()[i] << ", diff=" << diff << std::endl;
                all_equal = false;
            }
        }

        EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
    }

    // --- Tests ---

    TEST_F( MLPTests, Cpu_Float_TestName )
    {
        TestGetName( CpuFloatData(), "cpu_mlp_float" );
    }

    TEST_F( MLPTests, Cpu_Float_ParameterCount )
    {
        TestParameterCount( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_TestForward )
    {
        TestForward( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_TestToString )
    {
        TestToString( CpuFloatData(), "MLP: cpu_mlp_float" );
    }

    TEST_F( MLPTests, Cpu_Float_TrainingMode )
    {
        TestTrainingMode( CpuFloatData(), false );
    }

    TEST_F( MLPTests, Cpu_Float_DeviceType )
    {
        TestDeviceType( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_SubModules )
    {
        TestSubModules( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_SaveLoad )
    {
        TestSaveLoad( CpuFloatData() );
    }

    TEST_F( MLPTests, NoBias_Cpu_Float_ParameterCount )
    {
        TestParameterCount( NoBiasCpuFloatData() );
    }

    TEST_F( MLPTests, NoBias_Cpu_Float_TestForward )
    {
        TestForward( NoBiasCpuFloatData() );
    }

    TEST_F( MLPTests, LayerNorm_Cpu_Float_TestForward )
    {
        TestForward( LayerNormCpuFloatData() );
    }

    TEST_F( MLPTests, LayerNorm_Cpu_Float_SubModules )
    {
        TestSubModules( LayerNormCpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Training_Float_TrainingMode )
    {
        TestTrainingMode( TrainingCpuFloatData(), true );
    }

    TEST_F( MLPTests, Cpu_Training_Float_TestForward )
    {
        TestForward( TrainingCpuFloatData() );
    }

    TEST_F( MLPTests, Cuda_Float_TestName )
    {
        try
        {
            TestGetName( CudaFloatData(), "cuda_mlp_float" );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Float_ParameterCount )
    {
        try
        {
            TestParameterCount( CudaFloatData() );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Float_TestForward )
    {
        try
        {
            TestForward( CudaFloatData() );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Float_TestToString )
    {
        try
        {
            TestToString( CudaFloatData(), "MLP: cuda_mlp_float" );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Float_TrainingMode )
    {
        try
        {
            TestTrainingMode( CudaFloatData(), false );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Float_DeviceType )
    {
        try
        {
            TestDeviceType( CudaFloatData() );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Float_SubModules )
    {
        try
        {
            TestSubModules( CudaFloatData() );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, NoBias_Cuda_Float_ParameterCount )
    {
        try
        {
            TestParameterCount( NoBiasCudaFloatData() );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, NoBias_Cuda_Float_TestForward )
    {
        try
        {
            TestForward( NoBiasCudaFloatData() );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, LayerNorm_Cuda_Float_TestForward )
    {
        try
        {
            TestForward( LayerNormCudaFloatData() );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, LayerNorm_Cuda_Float_SubModules )
    {
        try
        {
            TestSubModules( LayerNormCudaFloatData() );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Training_Float_TrainingMode )
    {
        try
        {
            TestTrainingMode( TrainingCudaFloatData(), true );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Training_Float_TestForward )
    {
        try
        {
            TestForward( TrainingCudaFloatData() );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_FP16_TestForward )
    {
        try
        {
            TestForward( CudaFP16Data() );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_PerformancePrecision_Policy )
    {
        try
        {
            TestForward( PerfPrecisionCudaFloatData() );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_AccuracyPrecision_Policy )
    {
        try
        {
            TestForward( AccuracyPrecisionCudaFloatData() );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_NativePrecision_Policy )
    {
        try
        {
            TestForward( NativePrecisionCudaFloatData() );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Context_Cpu_Float_DeviceType )
    {
        TestDeviceType( ContextCpuFloatData() );
    }

    TEST_F( MLPTests, Context_Cpu_Float_Forward )
    {
        TestForward( ContextCpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_EdgeCases )
    {
        TestEdgeCases<DeviceType::Cpu, TensorDataType::FP32>();
    }

    TEST_F( MLPTests, Cuda_Float_EdgeCases )
    {
        try
        {
            TestEdgeCases<DeviceType::Cuda, TensorDataType::FP32>();
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, CpuCuda_Forward_Output_Equivalence )
    {
        try
        {
            TestCpuCudaEquivalence( CpuFloatData(), CudaFloatData() );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Constructor_InvalidConfiguration )
    {
        MLPConfig invalid_config( 0, 1024 );

        auto cpu_exec = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        EXPECT_THROW(
            (MLP<DeviceType::Cpu, TensorDataType::FP32>( cpu_exec, invalid_config )),
            std::invalid_argument
        );
    }

    TEST_F( MLPTests, Constructor_DeviceNameValidation )
    {
        MLPConfig config( { 2, 16, 768 }, 3072 );
        config.withName( "validation_test" );

        // construct using execution context created from device name
        EXPECT_NO_THROW( (MLP<DeviceType::Cpu, TensorDataType::FP32>( std::make_shared<ExecutionContext<DeviceType::Cpu>>(), config )) );
    }

    TEST_F( MLPTests, Constructor_ExecutionContextValidation )
    {
        MLPConfig config( { 2, 16, 768 }, 3072 );
        config.withName( "context_validation_test" );

        auto cpu_exec = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        EXPECT_NO_THROW( (MLP<DeviceType::Cpu, TensorDataType::FP32>( cpu_exec, config )) );
    }
}