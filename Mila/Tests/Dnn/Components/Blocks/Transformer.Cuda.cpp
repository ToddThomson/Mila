#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <type_traits>

import Mila;

namespace Modules::Blocks::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Helper alias to map device and precision to the correct Tensor instantiation
    template<DeviceType TDevice, TensorDataType TPrecision>
    using TestTensor = Tensor<TPrecision, std::conditional_t<TDevice == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>>;

    template<DeviceType TDevice, TensorDataType TPrecision>
    struct TransformerTestData
    {
        shape_t input_shape;
        size_t num_heads;
        std::shared_ptr<Transformer<TDevice, TPrecision>> transformer_module;
        bool is_training{ false };

        // Create using an ExecutionContext
        static TransformerTestData Create(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t embedding_dim,
            size_t num_heads,
            bool is_training = false )
        {
            TransformerTestData data;
            data.input_shape = { static_cast<int64_t>(batch_size),
                                 static_cast<int64_t>(sequence_length),
                                 static_cast<int64_t>(embedding_dim) };
            data.num_heads = num_heads;
            data.is_training = is_training;

            auto exec_ctx = std::make_shared<ExecutionContext<TDevice>>( TDevice == DeviceType::Cuda ? 0 : -1 );

            // TransformerConfig now constructed with embedding_dim and num_heads
            TransformerConfig cfg( static_cast<dim_t>(embedding_dim),
                static_cast<dim_t>(num_heads) );
            cfg.withName( name );

            data.transformer_module = std::make_shared<Transformer<TDevice, TPrecision>>( exec_ctx, cfg );

            // Set training mode if requested
            data.transformer_module->setTraining( is_training );

            return data;
        }

        // Create with provided execution context
        static TransformerTestData CreateWithContext(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t embedding_dim,
            size_t num_heads,
            std::shared_ptr<ExecutionContext<TDevice>> context,
            bool is_training = false )
        {
            TransformerTestData data;
            data.input_shape = { static_cast<int64_t>(batch_size),
                                 static_cast<int64_t>(sequence_length),
                                 static_cast<int64_t>(embedding_dim) };
            data.num_heads = num_heads;
            data.is_training = is_training;

            TransformerConfig cfg( static_cast<dim_t>(embedding_dim),
                static_cast<dim_t>(num_heads) );
            cfg.withName( name );

            data.transformer_module = std::make_shared<Transformer<TDevice, TPrecision>>( context, cfg );
            data.transformer_module->setTraining( is_training );

            return data;
        }
    };

    class TransformerCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // CUDA-specific parameters
            cuda_batch_size_ = 4;
            cuda_sequence_length_ = 32;

            // Common
            embedding_dim_ = 256;
            num_heads_ = 8;
        }

        TransformerTestData<DeviceType::Cuda, TensorDataType::FP32>& CudaFloatData()
        {
            if (!cuda_float_data_.transformer_module)
            {
                cuda_float_data_ = TransformerTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                    "cuda_transformer_float", cuda_batch_size_, cuda_sequence_length_, embedding_dim_, num_heads_ );
            }
            return cuda_float_data_;
        }

        TransformerTestData<DeviceType::Cuda, TensorDataType::FP32>& TrainingCudaFloatData()
        {
            if (!training_cuda_float_data_.transformer_module)
            {
                training_cuda_float_data_ = TransformerTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                    "cuda_transformer_float_training", cuda_batch_size_, cuda_sequence_length_, embedding_dim_, num_heads_, true );
            }
            return training_cuda_float_data_;
        }

        TransformerTestData<DeviceType::Cuda, TensorDataType::FP16>& CudaHalfData()
        {
            if (!cuda_half_data_.transformer_module)
            {
                cuda_half_data_ = TransformerTestData<DeviceType::Cuda, TensorDataType::FP16>::Create(
                    "cuda_transformer_half", cuda_batch_size_, cuda_sequence_length_, embedding_dim_, num_heads_ );
            }
            return cuda_half_data_;
        }

        TransformerTestData<DeviceType::Cuda, TensorDataType::FP16>& TrainingCudaHalfData()
        {
            if (!training_cuda_half_data_.transformer_module)
            {
                training_cuda_half_data_ = TransformerTestData<DeviceType::Cuda, TensorDataType::FP16>::Create(
                    "cuda_transformer_half_training", cuda_batch_size_, cuda_sequence_length_, embedding_dim_, num_heads_, true );
            }
            return training_cuda_half_data_;
        }

        size_t cuda_batch_size_{ 0 };
        size_t cuda_sequence_length_{ 0 };
        size_t embedding_dim_{ 0 };
        size_t num_heads_{ 0 };

        TransformerTestData<DeviceType::Cuda, TensorDataType::FP32> cuda_float_data_;
        TransformerTestData<DeviceType::Cuda, TensorDataType::FP32> training_cuda_float_data_;

        TransformerTestData<DeviceType::Cuda, TensorDataType::FP16> cuda_half_data_;
        TransformerTestData<DeviceType::Cuda, TensorDataType::FP16> training_cuda_half_data_;
    };

    // Reuse test helpers (forward, parameter count, etc.)
    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestGetName( const TransformerTestData<TDevice, TPrecision>& data, const std::string& expected_name )
    {
        EXPECT_EQ( data.transformer_module->getName(), expected_name );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestParameterCount( const TransformerTestData<TDevice, TPrecision>& data )
    {
        data.transformer_module->build( data.input_shape );
        size_t params_count = data.transformer_module->parameterCount();
        EXPECT_GT( params_count, 0 );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestForward( const TransformerTestData<TDevice, TPrecision>& data )
    {
        using TensorT = TestTensor<TDevice, TPrecision>;

        data.transformer_module->build( data.input_shape );

        // Construct tensors bound to module device
        auto device = data.transformer_module->getDevice();

        TensorT input( device, data.input_shape );
        TensorT output( device, data.input_shape );

        // Initialize device tensor directly using initializer API.
        // Use magnitude ~0.9 to approximate previous 0..0.9 pattern.
        random( input, 0.9f );

        data.transformer_module->forward( input, output );

        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestPrint( const TransformerTestData<TDevice, TPrecision>& data, const std::string& expected_substring )
    {
        std::string output = data.transformer_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestTrainingMode( const TransformerTestData<TDevice, TPrecision>& data, bool expected_mode )
    {
        EXPECT_EQ( data.transformer_module->isTraining(), expected_mode );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestDeviceType( const TransformerTestData<TDevice, TPrecision>& data )
    {
        auto device = data.transformer_module->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestSubModules( const TransformerTestData<TDevice, TPrecision>& data )
    {
        data.transformer_module->build( data.input_shape );

        EXPECT_NE( data.transformer_module->getAttention(), nullptr );
        EXPECT_NE( data.transformer_module->getLn1(), nullptr );
        EXPECT_NE( data.transformer_module->getLn2(), nullptr );
        EXPECT_NE( data.transformer_module->getFFN(), nullptr );
    }

    // CPU-CUDA equivalence test (constructs independent CPU and CUDA transformers)
    template<TensorDataType TPrecision>
    void TestCpuCudaEquivalence()
    {
        shape_t test_shape = { 1, 2, 64 };
        size_t test_num_heads = 2;

        auto cpu_exec = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );
        auto cuda_exec = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        TransformerConfig cpu_cfg( static_cast<dim_t>(test_shape[2]), static_cast<dim_t>(test_num_heads) );
        cpu_cfg.withName( "test_cpu_transformer" );

        TransformerConfig cuda_cfg( static_cast<dim_t>(test_shape[2]), static_cast<dim_t>(test_num_heads) );
        cuda_cfg.withName( "test_cuda_transformer" );

        auto cpu_transformer = std::make_shared<Transformer<DeviceType::Cpu, TPrecision>>( cpu_exec, cpu_cfg );
        auto cuda_transformer = std::make_shared<Transformer<DeviceType::Cuda, TPrecision>>( cuda_exec, cuda_cfg );

        cpu_transformer->build( test_shape );
        cuda_transformer->build( test_shape );

        // Create host tensors with an explicit CPU device
        auto cpu_device = cpu_exec->getDevice();

        Tensor<TPrecision, HostMemoryResource> host_input( cpu_device, test_shape );
        random( host_input, 2.0f );

        Tensor<TPrecision, HostMemoryResource> cpu_output( cpu_device, test_shape );
        cpu_transformer->forward( host_input, cpu_output );

        Tensor<TPrecision, CudaDeviceMemoryResource> device_input( cuda_transformer->getDevice(), test_shape );
        copy( host_input, device_input );

        Tensor<TPrecision, CudaDeviceMemoryResource> cuda_output( cuda_transformer->getDevice(), test_shape );
        cuda_transformer->forward( device_input, cuda_output );

        cuda_transformer->synchronize();

        Tensor<TPrecision, HostMemoryResource> cuda_output_host( cpu_device, test_shape );
        // bring device->host via tensor transfer API
        copy( cuda_output, cuda_output_host );

        const float epsilon = 1e-2f;
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

    TEST_F( TransformerCudaTests, Cuda_Float_TestName )
    {
        TestGetName<DeviceType::Cuda, TensorDataType::FP32>( CudaFloatData(), "cuda_transformer_float" );
    }

    TEST_F( TransformerCudaTests, Cuda_Float_ParameterCount )
    {
        TestParameterCount<DeviceType::Cuda, TensorDataType::FP32>( CudaFloatData() );
    }

    TEST_F( TransformerCudaTests, Cuda_Float_TestForward )
    {
        TestForward<DeviceType::Cuda, TensorDataType::FP32>( CudaFloatData() );
    }

    TEST_F( TransformerCudaTests, Cuda_Float_TestPrint )
    {
        TestPrint<DeviceType::Cuda, TensorDataType::FP32>( CudaFloatData(), "Transformer: cuda_transformer_float" );
    }

    TEST_F( TransformerCudaTests, Cuda_Float_TrainingMode )
    {
        TestTrainingMode<DeviceType::Cuda, TensorDataType::FP32>( CudaFloatData(), false );
    }

    TEST_F( TransformerCudaTests, Cuda_Float_DeviceType )
    {
        TestDeviceType<DeviceType::Cuda, TensorDataType::FP32>( CudaFloatData() );
    }

    TEST_F( TransformerCudaTests, Cuda_Float_SubModules )
    {
        TestSubModules<DeviceType::Cuda, TensorDataType::FP32>( CudaFloatData() );
    }

    TEST_F( TransformerCudaTests, Cuda_Training_Float_TrainingMode )
    {
        TestTrainingMode<DeviceType::Cuda, TensorDataType::FP32>( TrainingCudaFloatData(), true );
    }

    TEST_F( TransformerCudaTests, Cuda_Training_Float_TestForward )
    {
        TestForward<DeviceType::Cuda, TensorDataType::FP32>( TrainingCudaFloatData() );
    }

    TEST_F( TransformerCudaTests, Cuda_Half_TestName )
    {
        TestGetName<DeviceType::Cuda, TensorDataType::FP16>( CudaHalfData(), "cuda_transformer_half" );
    }

    TEST_F( TransformerCudaTests, Cuda_Half_ParameterCount )
    {
        TestParameterCount<DeviceType::Cuda, TensorDataType::FP16>( CudaHalfData() );
    }

    TEST_F( TransformerCudaTests, Cuda_Half_TestForward )
    {
        TestForward<DeviceType::Cuda, TensorDataType::FP16>( CudaHalfData() );
    }

    TEST_F( TransformerCudaTests, Cuda_Half_TestPrint )
    {
        TestPrint<DeviceType::Cuda, TensorDataType::FP16>( CudaHalfData(), "Transformer: cuda_transformer_half" );
    }

    TEST_F( TransformerCudaTests, Cuda_Half_TrainingMode )
    {
        TestTrainingMode<DeviceType::Cuda, TensorDataType::FP16>( CudaHalfData(), false );
    }

    //TEST_F( TransformerCudaTests, Cuda_Float_EdgeCases )
    //{
    //    // Reuse same edge-case helper but for CUDA device and FP32
    //    TestEdgeCases<TensorDataType::FP32, DeviceType::Cuda>();
    //}

    TEST_F( TransformerCudaTests, CpuCuda_Forward_Output_Equivalence )
    {
        TestCpuCudaEquivalence<TensorDataType::FP32>();
    }
}