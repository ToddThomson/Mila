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

        TransformerTestData<DeviceType::Cuda, TensorDataType::FP32> CudaFloatData()
        {
            return TransformerTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                "cuda_transformer_float", cuda_batch_size_, cuda_sequence_length_, embedding_dim_, num_heads_ );
        }

        TransformerTestData<DeviceType::Cuda, TensorDataType::FP32> TrainingCudaFloatData()
        {
            return TransformerTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                "cuda_transformer_float_training", cuda_batch_size_, cuda_sequence_length_, embedding_dim_, num_heads_, true );
        }

        TransformerTestData<DeviceType::Cuda, TensorDataType::FP16> CudaHalfData()
        {
            return TransformerTestData<DeviceType::Cuda, TensorDataType::FP16>::Create(
                "cuda_transformer_half", cuda_batch_size_, cuda_sequence_length_, embedding_dim_, num_heads_ );
        }

        TransformerTestData<DeviceType::Cuda, TensorDataType::FP16> TrainingCudaHalfData()
        {
            return TransformerTestData<DeviceType::Cuda, TensorDataType::FP16>::Create(
                "cuda_transformer_half_training", cuda_batch_size_, cuda_sequence_length_, embedding_dim_, num_heads_, true );
        }

        size_t cuda_batch_size_{ 0 };
        size_t cuda_sequence_length_{ 0 };
        size_t embedding_dim_{ 0 };
        size_t num_heads_{ 0 };
    };

    // NOTE: The helper functions below remain for future reuse but tests are written
    //       to be self-contained (setup, execution, assertions inline).

    // CPU-CUDA equivalence test (constructs independent CPU and CUDA transformers)
    template<TensorDataType TPrecision>
    void CpuCudaEquivalenceInline()
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

    // ---------------------------------------------------------------------
    // Self-contained tests (setup, execute, assert inline) for CUDA device
    // ---------------------------------------------------------------------

    TEST_F( TransformerCudaTests, Cuda_Float_TestName )
    {
        auto data = CudaFloatData();

        ASSERT_NE( data.transformer_module, nullptr );
        EXPECT_EQ( data.transformer_module->getName(), "cuda_transformer_float" );
    }

    TEST_F( TransformerCudaTests, Cuda_Float_ParameterCount )
    {
        auto data = CudaFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        EXPECT_NO_THROW( data.transformer_module->build( data.input_shape ) );

        size_t params_count = data.transformer_module->parameterCount();
        EXPECT_GT( params_count, 0u );
    }

    TEST_F( TransformerCudaTests, Cuda_Float_TestForward )
    {
        auto data = CudaFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        using TensorT = TestTensor<DeviceType::Cuda, TensorDataType::FP32>;

        EXPECT_NO_THROW( data.transformer_module->build( data.input_shape ) );

        auto device = data.transformer_module->getDevice();

        TensorT input( device, data.input_shape );
        TensorT output( device, data.input_shape );

        random( input, 0.9f );

        EXPECT_NO_THROW( data.transformer_module->forward( input, output ) );

        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    TEST_F( TransformerCudaTests, Cuda_Float_TestPrint )
    {
        auto data = CudaFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        std::string s = data.transformer_module->toString();
        EXPECT_NE( s.find( "Transformer: cuda_transformer_float" ), std::string::npos );
    }

    TEST_F( TransformerCudaTests, Cuda_Float_TrainingMode )
    {
        auto data = CudaFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        EXPECT_FALSE( data.transformer_module->isTraining() );
    }

    TEST_F( TransformerCudaTests, Cuda_Float_DeviceType )
    {
        auto data = CudaFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        auto device = data.transformer_module->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );
    }

    TEST_F( TransformerCudaTests, Cuda_Training_Float_TrainingMode )
    {
        auto data = TrainingCudaFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        EXPECT_TRUE( data.transformer_module->isTraining() );
    }

    TEST_F( TransformerCudaTests, Cuda_Training_Float_TestForward )
    {
        auto data = TrainingCudaFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        using TensorT = TestTensor<DeviceType::Cuda, TensorDataType::FP32>;

        EXPECT_NO_THROW( data.transformer_module->build( data.input_shape ) );

        auto device = data.transformer_module->getDevice();

        TensorT input( device, data.input_shape );
        TensorT output( device, data.input_shape );

        random( input, 0.9f );

        EXPECT_NO_THROW( data.transformer_module->forward( input, output ) );
    }

    TEST_F( TransformerCudaTests, Cuda_Half_TestName )
    {
        auto data = CudaHalfData();

        ASSERT_NE( data.transformer_module, nullptr );
        EXPECT_EQ( data.transformer_module->getName(), "cuda_transformer_half" );
    }

    TEST_F( TransformerCudaTests, Cuda_Half_ParameterCount )
    {
        auto data = CudaHalfData();

        ASSERT_NE( data.transformer_module, nullptr );

        EXPECT_NO_THROW( data.transformer_module->build( data.input_shape ) );

        size_t params_count = data.transformer_module->parameterCount();
        EXPECT_GT( params_count, 0u );
    }

    TEST_F( TransformerCudaTests, Cuda_Half_TestForward )
    {
        auto data = CudaHalfData();

        ASSERT_NE( data.transformer_module, nullptr );

        using TensorT = TestTensor<DeviceType::Cuda, TensorDataType::FP16>;

        EXPECT_NO_THROW( data.transformer_module->build( data.input_shape ) );

        auto device = data.transformer_module->getDevice();

        TensorT input( device, data.input_shape );
        TensorT output( device, data.input_shape );

        random( input, 0.9f );

        EXPECT_NO_THROW( data.transformer_module->forward( input, output ) );
    }

    TEST_F( TransformerCudaTests, Cuda_Half_TestPrint )
    {
        auto data = CudaHalfData();

        ASSERT_NE( data.transformer_module, nullptr );

        std::string s = data.transformer_module->toString();
        EXPECT_NE( s.find( "Transformer: cuda_transformer_half" ), std::string::npos );
    }

    TEST_F( TransformerCudaTests, Cuda_Half_TrainingMode )
    {
        auto data = CudaHalfData();

        ASSERT_NE( data.transformer_module, nullptr );

        EXPECT_FALSE( data.transformer_module->isTraining() );
    }

    TEST_F( TransformerCudaTests, CpuCuda_Forward_Output_Equivalence )
    {
        CpuCudaEquivalenceInline<TensorDataType::FP32>();
    }
}