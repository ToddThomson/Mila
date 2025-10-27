#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <iostream>

import Mila;

namespace Modules::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Memory resource selector based on device type
    template <DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;

    template <DeviceType TDevice, TensorDataType TPrecision> struct ResidualTestData
    {
        std::vector<size_t> shape;
        ResidualConfig config;
        std::shared_ptr<ExecutionContext<TDevice>> exec_context;
        std::shared_ptr<Residual<TDevice, TPrecision>> residual_module;
        bool is_training;

        // Create with automatic execution context (CPU default / CUDA device 0)
        static ResidualTestData Create( const std::string& name, size_t batch_size, size_t sequence_length, size_t channels,
            bool is_training = false )
        {
            ResidualTestData data;
            data.shape = { batch_size, sequence_length, channels };
            data.is_training = is_training;

            data.config = ResidualConfig();
            data.config.withName( name );

            if constexpr (TDevice == DeviceType::Cuda)
            {
                data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
            }
            else
            {
                data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            }

            data.residual_module = std::make_shared<Residual<TDevice, TPrecision>>( data.exec_context, data.config );
            data.residual_module->setTraining( is_training );

            return data;
        }

        // Create with provided execution context
        static ResidualTestData CreateWithContext( const std::string& name, size_t batch_size, size_t sequence_length,
            size_t channels, std::shared_ptr<ExecutionContext<TDevice>> context,
            bool is_training = false )
        {
            ResidualTestData data;
            data.shape = { batch_size, sequence_length, channels };
            data.is_training = is_training;

            data.config = ResidualConfig();
            data.config.withName( name );

            data.exec_context = context;

            data.residual_module = std::make_shared<Residual<TDevice, TPrecision>>( data.exec_context, data.config );
            data.residual_module->setTraining( is_training );

            return data;
        }
    };

    class ResidualTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            batch_size_ = 128;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;
        }

        ResidualTestData<DeviceType::Cpu, TensorDataType::FP32>& CpuFp32Data()
        {
            if (!cpu_fp32_.residual_module)
            {
                cpu_fp32_ = ResidualTestData<DeviceType::Cpu, TensorDataType::FP32>::Create(
                    "cpu_residual_fp32", cpu_batch_size_, sequence_length_, channels_ );
            }
            return cpu_fp32_;
        }

        ResidualTestData<DeviceType::Cuda, TensorDataType::FP32>& CudaFp32Data()
        {
            if (!cuda_fp32_.residual_module)
            {
                cuda_fp32_ = ResidualTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                    "cuda_residual_fp32", batch_size_, sequence_length_, channels_ );
            }
            return cuda_fp32_;
        }

        ResidualTestData<DeviceType::Cpu, TensorDataType::FP32>& TrainingCpuFp32Data()
        {
            if (!training_cpu_fp32_.residual_module)
            {
                training_cpu_fp32_ = ResidualTestData<DeviceType::Cpu, TensorDataType::FP32>::Create(
                    "cpu_residual_fp32_training", cpu_batch_size_, sequence_length_, channels_, true );
            }
            return training_cpu_fp32_;
        }

        ResidualTestData<DeviceType::Cuda, TensorDataType::FP32>& TrainingCudaFp32Data()
        {
            if (!training_cuda_fp32_.residual_module)
            {
                training_cuda_fp32_ = ResidualTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                    "cuda_residual_fp32_training", batch_size_, sequence_length_, channels_, true );
            }
            return training_cuda_fp32_;
        }

        ResidualTestData<DeviceType::Cpu, TensorDataType::FP32>& ContextCpuFp32Data()
        {
            if (!context_cpu_fp32_.residual_module)
            {
                auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
                context_cpu_fp32_ = ResidualTestData<DeviceType::Cpu, TensorDataType::FP32>::CreateWithContext(
                    "cpu_context_residual_fp32", cpu_batch_size_, sequence_length_, channels_, ctx );
            }
            return context_cpu_fp32_;
        }

        // Test parameters
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };

        // Test data
        ResidualTestData<DeviceType::Cpu, TensorDataType::FP32> cpu_fp32_;
        ResidualTestData<DeviceType::Cpu, TensorDataType::FP32> context_cpu_fp32_;
        ResidualTestData<DeviceType::Cpu, TensorDataType::FP32> training_cpu_fp32_;

        ResidualTestData<DeviceType::Cuda, TensorDataType::FP32> cuda_fp32_;
        ResidualTestData<DeviceType::Cuda, TensorDataType::FP32> training_cuda_fp32_;
    };

    // --------------------
    // Helper test utilities
    // --------------------

    template <DeviceType TDevice, TensorDataType TPrecision>
    void TestGetName( const ResidualTestData<TDevice, TPrecision>& data, const std::string& expected_name )
    {
        EXPECT_EQ( data.residual_module->getName(), expected_name );
    }

    template <DeviceType TDevice, TensorDataType TPrecision>
    void TestParameterCount( const ResidualTestData<TDevice, TPrecision>& data, size_t expected_count )
    {
        EXPECT_EQ( data.residual_module->parameterCount(), expected_count );
    }

    template <DeviceType TDevice, TensorDataType TPrecision>
    void TestTrainingMode( const ResidualTestData<TDevice, TPrecision>& data, bool expected_mode )
    {
        EXPECT_EQ( data.residual_module->isTraining(), expected_mode );
    }

    template <DeviceType TDevice, TensorDataType TPrecision>
    void TestPrint( const ResidualTestData<TDevice, TPrecision>& data )
    {
        std::string output = data.residual_module->toString();
        EXPECT_NE( output.find( "Residual" ), std::string::npos );
        EXPECT_NE( output.find( "Connection:" ), std::string::npos );
        EXPECT_NE( output.find( "Parameter count:" ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
    }

    template <DeviceType TDevice, TensorDataType TPrecision>
    void TestForward( const ResidualTestData<TDevice, TPrecision>& data )
    {
        using MR = MemoryResourceType<TDevice>;
        using TensorType = Tensor<TPrecision, MR>;

        TensorType input( std::string( TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU" ), data.shape );
        TensorType output( std::string( TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU" ), data.shape );

        // Fill input with deterministic values on CPU to keep tests stable
        if constexpr (TDevice == DeviceType::Cpu)
        {
            for (size_t i = 0; i < input.size(); ++i)
            {
                input.data()[i] = static_cast<float>( i % 7 ) * 0.1f - 0.3f;
            }

            data.residual_module->forward( input, output );
            EXPECT_EQ( output.size(), input.size() );
        }
        else
        {
            using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;
            HostTensor host_input( "CPU", data.shape );

            for (size_t i = 0; i < host_input.size(); ++i)
            {
                host_input.data()[i] = static_cast<float>( i % 7 ) * 0.1f - 0.3f;
            }

            Tensor<TPrecision, CudaDeviceMemoryResource> device_input( "CUDA:0", data.shape );
            Tensor<TPrecision, CudaDeviceMemoryResource> device_output( "CUDA:0", data.shape );

            copy( host_input, device_input );

            data.residual_module->forward( device_input, device_output );

            auto host_out = toHost<TPrecision>( device_output );

            EXPECT_EQ( host_out.size(), host_input.size() );
        }
    }

    template <DeviceType TDevice, TensorDataType TPrecision>
    void TestDeviceType( const ResidualTestData<TDevice, TPrecision>& data )
    {
        ASSERT_NE( data.exec_context, nullptr );
        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // CPU-CUDA equivalence (small shape)
    TEST_F( ResidualTests, CpuCuda_Forward_Output_Equivalence )
    {
        auto cpu = CpuFp32Data();
        auto cuda = CudaFp32Data();

        std::vector<size_t> test_shape = { 2, 4, 8 };

        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( "CPU", test_shape );

        for (size_t i = 0; i < host_input.size(); ++i)
        {
            host_input.data()[i] = static_cast<float>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_output( "CPU", test_shape );
        cpu.residual_module->forward( host_input, cpu_output );

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> device_input( "CUDA:0", test_shape );
        copy( host_input, device_input );

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> device_output( "CUDA:0", test_shape );
        cuda.residual_module->forward( device_input, device_output );

        Tensor<TensorDataType::FP32, CpuMemoryResource> cuda_output_host = toHost<TensorDataType::FP32>( device_output );

        const float epsilon = 1e-4f;
        bool all_equal = true;
        for (size_t i = 0; i < cpu_output.size(); ++i)
        {
            float a = cpu_output.data()[i];
            float b = cuda_output_host.data()[i];
            float diff = std::abs( a - b );
            if (diff > epsilon)
            {
                std::cout << "Difference at index " << i << ": CPU=" << a << ", CUDA=" << b << ", diff=" << diff
                    << std::endl;
                all_equal = false;
            }
        }

        EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
    }

    // Edge case tests - minimal and larger sizes
    TEST_F( ResidualTests, Cpu_Fp32_EdgeCases )
    {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cfg = ResidualConfig();
        cfg.withName( "minimal_residual" );
        auto module = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>( ctx, cfg );

        std::vector<size_t> minimal_shape = { 1, 1, 8 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> in( "CPU", minimal_shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> out( "CPU", minimal_shape );

        EXPECT_NO_THROW( module->forward( in, out ) );
        EXPECT_EQ( out.size(), 8 );

        cfg = ResidualConfig();
        cfg.withName( "large_residual" );
        std::vector<size_t> large_shape = { 2, 2, 1024 };
        auto module2 = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>( ctx, cfg );

        Tensor<TensorDataType::FP32, CpuMemoryResource> in2( "CPU", large_shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> out2( "CPU", large_shape );

        EXPECT_NO_THROW( module2->forward( in2, out2 ) );
        EXPECT_EQ( out2.size(), 4096 );
    }

    TEST_F( ResidualTests, Cuda_Fp32_EdgeCases )
    {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto cfg = ResidualConfig();
        cfg.withName( "minimal_residual_cuda" );
        auto module = std::make_shared<Residual<DeviceType::Cuda, TensorDataType::FP32>>( ctx, cfg );

        std::vector<size_t> minimal_shape = { 1, 1, 8 };
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> in( "CUDA:0", minimal_shape );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out( "CUDA:0", minimal_shape );

        EXPECT_NO_THROW( module->forward( in, out ) );
        EXPECT_EQ( out.size(), 8 );

        std::vector<size_t> large_shape = { 2, 2, 1024 };
        cfg = ResidualConfig();
        cfg.withName( "large_residual_cuda" );
        auto module2 = std::make_shared<Residual<DeviceType::Cuda, TensorDataType::FP32>>( ctx, cfg );

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> in2( "CUDA:0", large_shape );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out2( "CUDA:0", large_shape );

        EXPECT_NO_THROW( module2->forward( in2, out2 ) );
        EXPECT_EQ( out2.size(), 4096 );
    }

    // CPU Tests FP32
    TEST_F( ResidualTests, Cpu_Fp32_TestName )
    {
        TestGetName<DeviceType::Cpu, TensorDataType::FP32>( CpuFp32Data(), "cpu_residual_fp32" );
    }

    TEST_F( ResidualTests, Cpu_Fp32_ParameterCount )
    {
        TestParameterCount<DeviceType::Cpu, TensorDataType::FP32>( CpuFp32Data(), 0 );
    }

    TEST_F( ResidualTests, Cpu_Fp32_TestForward )
    {
        TestForward<DeviceType::Cpu, TensorDataType::FP32>( CpuFp32Data() );
    }

    TEST_F( ResidualTests, Cpu_Fp32_TestPrint )
    {
        TestPrint<DeviceType::Cpu, TensorDataType::FP32>( CpuFp32Data() );
    }

    TEST_F( ResidualTests, Cpu_Fp32_TrainingMode )
    {
        TestTrainingMode<DeviceType::Cpu, TensorDataType::FP32>( CpuFp32Data(), false );
    }

    TEST_F( ResidualTests, Cpu_Fp32_DeviceType )
    {
        TestDeviceType<DeviceType::Cpu, TensorDataType::FP32>( CpuFp32Data() );
    }

    // CPU Training Mode
    TEST_F( ResidualTests, Cpu_Training_Fp32_TrainingMode )
    {
        TestTrainingMode<DeviceType::Cpu, TensorDataType::FP32>( TrainingCpuFp32Data(), true );
    }

    // CUDA Tests FP32
    TEST_F( ResidualTests, Cuda_Fp32_TestName )
    {
        TestGetName<DeviceType::Cuda, TensorDataType::FP32>( CudaFp32Data(), "cuda_residual_fp32" );
    }

    TEST_F( ResidualTests, Cuda_Fp32_ParameterCount )
    {
        TestParameterCount<DeviceType::Cuda, TensorDataType::FP32>( CudaFp32Data(), 0 );
    }

    TEST_F( ResidualTests, Cuda_Fp32_TestForward )
    {
        TestForward<DeviceType::Cuda, TensorDataType::FP32>( CudaFp32Data() );
    }

    TEST_F( ResidualTests, Cuda_Fp32_TestPrint )
    {
        TestPrint<DeviceType::Cuda, TensorDataType::FP32>( CudaFp32Data() );
    }

    TEST_F( ResidualTests, Cuda_Fp32_TrainingMode )
    {
        TestTrainingMode<DeviceType::Cuda, TensorDataType::FP32>( CudaFp32Data(), false );
    }

    TEST_F( ResidualTests, Cuda_Fp32_DeviceType )
    {
        TestDeviceType<DeviceType::Cuda, TensorDataType::FP32>( CudaFp32Data() );
    }

    // CUDA Training Mode
    TEST_F( ResidualTests, Cuda_Training_Fp32_TrainingMode )
    {
        TestTrainingMode<DeviceType::Cuda, TensorDataType::FP32>( TrainingCudaFp32Data(), true );
    }

    // Context Construction Tests
    TEST_F( ResidualTests, Context_Cpu_Fp32_DeviceType )
    {
        TestDeviceType<DeviceType::Cpu, TensorDataType::FP32>( ContextCpuFp32Data() );
    }

    TEST_F( ResidualTests, Context_Cpu_Fp32_Forward )
    {
        TestForward<DeviceType::Cpu, TensorDataType::FP32>( ContextCpuFp32Data() );
    }
} // namespace Modules::Layers::Tests