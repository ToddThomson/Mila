#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <cstdint>

import Mila;

namespace Modules::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Memory resource selector based on device type
    template<DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == DeviceType::Cuda,
        CudaDeviceMemoryResource,
        CpuMemoryResource>;

    // Softmax test data aligned to new Softmax API (SoftmaxConfig + ExecutionContext)
    template<DeviceType TDevice, TensorDataType TPrecision>
    struct SoftmaxTestData
    {
        shape_t shape;
        SoftmaxConfig config;
        std::shared_ptr<ExecutionContext<TDevice>> exec_context;
        std::shared_ptr<Softmax<TDevice, TPrecision>> softmax_module;
        int64_t axis;
        bool is_training;

        // Create with automatic execution context (CPU default / CUDA device 0)
        static SoftmaxTestData Create(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t vocab_size,
            int64_t axis = -1,
            bool is_training = false )
        {
            SoftmaxTestData data;
            data.shape = { static_cast<dim_t>(batch_size), static_cast<dim_t>(sequence_length), static_cast<dim_t>(vocab_size) };
            data.axis = axis;
            data.is_training = is_training;

            data.config = SoftmaxConfig();
            data.config.withName( name ); // ConfigurationBase::withName exists in project conventions
            data.config.withAxis( axis );

            if constexpr (TDevice == DeviceType::Cuda)
            {
                data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
            }
            else
            {
                data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            }

            data.softmax_module = std::make_shared<Softmax<TDevice, TPrecision>>( data.config, data.exec_context );
            data.softmax_module->setTraining( is_training );

            return data;
        }

        // Create with provided execution context
        static SoftmaxTestData CreateWithContext(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t vocab_size,
            std::shared_ptr<ExecutionContext<TDevice>> context,
            int64_t axis = -1,
            bool is_training = false )
        {
            SoftmaxTestData data;
            data.shape = { static_cast<dim_t>(batch_size), static_cast<dim_t>(sequence_length), static_cast<dim_t>(vocab_size) };
            data.axis = axis;
            data.is_training = is_training;

            data.config = SoftmaxConfig();
            data.config.withName( name );
            data.config.withAxis( axis );

            data.exec_context = context;

            data.softmax_module = std::make_shared<Softmax<TDevice, TPrecision>>( data.config, data.exec_context );
            data.softmax_module->setTraining( is_training );

            return data;
        }
    };

    class SoftmaxTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            batch_size_ = 64;
            cpu_batch_size_ = 4;
            sequence_length_ = 128;
            vocab_size_ = 1024;
            axis_ = -1;
        }

        SoftmaxTestData<DeviceType::Cpu, TensorDataType::FP32>& CpuFp32Data()
        {
            if (!cpu_fp32_.softmax_module)
            {
                cpu_fp32_ = SoftmaxTestData<DeviceType::Cpu, TensorDataType::FP32>::Create(
                    "cpu_softmax_fp32", cpu_batch_size_, sequence_length_, vocab_size_, axis_ );
            }
            return cpu_fp32_;
        }

        SoftmaxTestData<DeviceType::Cuda, TensorDataType::FP32>& CudaFp32Data()
        {
            if (!cuda_fp32_.softmax_module)
            {
                cuda_fp32_ = SoftmaxTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                    "cuda_softmax_fp32", batch_size_, sequence_length_, vocab_size_, axis_ );
            }
            return cuda_fp32_;
        }

        SoftmaxTestData<DeviceType::Cpu, TensorDataType::FP32>& TrainingCpuFp32Data()
        {
            if (!training_cpu_fp32_.softmax_module)
            {
                training_cpu_fp32_ = SoftmaxTestData<DeviceType::Cpu, TensorDataType::FP32>::Create(
                    "cpu_softmax_fp32_training", cpu_batch_size_, sequence_length_, vocab_size_, axis_, true );
            }
            return training_cpu_fp32_;
        }

        SoftmaxTestData<DeviceType::Cuda, TensorDataType::FP32>& TrainingCudaFp32Data()
        {
            if (!training_cuda_fp32_.softmax_module)
            {
                training_cuda_fp32_ = SoftmaxTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                    "cuda_softmax_fp32_training", batch_size_, sequence_length_, vocab_size_, axis_, true );
            }
            return training_cuda_fp32_;
        }

        SoftmaxTestData<DeviceType::Cpu, TensorDataType::FP32>& ContextCpuFp32Data()
        {
            if (!context_cpu_fp32_.softmax_module)
            {
                auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
                context_cpu_fp32_ = SoftmaxTestData<DeviceType::Cpu, TensorDataType::FP32>::CreateWithContext(
                    "cpu_context_softmax_fp32", cpu_batch_size_, sequence_length_, vocab_size_, ctx, axis_ );
            }
            return context_cpu_fp32_;
        }

        // Test parameters
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t vocab_size_{ 0 };
        int64_t axis_{ -1 };

        // Test data
        SoftmaxTestData<DeviceType::Cpu, TensorDataType::FP32> cpu_fp32_;
        SoftmaxTestData<DeviceType::Cpu, TensorDataType::FP32> context_cpu_fp32_;
        SoftmaxTestData<DeviceType::Cpu, TensorDataType::FP32> training_cpu_fp32_;

        SoftmaxTestData<DeviceType::Cuda, TensorDataType::FP32> cuda_fp32_;
        SoftmaxTestData<DeviceType::Cuda, TensorDataType::FP32> training_cuda_fp32_;
    };

    // --------------------
    // Helper test utilities
    // --------------------

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestGetName( const SoftmaxTestData<TDevice, TPrecision>& data, const std::string& expected_name )
    {
        EXPECT_EQ( data.softmax_module->getName(), expected_name );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestParameterCount( const SoftmaxTestData<TDevice, TPrecision>& data, size_t expected_count )
    {
        EXPECT_EQ( data.softmax_module->parameterCount(), expected_count );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestTrainingMode( const SoftmaxTestData<TDevice, TPrecision>& data, bool expected_mode )
    {
        EXPECT_EQ( data.softmax_module->isTraining(), expected_mode );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestPrint( const SoftmaxTestData<TDevice, TPrecision>& data, const std::string& expected_substring )
    {
        std::string output = data.softmax_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
        // Confirm axis and device are present in toString()
        std::string axis_text = "Axis: " + std::to_string( data.axis );
        EXPECT_NE( output.find( axis_text ), std::string::npos );
        std::string device_text = (TDevice == DeviceType::Cuda) ? "Device: CUDA" : "Device: CPU";
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
    }

    // Forward test: validate shapes and (for CPU) normalization along last axis
    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestForward( const SoftmaxTestData<TDevice, TPrecision>& data )
    {
        using MR = MemoryResourceType<TDevice>;
        using TensorType = Tensor<TPrecision, MR>;

        TensorType input( std::string( TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU" ), data.shape );
        TensorType output( std::string( TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU" ), data.shape );

        // Fill host-accessible tensors directly (CPU). For CUDA create host tensor and copy.
        if constexpr (TDevice == DeviceType::Cpu)
        {
            // Fill using host pointer
            std::mt19937 rng( 1234 );
            std::uniform_real_distribution<float> dist( -5.0f, 5.0f );

            auto ptr = input.data();
            for (size_t i = 0; i < input.size(); ++i)
            {
                ptr[i] = dist( rng );
            }

            data.softmax_module->forward( input, output );
            EXPECT_EQ( output.size(), input.size() );

            // Verify normalization along last axis (vocab dim)
            const auto& shape = output.shape();
            ASSERT_GE( shape.size(), 1u );
            size_t last_dim = static_cast<size_t>( shape.back() );

            // compute product of leading dims
            size_t outer = 1;
            for (size_t i = 0; i + 1 < shape.size(); ++i) outer *= static_cast<size_t>( shape[i] );

            for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx)
            {
                float sum = 0.0f;
                for (size_t v = 0; v < last_dim; ++v)
                {
                    // compute linear index: outer_idx * last_dim + v (row-major after flatten)
                    size_t idx = outer_idx * last_dim + v;
                    sum += static_cast<float>( output.data()[idx] );
                }
                EXPECT_NEAR( sum, 1.0f, 1e-4f );
            }
        }
        else
        {
            // For CUDA: create host FP32 tensor, fill, copy to device, run forward, copy back and basic checks
            using HostTensorType = Tensor<TensorDataType::FP32, CpuMemoryResource>;
            HostTensorType host_input( "CPU", data.shape );
            HostTensorType host_output( "CPU", data.shape );

            std::mt19937 rng( 1234 );
            std::uniform_real_distribution<float> dist( -5.0f, 5.0f );
            for (size_t i = 0; i < host_input.size(); ++i) 
                host_input.data()[i] = dist( rng );

            // Create device tensors
            Tensor<TPrecision, CudaDeviceMemoryResource> device_input( "CUDA:0", data.shape );
            Tensor<TPrecision, CudaDeviceMemoryResource> device_output( "CUDA:0", data.shape );

            copy( host_input, device_input );

            data.softmax_module->forward( device_input, device_output );

            // Copy back for basic sanity check (size equality)
            host_output = toHost<TensorDataType::FP32>( device_output );

            EXPECT_EQ( host_output.size(), host_input.size() );
        }
    }

    // Device type check via execution context provided to module
    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestDeviceType( const SoftmaxTestData<TDevice, TPrecision>& data )
    {
        ASSERT_NE( data.exec_context, nullptr );
        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // CPU Tests FP32
    TEST_F( SoftmaxTests, Cpu_Fp32_TestName )
    {
        TestGetName<DeviceType::Cpu, TensorDataType::FP32>( CpuFp32Data(), "cpu_softmax_fp32" );
    }

    TEST_F( SoftmaxTests, Cpu_Fp32_ParameterCount )
    {
        TestParameterCount<DeviceType::Cpu, TensorDataType::FP32>( CpuFp32Data(), 0 );
    }

    TEST_F( SoftmaxTests, Cpu_Fp32_TestForward )
    {
        TestForward<DeviceType::Cpu, TensorDataType::FP32>( CpuFp32Data() );
    }

    TEST_F( SoftmaxTests, Cpu_Fp32_TestPrint )
    {
        TestPrint<DeviceType::Cpu, TensorDataType::FP32>( CpuFp32Data(), "Softmax: cpu_softmax_fp32" );
    }

    TEST_F( SoftmaxTests, Cpu_Fp32_TrainingMode )
    {
        TestTrainingMode<DeviceType::Cpu, TensorDataType::FP32>( CpuFp32Data(), false );
    }

    TEST_F( SoftmaxTests, Cpu_Fp32_DeviceType )
    {
        TestDeviceType<DeviceType::Cpu, TensorDataType::FP32>( CpuFp32Data() );
    }

    // CPU Training Mode
    TEST_F( SoftmaxTests, Cpu_Training_Fp32_TrainingMode )
    {
        TestTrainingMode<DeviceType::Cpu, TensorDataType::FP32>( TrainingCpuFp32Data(), true );
    }

    // CUDA Tests FP32
    TEST_F( SoftmaxTests, Cuda_Fp32_TestName )
    {
        TestGetName<DeviceType::Cuda, TensorDataType::FP32>( CudaFp32Data(), "cuda_softmax_fp32" );
    }

    TEST_F( SoftmaxTests, Cuda_Fp32_ParameterCount )
    {
        TestParameterCount<DeviceType::Cuda, TensorDataType::FP32>( CudaFp32Data(), 0 );
    }

    TEST_F( SoftmaxTests, Cuda_Fp32_TestForward )
    {
        TestForward<DeviceType::Cuda, TensorDataType::FP32>( CudaFp32Data() );
    }

    TEST_F( SoftmaxTests, Cuda_Fp32_TestPrint )
    {
        TestPrint<DeviceType::Cuda, TensorDataType::FP32>( CudaFp32Data(), "Softmax: cuda_softmax_fp32" );
    }

    TEST_F( SoftmaxTests, Cuda_Fp32_TrainingMode )
    {
        TestTrainingMode<DeviceType::Cuda, TensorDataType::FP32>( CudaFp32Data(), false );
    }

    TEST_F( SoftmaxTests, Cuda_Fp32_DeviceType )
    {
        TestDeviceType<DeviceType::Cuda, TensorDataType::FP32>( CudaFp32Data() );
    }

    // CUDA Training Mode
    TEST_F( SoftmaxTests, Cuda_Training_Fp32_TrainingMode )
    {
        TestTrainingMode<DeviceType::Cuda, TensorDataType::FP32>( TrainingCudaFp32Data(), true );
    }

    // Context Construction Tests
    TEST_F( SoftmaxTests, Context_Cpu_Fp32_DeviceType )
    {
        TestDeviceType<DeviceType::Cpu, TensorDataType::FP32>( ContextCpuFp32Data() );
    }

    TEST_F( SoftmaxTests, Context_Cpu_Fp32_Forward )
    {
        TestForward<DeviceType::Cpu, TensorDataType::FP32>( ContextCpuFp32Data() );
    }

    // Edge case tests - minimal and larger sizes
    TEST_F( SoftmaxTests, Cpu_Fp32_EdgeCases )
    {
        // minimal
        auto cfg = SoftmaxConfig().withAxis( -1 ).withName( "minimal_softmax" );
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto module = std::make_shared<Softmax<DeviceType::Cpu, TensorDataType::FP32>>( cfg, ctx );

        shape_t minimal_shape = { static_cast<dim_t>(1), static_cast<dim_t>(1), static_cast<dim_t>(8) };
        Tensor<TensorDataType::FP32, CpuMemoryResource> in( "CPU", minimal_shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> out( "CPU", minimal_shape );

        EXPECT_NO_THROW( module->forward( in, out ) );
        EXPECT_EQ( out.size(), 8 );

        // large
        cfg = SoftmaxConfig().withAxis( -1 ).withName( "large_softmax" );
        shape_t large_shape = { static_cast<dim_t>(2), static_cast<dim_t>(2), static_cast<dim_t>(1024) };
        auto module2 = std::make_shared<Softmax<DeviceType::Cpu, TensorDataType::FP32>>( cfg, ctx );
        
        Tensor<TensorDataType::FP32, CpuMemoryResource> in2( "CPU", large_shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> out2( "CPU", large_shape );

        EXPECT_NO_THROW( module2->forward( in2, out2 ) );
        EXPECT_EQ( out2.size(), 4096 );
    }

    TEST_F( SoftmaxTests, Cuda_Fp32_EdgeCases )
    {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto cfg = SoftmaxConfig().withAxis( -1 ).withName( "minimal_softmax_cuda" );

        auto module = std::make_shared<Softmax<DeviceType::Cuda, TensorDataType::FP32>>( cfg, ctx );

        shape_t minimal_shape = { static_cast<dim_t>(1), static_cast<dim_t>(1), static_cast<dim_t>(8) };
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> in( "CUDA:0", minimal_shape );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out( "CUDA:0", minimal_shape );

        EXPECT_NO_THROW( module->forward( in, out ) );
        EXPECT_EQ( out.size(), 8 );

        shape_t large_shape = { static_cast<dim_t>(2), static_cast<dim_t>(2), static_cast<dim_t>(1024) };
        cfg = SoftmaxConfig().withAxis( -1 ).withName( "large_softmax_cuda" );
        auto module2 = std::make_shared<Softmax<DeviceType::Cuda, TensorDataType::FP32>>( cfg, ctx );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> in2( "CUDA:0", large_shape );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out2( "CUDA:0", large_shape );

        EXPECT_NO_THROW( module2->forward( in2, out2 ) );
        EXPECT_EQ( out2.size(), 4096 );
    }

    // Axis Tests
    TEST_F( SoftmaxTests, Cpu_Fp32_DifferentAxes )
    {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        shape_t test_shape = { static_cast<dim_t>(2), static_cast<dim_t>(3), static_cast<dim_t>(4) };
        Tensor<TensorDataType::FP32, CpuMemoryResource> input( "CPU", test_shape );

        SoftmaxConfig cfg = SoftmaxConfig();
        cfg.withName( "axis0" ).withAxis( 0 );

        auto axis0 = std::make_shared<Softmax<DeviceType::Cpu, TensorDataType::FP32>>( cfg, ctx );
        Tensor<TensorDataType::FP32, CpuMemoryResource> out0( "CPU", test_shape );
        EXPECT_NO_THROW( axis0->forward( input, out0 ) );

        cfg = SoftmaxConfig();
        cfg.withName("axis1").withAxis(1);
        auto axis1 = std::make_shared<Softmax<DeviceType::Cpu, TensorDataType::FP32>>( cfg, ctx );
        Tensor<TensorDataType::FP32, CpuMemoryResource> out1( "CPU", test_shape );
        EXPECT_NO_THROW( axis1->forward( input, out1 ) );

        cfg.withName( "axis2" ).withAxis( 2 );
        auto axis2 = std::make_shared<Softmax<DeviceType::Cpu, TensorDataType::FP32>>( cfg, ctx );
        Tensor<TensorDataType::FP32, CpuMemoryResource> out2( "CPU", test_shape );
        EXPECT_NO_THROW( axis2->forward( input, out2 ) );
    }

    TEST_F( SoftmaxTests, Cuda_Fp32_DifferentAxes )
    {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        shape_t test_shape = { static_cast<dim_t>(2), static_cast<dim_t>(3), static_cast<dim_t>(4) };
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> input( "CUDA:0", test_shape );
        auto cfg = SoftmaxConfig();
        cfg.withName( "axis0_cuda" ).withAxis( 0 );
        
        auto axis0 = std::make_shared<Softmax<DeviceType::Cuda, TensorDataType::FP32>>( cfg, ctx );
        
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out0( "CUDA:0", test_shape );
        EXPECT_NO_THROW( axis0->forward( input, out0 ) );

        cfg.withName( "axis1_cuda" ).withAxis( 1 );
        auto axis1 = std::make_shared<Softmax<DeviceType::Cuda, TensorDataType::FP32>>( cfg, ctx );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out1( "CUDA:0", test_shape );
        EXPECT_NO_THROW( axis1->forward( input, out1 ) );

        cfg.withName( "axis2_cuda" ).withAxis( 2 );
        auto axis2 = std::make_shared<Softmax<DeviceType::Cuda, TensorDataType::FP32>>( cfg, ctx );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out2( "CUDA:0", test_shape );
        EXPECT_NO_THROW( axis2->forward( input, out2 ) );
    }

    // CPU-CUDA equivalence (small shape)
    TEST_F( SoftmaxTests, CpuCuda_Forward_Output_Equivalence )
    {
        auto cpu = CpuFp32Data();
        auto cuda = CudaFp32Data();

        // small shape for quick comparison
        shape_t test_shape = { static_cast<dim_t>(2), static_cast<dim_t>(4), static_cast<dim_t>(8) };

        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( "CPU", test_shape );
        
        for (size_t i = 0; i < host_input.size(); ++i)
            host_input.data()[i] = static_cast<float>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input.size()) );

        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_output( "CPU", test_shape );
        cpu.softmax_module->forward( host_input, cpu_output );

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> device_input( "CUDA:0", test_shape );
        copy( host_input, device_input );

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> device_output( "CUDA:0", test_shape );
        cuda.softmax_module->forward( device_input, device_output );

        Tensor<TensorDataType::FP32, CpuMemoryResource> cuda_output_host( "CPU", test_shape );
        cuda_output_host = toHost<TensorDataType::FP32>( device_output );

        const float epsilon = 1e-4f;
        bool all_equal = true;
        for (size_t i = 0; i < cpu_output.size(); ++i)
        {
            float a = cpu_output.data()[i];
            float b = cuda_output_host.data()[i];
            float diff = std::abs( a - b );
            if (diff > epsilon)
            {
                std::cout << "Difference at index " << i << ": CPU=" << a << ", CUDA=" << b << ", diff=" << diff << std::endl;
                all_equal = false;
            }
        }

        EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
    }
}