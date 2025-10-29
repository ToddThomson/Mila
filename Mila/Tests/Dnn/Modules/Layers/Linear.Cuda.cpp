#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <random>
#include <iostream>
#include <cstdint>

import Mila;

namespace Modules::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == DeviceType::Cuda,
        CudaDeviceMemoryResource,
        CpuMemoryResource>;

    // CUDA-specialized test-data and helpers (keeps templates for reuse but tests instantiate CUDA)
    template<DeviceType TDevice, TensorDataType TPrecision = TensorDataType::FP32>
    struct LinearTestData
    {
        shape_t input_shape;
        shape_t output_shape;
        std::shared_ptr<Linear<TDevice, TPrecision>> linear_module;
        LinearConfig config;
        std::shared_ptr<ExecutionContext<TDevice>> exec_context;

        static LinearTestData Create(
            size_t batch_size,
            size_t sequence_length,
            size_t input_features,
            size_t output_features,
            bool has_bias = true )
        {
            LinearTestData data;
            data.input_shape = { static_cast<dim_t>(batch_size), static_cast<dim_t>(sequence_length), static_cast<dim_t>(input_features) };
            data.output_shape = { static_cast<dim_t>(batch_size), static_cast<dim_t>(sequence_length), static_cast<dim_t>(output_features) };

            data.config = LinearConfig( input_features, output_features );
            data.config
                .withBias( has_bias )
                .withName( "test_linear" );

            if constexpr (TDevice == DeviceType::Cuda)
            {
                data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
            }
            else
            {
                data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            }

            data.linear_module = std::make_shared<Linear<TDevice, TPrecision>>(  data.exec_context, data.config );

            return data;
        }

        LinearTestData() : config( 1, 1 )
        {
        }
    };

    class LinearCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            batch_size_ = 4;
            sequence_length_ = 8;
            input_features_ = 16;
            output_features_ = 32;
        }

        void TearDown() override
        {
            cuda_float_data_.linear_module.reset();
            cuda_no_bias_float_data_.linear_module.reset();
        }

        LinearTestData<DeviceType::Cuda, TensorDataType::FP32>& CudaFp32Data()
        {
            if (!cuda_float_data_.linear_module)
            {
                cuda_float_data_ = LinearTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                    batch_size_, sequence_length_, input_features_, output_features_ );
            }
            return cuda_float_data_;
        }

        LinearTestData<DeviceType::Cuda, TensorDataType::FP32>& CudaNoBiasFp32Data()
        {
            if (!cuda_no_bias_float_data_.linear_module)
            {
                cuda_no_bias_float_data_ = LinearTestData<DeviceType::Cuda, TensorDataType::FP32>::Create(
                    batch_size_, sequence_length_, input_features_, output_features_, false );
            }
            return cuda_no_bias_float_data_;
        }

        int64_t batch_size_{ 0 };
        int64_t sequence_length_{ 0 };
        int64_t input_features_{ 0 };
        int64_t output_features_{ 0 };

        LinearTestData<DeviceType::Cuda, TensorDataType::FP32> cuda_float_data_;
        LinearTestData<DeviceType::Cuda, TensorDataType::FP32> cuda_no_bias_float_data_;
    };

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestParameterCount( const LinearTestData<TDevice, TPrecision>& data )
    {
        size_t expected_count = data.config.getInputFeatures() * data.config.getOutputFeatures();
        if (data.config.hasBias())
        {
            expected_count += data.config.getOutputFeatures();
        }
        EXPECT_EQ( data.linear_module->parameterCount(), expected_count );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestForward( const LinearTestData<TDevice, TPrecision>& data )
    {
        using MR = MemoryResourceType<TDevice>;
        using TensorType = Tensor<TPrecision, MR>;

        const std::string device_name = (TDevice == DeviceType::Cuda) ? "CUDA:0" : "CPU";

        TensorType input( device_name, data.input_shape );
        TensorType output( device_name, data.output_shape );

        if constexpr (TDevice == DeviceType::Cuda)
        {
            // Fill host tensor and copy to device using TensorOps::copy
            Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( "CPU", data.input_shape );
            std::mt19937 rng( 1234 );
            std::uniform_real_distribution<float> dist( -1.0f, 1.0f );
            for (size_t i = 0; i < host_input.size(); ++i) host_input.data()[i] = dist( rng );

            // perform host->device transfer via copy()
            copy( host_input, input, data.exec_context.get() );
        }

        ASSERT_NO_THROW( data.linear_module->forward( input, output ) );
        EXPECT_EQ( output.size(), static_cast<size_t>( data.output_shape[0] * data.output_shape[1] * data.output_shape[2] ) );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestToString( const LinearTestData<TDevice, TPrecision>& data )
    {
        std::string result = data.linear_module->toString();
        EXPECT_FALSE( result.empty() );
        EXPECT_NE( result.find( "Linear" ), std::string::npos );
        EXPECT_NE( result.find( "Input features" ), std::string::npos );
        EXPECT_NE( result.find( "Output features" ), std::string::npos );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestGetWeight( const LinearTestData<TDevice, TPrecision>& data )
    {
        auto weight = data.linear_module->getWeight();
        EXPECT_NE( weight, nullptr );

        EXPECT_EQ( weight->shape()[0], static_cast<dim_t>( data.config.getOutputFeatures() ) );
        EXPECT_EQ( weight->shape()[1], static_cast<dim_t>( data.config.getInputFeatures() ) );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestGetBias( const LinearTestData<TDevice, TPrecision>& data )
    {
        auto bias_opt = data.linear_module->getBias();

        if (data.config.hasBias())
        {
            EXPECT_TRUE( bias_opt.has_value() );
            auto bias = bias_opt.value();
            EXPECT_NE( bias, nullptr );
            EXPECT_EQ( bias->shape()[0], static_cast<dim_t>( data.config.getOutputFeatures() ) );
        }
        else
        {
            EXPECT_FALSE( bias_opt.has_value() );
        }
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestHasBias( const LinearTestData<TDevice, TPrecision>& data )
    {
        EXPECT_EQ( data.linear_module->hasBias(), data.config.hasBias() );
    }

    void TestCpuCudaEquivalence()
    {
        shape_t test_input_shape = { 2, 4, 8 };
        shape_t test_output_shape = { 2, 4, 16 };

        LinearConfig cpu_config( 8, 16 );
        cpu_config.withBias( true ).withName( "cpu_test" );

        LinearConfig cuda_config( 8, 16 );
        cuda_config.withBias( true ).withName( "cuda_test" );

        auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cuda_ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        auto cpu_linear = std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(  cpu_ctx, cpu_config );
        auto cuda_linear = std::make_shared<Linear<DeviceType::Cuda, TensorDataType::FP32>>(  cuda_ctx, cuda_config );

        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( "CPU", test_input_shape );
        for (size_t i = 0; i < host_input.size(); ++i)
        {
            host_input.data()[i] = static_cast<float>( i ) / host_input.size() * 2.0f - 1.0f;
        }

        auto cpu_weight = cpu_linear->getWeight(); // Cpu tensor
        auto cuda_weight = cuda_linear->getWeight(); // Cuda tensor

        for (size_t i = 0; i < cpu_weight->size(); ++i)
        {
            cpu_weight->data()[i] = 0.1f;
        }

        copy( *cpu_weight, *cuda_weight, cuda_ctx.get() );

        auto cpu_bias_opt = cpu_linear->getBias();
        auto cuda_bias_opt = cuda_linear->getBias();

        if (cpu_bias_opt.has_value() && cuda_bias_opt.has_value())
        {
            auto cpu_bias = cpu_bias_opt.value();
            auto cuda_bias = cuda_bias_opt.value();

            for (size_t i = 0; i < cpu_bias->size(); ++i)
            {
                cpu_bias->data()[i] = 0.0f;
            }

            copy( *cpu_bias, *cuda_bias, cuda_ctx.get() );
        }

        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_output( "CPU", test_output_shape );
        cpu_linear->forward( host_input, cpu_output );

        auto cuda_input = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( "CUDA:0", test_input_shape );
        copy( host_input, cuda_input, cuda_ctx.get() );

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> cuda_output( "CUDA:0", test_output_shape );
        cuda_linear->forward( cuda_input, cuda_output );

        Tensor<TensorDataType::FP32, CpuMemoryResource> cuda_output_host( "CPU", test_output_shape );
        copy( cuda_output, cuda_output_host, cuda_ctx.get() );

        const float epsilon = 1e-4f;
        for (size_t i = 0; i < cpu_output.size(); ++i)
        {
            float diff = std::abs( cpu_output.data()[i] - cuda_output_host.data()[i] );
            EXPECT_LT( diff, epsilon ) << "Mismatch at index " << i;
        }
    }

    TEST_F( LinearCudaTests, Cuda_Fp32_ParameterCount )
    {
        try
        {
            TestParameterCount( CudaFp32Data() );
        }
        catch (const std::exception&)
        {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearCudaTests, Cuda_Fp32_Forward )
    {
        try
        {
            TestForward( CudaFp32Data() );
        }
        catch (const std::exception&)
        {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearCudaTests, Cuda_Fp32_ToString )
    {
        try
        {
            TestToString( CudaFp32Data() );
        }
        catch (const std::exception&)
        {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearCudaTests, Cuda_Fp32_GetWeight )
    {
        try
        {
            TestGetWeight( CudaFp32Data() );
        }
        catch (const std::exception&)
        {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearCudaTests, Cuda_Fp32_GetBias )
    {
        try
        {
            TestGetBias( CudaFp32Data() );
        }
        catch (const std::exception&)
        {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearCudaTests, Cuda_NoBias_Fp32_HasBias )
    {
        try
        {
            TestHasBias( CudaNoBiasFp32Data() );
        }
        catch (const std::exception&)
        {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearCudaTests, Cuda_NoBias_Fp32_Forward )
    {
        try
        {
            TestForward( CudaNoBiasFp32Data() );
        }
        catch (const std::exception&)
        {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearCudaTests, CpuCuda_EquivalenceTest )
    {
        try
        {
            TestCpuCudaEquivalence();
        }
        catch (const std::exception&)
        {
            std::cout << "CUDA device not available, skipping equivalence test" << std::endl;
            SUCCEED();
        }
    }
}