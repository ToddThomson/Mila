#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <random>
#include <iostream>
#include <cstdint>
#include <stdexcept>

import Mila;

namespace Modules::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == DeviceType::Cuda,
        CudaDeviceMemoryResource,
        CpuMemoryResource>;

    // CPU-specialized test-data and helpers (keeps templates for reuse but tests instantiate CPU)
    template<DeviceType TDevice, TensorDataType TPrecision = TensorDataType::FP32>
    struct LinearTestData
    {
        shape_t input_shape;
        shape_t output_shape;
        std::shared_ptr<Linear<TDevice, TPrecision>> linear_module;
        LinearConfig config;
        std::shared_ptr<ExecutionContext<TDevice>> exec_context;

        static LinearTestData Create(
            int64_t batch_size,
            int64_t sequence_length,
            int64_t input_features,
            int64_t output_features,
            bool has_bias = true )
        {
            LinearTestData data;
            data.input_shape = { static_cast<dim_t>(batch_size), static_cast<dim_t>(sequence_length), static_cast<dim_t>(input_features) };
            data.output_shape = { static_cast<dim_t>(batch_size), static_cast<dim_t>(sequence_length), static_cast<dim_t>(output_features) };

            data.config = LinearConfig( static_cast<size_t>(input_features), static_cast<size_t>(output_features) );
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

            data.linear_module = std::make_shared<Linear<TDevice, TPrecision>>( data.exec_context, data.config );

            return data;
        }

        static LinearTestData CreateWithContext(
            std::shared_ptr<ExecutionContext<TDevice>> context,
            int64_t batch_size,
            int64_t sequence_length,
            int64_t input_features,
            int64_t output_features,
            bool has_bias = true )
        {
            LinearTestData data;
            data.input_shape = { static_cast<dim_t>(batch_size), static_cast<dim_t>(sequence_length), static_cast<dim_t>(input_features) };
            data.output_shape = { static_cast<dim_t>(batch_size), static_cast<dim_t>(sequence_length), static_cast<dim_t>(output_features) };

            data.config = LinearConfig( static_cast<size_t>(input_features), static_cast<size_t>(output_features) );
            data.config
                .withBias( has_bias )
                .withName( "test_linear_context" );

            data.exec_context = context;
            data.linear_module = std::make_shared<Linear<TDevice, TPrecision>>( data.exec_context, data.config );

            return data;
        }

        LinearTestData() : config( 1, 1 )
        {
        }
    };

    class LinearCpuTests : public ::testing::Test
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
            cpu_float_data_.linear_module.reset();
            cpu_no_bias_float_data_.linear_module.reset();
            context_cpu_float_data_.linear_module.reset();
        }

        LinearTestData<DeviceType::Cpu, TensorDataType::FP32>& CpuFp32Data()
        {
            if (!cpu_float_data_.linear_module)
            {
                cpu_float_data_ = LinearTestData<DeviceType::Cpu, TensorDataType::FP32>::Create(
                    batch_size_, sequence_length_, input_features_, output_features_ );
            }
            return cpu_float_data_;
        }

        LinearTestData<DeviceType::Cpu, TensorDataType::FP32>& CpuNoBiasFp32Data()
        {
            if (!cpu_no_bias_float_data_.linear_module)
            {
                cpu_no_bias_float_data_ = LinearTestData<DeviceType::Cpu, TensorDataType::FP32>::Create(
                    batch_size_, sequence_length_, input_features_, output_features_, false );
            }
            return cpu_no_bias_float_data_;
        }

        LinearTestData<DeviceType::Cpu, TensorDataType::FP32>& ContextCpuFp32Data()
        {
            if (!context_cpu_float_data_.linear_module)
            {
                auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
                context_cpu_float_data_ = LinearTestData<DeviceType::Cpu, TensorDataType::FP32>::CreateWithContext(
                    ctx, batch_size_, sequence_length_, input_features_, output_features_ );
            }
            return context_cpu_float_data_;
        }

        int64_t batch_size_{ 0 };
        int64_t sequence_length_{ 0 };
        int64_t input_features_{ 0 };
        int64_t output_features_{ 0 };

        LinearTestData<DeviceType::Cpu, TensorDataType::FP32> cpu_float_data_;
        LinearTestData<DeviceType::Cpu, TensorDataType::FP32> cpu_no_bias_float_data_;
        LinearTestData<DeviceType::Cpu, TensorDataType::FP32> context_cpu_float_data_;
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

        if constexpr (TDevice == DeviceType::Cpu)
        {
            // Fill CPU tensor directly
            std::mt19937 rng( 1234 );
            std::uniform_real_distribution<float> dist( -1.0f, 1.0f );
            auto ptr = input.data();
            for (size_t i = 0; i < input.size(); ++i) ptr[i] = static_cast<typename TensorHostTypeMap<TPrecision>::host_type>( dist( rng ) );
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

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestDeviceType( const LinearTestData<TDevice, TPrecision>& data )
    {
        ASSERT_NE( data.exec_context, nullptr );
        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    void TestEdgeCases()
    {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        LinearConfig minimal_config( 1, 1 );
        minimal_config.withName( "minimal_linear" );

        auto minimal_linear = std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(  ctx, minimal_config );

        Tensor<TensorDataType::FP32, CpuMemoryResource> minimal_input( "CPU", shape_t{1, 1, 1} );
        Tensor<TensorDataType::FP32, CpuMemoryResource> minimal_output( "CPU", shape_t{1, 1, 1} );

        minimal_input.data()[0] = 1.0f;

        EXPECT_NO_THROW( minimal_linear->forward( minimal_input, minimal_output ) );

        LinearConfig large_config( 128, 64 );
        large_config.withName( "large_linear" );

        auto large_linear = std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>( ctx, large_config );

        Tensor<TensorDataType::FP32, CpuMemoryResource> large_input( "CPU", shape_t{2, 4, 128} );
        Tensor<TensorDataType::FP32, CpuMemoryResource> large_output( "CPU", shape_t{2, 4, 64} );

        for (size_t i = 0; i < large_input.size(); ++i)
        {
            large_input.data()[i] = static_cast<float>( i ) / large_input.size();
        }

        EXPECT_NO_THROW( large_linear->forward( large_input, large_output ) );
        EXPECT_EQ( large_output.size(), 512 );
    }

    TEST_F( LinearCpuTests, Cpu_Fp32_ParameterCount )
    {
        TestParameterCount( CpuFp32Data() );
    }

    TEST_F( LinearCpuTests, Cpu_Fp32_Forward )
    {
        TestForward( CpuFp32Data() );
    }

    TEST_F( LinearCpuTests, Cpu_Fp32_ToString )
    {
        TestToString( CpuFp32Data() );
    }

    TEST_F( LinearCpuTests, Cpu_Fp32_GetWeight )
    {
        TestGetWeight( CpuFp32Data() );
    }

    TEST_F( LinearCpuTests, Cpu_Fp32_GetBias )
    {
        TestGetBias( CpuFp32Data() );
    }

    TEST_F( LinearCpuTests, Cpu_Fp32_HasBias )
    {
        TestHasBias( CpuFp32Data() );
    }

    TEST_F( LinearCpuTests, Cpu_Fp32_DeviceType )
    {
        TestDeviceType( CpuFp32Data() );
    }

    TEST_F( LinearCpuTests, Cpu_NoBias_Fp32_ParameterCount )
    {
        TestParameterCount( CpuNoBiasFp32Data() );
    }

    TEST_F( LinearCpuTests, Cpu_NoBias_Fp32_GetBias )
    {
        TestGetBias( CpuNoBiasFp32Data() );
    }

    TEST_F( LinearCpuTests, Cpu_NoBias_Fp32_HasBias )
    {
        TestHasBias( CpuNoBiasFp32Data() );
    }

    TEST_F( LinearCpuTests, Cpu_NoBias_Fp32_Forward )
    {
        TestForward( CpuNoBiasFp32Data() );
    }

    TEST_F( LinearCpuTests, Context_Cpu_Fp32_Forward )
    {
        TestForward( ContextCpuFp32Data() );
    }

    TEST_F( LinearCpuTests, Context_Cpu_Fp32_DeviceType )
    {
        TestDeviceType( ContextCpuFp32Data() );
    }

    TEST_F( LinearCpuTests, EdgeCases )
    {
        TestEdgeCases();
    }

    TEST_F( LinearCpuTests, Constructor_ExecutionContextConstruction )
    {
        LinearConfig config( 16, 32 );
        config.withName( "validation_test" );

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        EXPECT_NO_THROW( (Linear<DeviceType::Cpu, TensorDataType::FP32>( ctx, config )) );
    }

    TEST_F( LinearCpuTests, Constructor_InvalidConfig )
    {
        LinearConfig invalid_config( 0, 32 );
        invalid_config.withName( "invalid_test" );

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        EXPECT_THROW( (Linear<DeviceType::Cpu, TensorDataType::FP32>( ctx, invalid_config )), std::invalid_argument );
    }
}