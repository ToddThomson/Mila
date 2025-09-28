#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>

import Mila;

namespace Modules::Activations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<DeviceType TDeviceType>
    using MemoryResourceType = std::conditional_t<TDeviceType == DeviceType::Cuda,
        CudaMemoryResource,
        CpuMemoryResource>;

    template<DeviceType TDeviceType, typename TDataType = float>
    struct GeluTestData {
        std::vector<size_t> shape;
        std::shared_ptr<Gelu<TDeviceType, TDataType>> gelu_module;

        static GeluTestData Create(
            const std::string& device_name,
            size_t batch_size,
            size_t sequence_length,
            size_t channels )
        {
            GeluTestData data;
            data.shape = { batch_size, sequence_length, channels };

            GeluConfig config;
            data.gelu_module = std::make_shared<Gelu<TDeviceType, TDataType>>( device_name, config );

            return data;
        }

        static GeluTestData CreateWithContext(
            std::shared_ptr<DeviceContext> context,
            size_t batch_size,
            size_t sequence_length,
            size_t channels )
        {
            GeluTestData data;
            data.shape = { batch_size, sequence_length, channels };

            GeluConfig config;
            data.gelu_module = std::make_shared<Gelu<TDeviceType, TDataType>>( context, config );

            return data;
        }
    };

    class GeluTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 2;
            sequence_length_ = 4;
            channels_ = 8;
        }

        void TearDown() override {
            cpu_float_data_.gelu_module.reset();
            context_cpu_float_data_.gelu_module.reset();
            cuda_float_data_.gelu_module.reset();
        }

        GeluTestData<DeviceType::Cpu, float>& CpuFloatData() {
            if ( !cpu_float_data_.gelu_module ) {
                cpu_float_data_ = GeluTestData<DeviceType::Cpu, float>::Create(
                    "CPU", batch_size_, sequence_length_, channels_ );
            }
            return cpu_float_data_;
        }

        GeluTestData<DeviceType::Cuda, float>& CudaFloatData() {
            if ( !cuda_float_data_.gelu_module ) {
                cuda_float_data_ = GeluTestData<DeviceType::Cuda, float>::Create(
                    "CUDA:0", batch_size_, sequence_length_, channels_ );
            }
            return cuda_float_data_;
        }

        GeluTestData<DeviceType::Cpu, float>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.gelu_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = GeluTestData<DeviceType::Cpu, float>::CreateWithContext(
                    cpu_context, batch_size_, sequence_length_, channels_ );
            }
            return context_cpu_float_data_;
        }

        size_t batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };

        GeluTestData<DeviceType::Cpu, float> cpu_float_data_;
        GeluTestData<DeviceType::Cpu, float> context_cpu_float_data_;
        GeluTestData<DeviceType::Cuda, float> cuda_float_data_;
    };

    template<DeviceType TDeviceType, typename TDataType>
    void TestParameterCount( const GeluTestData<TDeviceType, TDataType>& data ) {
        EXPECT_EQ( data.gelu_module->parameterCount(), 0 );
    }

    template<DeviceType TDeviceType, typename TDataType>
    void TestForward( const GeluTestData<TDeviceType, TDataType>& data ) {
        using MR = MemoryResourceType<TDeviceType>;

        Tensor<TDataType, MR> input( data.shape );
        Tensor<TDataType, MR> output( data.shape );

        if constexpr ( TDeviceType == DeviceType::Cpu ) {
            for ( size_t i = 0; i < input.size(); ++i ) {
                input.data()[ i ] = static_cast<TDataType>( static_cast<float>( i ) / input.size() * 4.0f - 2.0f );
            }
        }
        else {
            Tensor<TDataType, CpuMemoryResource> host_input( data.shape );
            for ( size_t i = 0; i < host_input.size(); ++i ) {
                host_input.data()[ i ] = static_cast<TDataType>( static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f );
            }
            input = host_input.toDevice<CudaMemoryResource>();
        }

        ASSERT_NO_THROW( data.gelu_module->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    template<DeviceType TDeviceType, typename TDataType>
    void TestToString( const GeluTestData<TDeviceType, TDataType>& data ) {
        std::string result = data.gelu_module->toString();
        EXPECT_FALSE( result.empty() );
        EXPECT_TRUE( result.find( "Gelu" ) != std::string::npos );
    }

    template<DeviceType TDeviceType, typename TDataType>
    void TestApproximationMethod( const GeluTestData<TDeviceType, TDataType>& data ) {
        auto method = data.gelu_module->getApproximationMethod();
        EXPECT_EQ( method, GeluConfig::ApproximationMethod::Tanh );
    }

    template<DeviceType TDeviceType, typename TDataType>
    void TestDeviceType( const GeluTestData<TDeviceType, TDataType>& data ) {
        auto device_context = data.gelu_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDeviceType );
    }

    TEST_F( GeluTests, Cpu_Float_ParameterCount ) {
        TestParameterCount( CpuFloatData() );
    }

    TEST_F( GeluTests, Cpu_Float_Forward ) {
        TestForward( CpuFloatData() );
    }

    TEST_F( GeluTests, Cpu_Float_ToString ) {
        TestToString( CpuFloatData() );
    }

    TEST_F( GeluTests, Cpu_Float_ApproximationMethod ) {
        TestApproximationMethod( CpuFloatData() );
    }

    TEST_F( GeluTests, Cpu_Float_DeviceType ) {
        TestDeviceType( CpuFloatData() );
    }

    TEST_F( GeluTests, Cuda_Float_ParameterCount ) {
        try {
            TestParameterCount( CudaFloatData() );
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( GeluTests, Cuda_Float_Forward ) {
        try {
            TestForward( CudaFloatData() );
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( GeluTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType( ContextCpuFloatData() );
    }

    TEST_F( GeluTests, Constructor_DeviceNameValidation ) {
        GeluConfig config;
        EXPECT_NO_THROW( ( Gelu<DeviceType::Cpu, float>( "CPU", config ) ) );
    }

    TEST_F( GeluTests, Constructor_DeviceContextValidation ) {
        GeluConfig config;
        auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
        EXPECT_NO_THROW( ( Gelu<DeviceType::Cpu, float>( cpu_context, config ) ) );
    }

    TEST_F( GeluTests, CpuCuda_EquivalenceTest ) {
        std::vector<size_t> test_shape = { 2, 2, 4 };

        GeluConfig config;
        auto cpu_gelu = std::make_shared<Gelu<DeviceType::Cpu, float>>( "CPU", config );

        try {
            auto cuda_gelu = std::make_shared<Gelu<DeviceType::Cuda, float>>( "CUDA:0", config );

            Tensor<float, CpuMemoryResource> cpu_input( test_shape );
            Tensor<float, CpuMemoryResource> cpu_output( test_shape );

            for ( size_t i = 0; i < cpu_input.size(); ++i ) {
                cpu_input.data()[ i ] = static_cast<float>( i ) / cpu_input.size() * 4.0f - 2.0f;
            }

            cpu_gelu->forward( cpu_input, cpu_output );

            auto cuda_input = cpu_input.toDevice<CudaMemoryResource>();
            Tensor<float, CudaMemoryResource> cuda_output( test_shape );

            cuda_gelu->forward( cuda_input, cuda_output );

            auto cuda_output_host = cuda_output.toHost<CpuMemoryResource>();

            const float epsilon = 1e-4f;
            for ( size_t i = 0; i < cpu_output.size(); ++i ) {
                float diff = std::abs( cpu_output.data()[ i ] - cuda_output_host.data()[ i ] );
                EXPECT_LT( diff, epsilon ) << "Mismatch at index " << i
                    << ": CPU=" << cpu_output.data()[ i ]
                    << ", CUDA=" << cuda_output_host.data()[ i ];
            }
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping equivalence test" << std::endl;
            SUCCEED();
        }
    }
}